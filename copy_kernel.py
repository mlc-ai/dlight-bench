# pylint: disable=missing-docstring
import os
from typing import Callable, Dict, List, Optional

import numpy as np

import tvm

import tvm.testing
from tvm import ir, te, tir, dlight
from tvm.contrib import nvcc, rpc, utils, ndk
from tvm.script import tir as T, ir as I

############ CUDA
# TARGET = tvm.target.Target("nvidia/geforce-rtx-3090")
# DEVICE = tvm.cuda(0)
# LOAD_V_SHARED = True
# LOAD_V_VEC = 8
# UNROLL = 256
# USE_REMOTE_CL = False

########### Rocm
TARGET = tvm.target.Target("rocm")
DEVICE = tvm.rocm(0)
LOAD_V_SHARED = True
LOAD_V_VEC = 8
UNROLL = 256
USE_REMOTE_CL = False

############ Vulkan
# TARGET = tvm.target.Target(
#     tvm.target.Target(
#         {
#             "kind": "vulkan",
#             "max_threads_per_block": 256,
#             "max_shared_memory_per_block": 32768,
#             "thread_warp_size": 1,
#             "supports_float16": 1,
#             "supports_int16": 1,
#             "supports_int8": 1,
#             "supports_int64": 1,
#             "supports_8bit_buffer": 1,
#             "supports_16bit_buffer": 1,
#             "supports_storage_buffer_storage_class": 1,
#         }
#     ),
#     host="llvm",
# )
# DEVICE = tvm.vulkan(0)
# LOAD_V_SHARED = True
# LOAD_V_VEC = 4
# UNROLL = 256
# USE_REMOTE_CL = False

############ Metal
# TARGET = tvm.target.Target("metal")
# DEVICE = tvm.metal(0)
# LOAD_V_SHARED = True
# LOAD_V_VEC = 4
# UNROLL = 256
# USE_REMOTE_CL = False

############ Mali
# tracker_host = "192.168.10.1"
# tracker_port = 9191
# key = "orangepi"

# TARGET = tvm.target.Target(
#     "opencl -device=mali", host="llvm -mtriple=aarch64-linux-gnu"
# )

# tracker = rpc.connect_tracker(tracker_host, tracker_port)
# remote = tracker.request(key, priority=0, session_timeout=0)
# DEVICE = remote.cl(0)
# LOAD_V_SHARED = False
# UNROLL = 8
# USE_REMOTE_CL = True

############ Android
# # Set to be address of tvm proxy.
# tracker_host = "0.0.0.0"
# tracker_port = 9090
# key = "android"

# # Change target configuration.
# # Run `adb shell cat /proc/cpuinfo` to find the arch.
# arch = "arm64"
# target = "llvm -mtriple=%s-linux-android" % arch
# TARGET = tvm.target.Target("opencl", host=target)

# tracker = rpc.connect_tracker(tracker_host, tracker_port)
# remote = tracker.request(key, priority=0, session_timeout=0)
# DEVICE = remote.cl(0)
# LOAD_V_SHARED = False
# UNROLL = 8
# USE_REMOTE_CL = True

############

N = 12288
K = 4096
# N = 15360
# K = 5120

cur_best = 1e6
cur_best_dict = None

# fmt: off

@T.prim_func
def copy_kernel(
    A_in: T.Buffer((T.int64(N), T.int64(K // 8)), "uint32"),
    A_out: T.Buffer((T.int64(N), T.int64(K // 8)), "uint32")
):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    for i, j in T.grid(T.int64(N), T.int64(K // 8)):
        with T.block("copy"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(A_in[v_i, v_j])
            T.writes(A_out[v_i, v_j])
            A_out[v_i, v_j] = A_in[v_i, v_j]


def get_copy_kernel_n_k(n, k):
    @T.prim_func
    def copy_kernel(
        A_in: T.Buffer((T.int64(N // n), T.int64(K // 8 // k), T.int64(n), T.int64(k)), "uint32"),
        A_out: T.Buffer((T.int64(N // n), T.int64(K // 8 // k), T.int64(n), T.int64(k)), "uint32")
    ):
        T.func_attr({"tir.noalias": T.bool(True)})
        # with T.block("root"):
        for i, j in T.grid(T.int64(N), T.int64(K // 8)):
            with T.block("copy"):
                v_i, v_j = T.axis.remap("SS", [i, j])
                T.reads(A_in[v_i // n, v_j // k, v_i % n, v_j % k])
                T.writes(A_out[v_i // n, v_j // k, v_i % n, v_j % k])
                A_out[v_i // n, v_j // k, v_i % n, v_j % k] = A_in[v_i // n, v_j // k, v_i % n, v_j % k]
    return copy_kernel

# fmt: on


def prepare_args(func: tir.PrimFunc, var_dict: Dict[str, int]):
    np.random.seed(0)
    args: List[np.ndarray] = []
    analyzer = tvm.arith.Analyzer()
    total_bytes = 0
    for param in func.params:
        buffer = func.buffer_map[param]
        shape = []
        for dim in buffer.shape:
            if isinstance(dim, tir.IntImm):
                shape.append(dim.value)
            elif isinstance(dim, tir.Var):
                assert dim.name in var_dict
                value = var_dict[dim.name]
                shape.append(value)
                analyzer.bind(dim, value)
            else:
                raise ValueError(f"Unknown shape: {buffer.shape}")
        if buffer.dtype == "uint32":
            np_array = np.random.randint(0, 2**16, size=shape).astype(buffer.dtype)
        else:
            np_array = np.random.uniform(high=0.01, size=shape).astype(buffer.dtype)
        total_bytes += np_array.size * np_array.itemsize
        tvm_array = tvm.nd.array(np_array, DEVICE)
        args.append(tvm_array)
    return args, total_bytes


def build_and_measure(func: tir.PrimFunc, args, total_bytes, config, run_only=False):
    rt_mod = tvm.build(func, target=TARGET)
    ################# Android or Mali
    if USE_REMOTE_CL:
        temp = utils.tempdir()
        path_dso_cl = temp.relpath("dev_lib_cl.so")
        rt_mod.export_library(path_dso_cl, ndk.create_shared)
        remote.upload(path_dso_cl)
        rt_mod = remote.load_module("dev_lib_cl.so")
    ################# Android or Mali
    rt_mod(*args)
    ret = args[-1]
    if not run_only:
        DEVICE.sync()
        time_eval = rt_mod.time_evaluator(
            rt_mod.entry_name,
            DEVICE,
            # number=20,
            # repeat=3,
            number=1,
            repeat=100,
            cache_flush_bytes=256 * 10**6,
        )
        DEVICE.sync()
        time = time_eval(*args).mean * 1e3
        DEVICE.sync()
        bandwidth = total_bytes / time / (1024**2)

        global cur_best, cur_best_dict
        if time < cur_best and config is not None:
            cur_best = time
            cur_best_dict = config
        print(
            f"Time (ms): {time:.6f}",
            f"Total Bytes (MB): {total_bytes / (1024**2):.6f}",
            f"Memory (GB/s): {bandwidth:.6f}",
            sep="\t",
        )
        print(
            f"Best time (ms): {cur_best:.6f}",
            f"Best Memory (GB/s): {total_bytes / cur_best / (1024**2):.6f}",
            f"Best config: {cur_best_dict}",
            sep="\t",
        )
    return ret


def export_source(mod):
    lib = tvm.build(mod, target=TARGET)
    source = lib.imported_modules[0].get_source()
    # remove content before extern "C"
    print(source[source.index('extern "C"') :])
    # with open("./gemv.cu", "w") as f:
    #     f.write(source)


def get_max_factor(n, factors):
    for factor in factors[::-1]:
        if n % factor == 0:
            return factor


def schedule(ret):
    # fmt: off
    def apply(mod):
        if K % VEC_LOAD != 0:
            return None
        sch = tir.Schedule(mod)
        block = sch.get_block("copy")
        s, r = sch.get_loops(block)
        bx, ty = sch.split(s, factors=[None, TS])
        r, tx, vec = sch.split(r, factors=[None, TR, VEC_LOAD])
        sch.reorder(bx, ty, tx, r, vec)
        sch.bind(bx, "blockIdx.x")
        sch.bind(ty, "threadIdx.y")
        sch.bind(tx, "threadIdx.x")
        sch.vectorize(vec)
        sch.unroll(r)
        return sch
    # fmt: on

    func_dict = dict()
    arg_dict = dict()
    for n in [1]:
        for k in [1]:
            func_dict[(n, k)] = copy_kernel
            arg_dict[(n, k)] = prepare_args(copy_kernel, {"n": n, "k": k})
    for all_thread in [1024, 512, 256, 128, 64, 32]:
        for TR in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
            for VEC_LOAD in [4, 2, 1]:
                TS = all_thread // TR
                if TS <= 0 or TR <= 0:
                    continue
                for (n, k), func in func_dict.items():
                    if N % n != 0 or K % k != 0:
                        continue
                    sch = apply(func)
                    if sch is None:
                        continue
                    try:
                        print("====")
                        print(
                            f"schedule 1:",
                            f"vec_load={VEC_LOAD}",
                            f"tr={TR}",
                            f"ts={TS}",
                            f"n={n}",
                            f"k={k}",
                            sep="\t",
                        )
                        # sch.mod.show(black_format=False)
                        ret_cur = build_and_measure(
                            sch.mod["main"],
                            arg_dict[(n, k)][0],
                            arg_dict[(n, k)][1],
                            config={
                                "VEC_LOAD": VEC_LOAD,
                                "TR": TR,
                                "TS": TS,
                                "n": n,
                                "k": k,
                            },
                        )
                        if n == 1:
                            tvm.testing.assert_allclose(
                                ret.numpy(),
                                ret_cur.numpy(),
                                rtol=5e-2,
                                atol=5e-2,
                            )
                        # export_source(sch.mod["main"])
                    except Exception as e:
                        print("Error", e)


def main():
    dlight_sch = dlight.gpu.Fallback().apply(copy_kernel, TARGET, False)
    dlight_mod = dlight_sch.mod
    # dlight_mod.show(black_format=False)
    print("dlight:")
    args = prepare_args(dlight_mod["main"], {"n": 256})
    ret = build_and_measure(dlight_mod["main"], *args, None, run_only=False)

    schedule(ret)


if __name__ == "__main__":
    main()

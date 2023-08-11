# Please save this file to dlight_bench/models and add
# `from .llama_2_7b_chat_hf_q4f16_1 import *` to dlight_bench/models/__init__.py
from dlight_bench import DlightBench
from tvm.script import tir as T


@T.prim_func
def reshape(var_A: T.handle, var_T_reshape: T.handle):
    T.func_attr({"global_symbol": "reshape", "tir.noalias": T.bool(True)})
    n = T.int64()
    A = T.match_buffer(var_A, (T.int64(1), n), "int32")
    T_reshape = T.match_buffer(var_T_reshape, (n,), "int32")
    # with T.block("root"):
    for ax0 in range(n):
        with T.block("T_reshape"):
            v_ax0 = T.axis.spatial(n, ax0)
            T.reads(A[T.int64(0), v_ax0 % n])
            T.writes(T_reshape[v_ax0])
            T_reshape[v_ax0] = A[T.int64(0), v_ax0 % n]


@T.prim_func
def fused_fused_decode1_take(
    lv: T.Buffer((32000, 512), "uint32"),
    lv1: T.Buffer((32000, 128), "float16"),
    p_lv: T.handle,
    p_output0: T.handle,
):
    T.func_attr(
        {"global_symbol": "fused_fused_decode1_take", "tir.noalias": T.bool(True)}
    )
    n = T.int32()
    lv_1 = T.match_buffer(p_lv, (n,), "int32")
    var_T_take_intermediate = T.match_buffer(p_output0, (n, 4096), "float16")
    # with T.block("root"):
    for ax0, ax1 in T.grid(n, 4096):
        with T.block("T_take"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(
                lv[lv_1[v_ax0], v_ax1 // 8], lv_1[v_ax0], lv1[lv_1[v_ax0], v_ax1 // 32]
            )
            T.writes(var_T_take_intermediate[v_ax0, v_ax1])
            var_T_take_intermediate[v_ax0, v_ax1] = (
                T.Cast(
                    "float16",
                    T.bitwise_and(
                        T.shift_right(
                            lv[lv_1[v_ax0], v_ax1 // 8],
                            T.Cast("uint32", v_ax1 % 8) * T.uint32(4),
                        ),
                        T.uint32(15),
                    ),
                )
                - T.float16(7)
            ) * lv1[lv_1[v_ax0], v_ax1 // 32]


@T.prim_func
def reshape1(var_A: T.handle, var_T_reshape: T.handle):
    T.func_attr({"global_symbol": "reshape1", "tir.noalias": T.bool(True)})
    n = T.int64()
    A = T.match_buffer(var_A, (n, T.int64(4096)), "float16")
    T_reshape = T.match_buffer(var_T_reshape, (T.int64(1), n, T.int64(4096)), "float16")
    # with T.block("root"):
    for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(4096)):
        with T.block("T_reshape"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(
                A[
                    (v_ax2 // T.int64(4096) + v_ax0 * n + v_ax1) % n,
                    v_ax2 % T.int64(4096),
                ]
            )
            T.writes(T_reshape[v_ax0, v_ax1, v_ax2])
            T_reshape[v_ax0, v_ax1, v_ax2] = A[
                (v_ax2 // T.int64(4096) + v_ax0 * n + v_ax1) % n, v_ax2 % T.int64(4096)
            ]


@T.prim_func
def fused_fused_decode2_NT_matmul(
    lv3: T.Buffer((T.int64(12288), T.int64(512)), "uint32"),
    lv4: T.Buffer((T.int64(12288), T.int64(128)), "float16"),
    p_lv: T.handle,
    p_output0: T.handle,
):
    T.func_attr(
        {"global_symbol": "fused_fused_decode2_NT_matmul", "tir.noalias": T.bool(True)}
    )
    n = T.int64()
    lv = T.match_buffer(p_lv, (T.int64(1), n, T.int64(4096)), "float16")
    var_NT_matmul_intermediate = T.match_buffer(
        p_output0, (T.int64(1), n, T.int64(12288)), "float16"
    )
    # with T.block("root"):
    p_output0_intermediate = T.alloc_buffer((T.int64(12288), T.int64(4096)), "float16")
    for i, j in T.grid(T.int64(12288), T.int64(4096)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv3[v_i, v_j // T.int64(8)], lv4[v_i, v_j // T.int64(32)])
            T.writes(p_output0_intermediate[v_i, v_j])
            p_output0_intermediate[v_i, v_j] = (
                T.Cast(
                    "float16",
                    T.bitwise_and(
                        T.shift_right(
                            lv3[v_i, v_j // T.int64(8)],
                            T.Cast("uint32", v_j % T.int64(8)) * T.uint32(4),
                        ),
                        T.uint32(15),
                    ),
                )
                - T.float16(7)
            ) * lv4[v_i, v_j // T.int64(32)]
    for i0, i1, i2, k in T.grid(T.int64(1), n, T.int64(12288), T.int64(4096)):
        with T.block("NT_matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv[v_i0, v_i1, v_k], p_output0_intermediate[v_i2, v_k])
            T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = (
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2]
                + lv[v_i0, v_i1, v_k] * p_output0_intermediate[v_i2, v_k]
            )


@T.prim_func
def split(
    var_A: T.handle,
    var_T_split: T.handle,
    var_T_split_1: T.handle,
    var_T_split_2: T.handle,
):
    T.func_attr({"global_symbol": "split", "tir.noalias": T.bool(True)})
    n = T.int64()
    A = T.match_buffer(var_A, (T.int64(1), n, T.int64(12288)), "float16")
    T_split = T.match_buffer(var_T_split, (T.int64(1), n, T.int64(4096)), "float16")
    T_split_1 = T.match_buffer(var_T_split_1, (T.int64(1), n, T.int64(4096)), "float16")
    T_split_2 = T.match_buffer(var_T_split_2, (T.int64(1), n, T.int64(4096)), "float16")
    # with T.block("root"):
    for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(4096)):
        with T.block("T_split"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(A[v_ax0, v_ax1, v_ax2])
            T.writes(T_split[v_ax0, v_ax1, v_ax2])
            T_split[v_ax0, v_ax1, v_ax2] = A[v_ax0, v_ax1, v_ax2]
    for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(4096)):
        with T.block("T_split_1"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(A[v_ax0, v_ax1, v_ax2 + T.int64(4096)])
            T.writes(T_split_1[v_ax0, v_ax1, v_ax2])
            T_split_1[v_ax0, v_ax1, v_ax2] = A[v_ax0, v_ax1, v_ax2 + T.int64(4096)]
    for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(4096)):
        with T.block("T_split_2"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(A[v_ax0, v_ax1, v_ax2 + T.int64(8192)])
            T.writes(T_split_2[v_ax0, v_ax1, v_ax2])
            T_split_2[v_ax0, v_ax1, v_ax2] = A[v_ax0, v_ax1, v_ax2 + T.int64(8192)]


@T.prim_func
def reshape2(var_A: T.handle, var_T_reshape: T.handle):
    T.func_attr({"global_symbol": "reshape2", "tir.noalias": T.bool(True)})
    n = T.int64()
    A = T.match_buffer(var_A, (T.int64(1), n, T.int64(4096)), "float16")
    T_reshape = T.match_buffer(
        var_T_reshape, (T.int64(1), n, T.int64(32), T.int64(128)), "float16"
    )
    # with T.block("root"):
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), n, T.int64(32), T.int64(128)):
        with T.block("T_reshape"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(
                A[
                    T.int64(0),
                    (
                        (v_ax2 * T.int64(128) + v_ax3) // T.int64(4096)
                        + v_ax0 * n
                        + v_ax1
                    )
                    % n,
                    (v_ax2 * T.int64(128) + v_ax3) % T.int64(4096),
                ]
            )
            T.writes(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3])
            T_reshape[v_ax0, v_ax1, v_ax2, v_ax3] = A[
                T.int64(0),
                ((v_ax2 * T.int64(128) + v_ax3) // T.int64(4096) + v_ax0 * n + v_ax1)
                % n,
                (v_ax2 * T.int64(128) + v_ax3) % T.int64(4096),
            ]


@T.prim_func
def rotary_embedding(
    var_A: T.handle,
    B: T.Buffer((T.int64(2048), T.int64(128)), "float16"),
    C: T.Buffer((T.int64(2048), T.int64(128)), "float16"),
    var_rotary: T.handle,
    m: T.int64,
):
    T.func_attr({"global_symbol": "rotary_embedding", "tir.noalias": T.bool(True)})
    n = T.int64()
    A = T.match_buffer(var_A, (T.int64(1), n, T.int64(32), T.int64(128)), "float16")
    rotary = T.match_buffer(
        var_rotary, (T.int64(1), n, T.int64(32), T.int64(128)), "float16"
    )
    # with T.block("root"):
    for i0, i1, i2, i3 in T.grid(T.int64(1), n, T.int64(32), T.int64(128)):
        with T.block("rotary"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(
                B[m + v_i1 - n, v_i3],
                A[
                    v_i0,
                    v_i1,
                    v_i2,
                    v_i3 - T.int64(64) : v_i3 - T.int64(64) + T.int64(129),
                ],
                C[m + v_i1 - n, v_i3],
            )
            T.writes(rotary[v_i0, v_i1, v_i2, v_i3])
            rotary[v_i0, v_i1, v_i2, v_i3] = B[m + v_i1 - n, v_i3] * A[
                v_i0, v_i1, v_i2, v_i3
            ] + C[m + v_i1 - n, v_i3] * T.Select(
                T.int64(64) <= v_i3,
                A[v_i0, v_i1, v_i2, v_i3 - T.int64(64)],
                A[v_i0, v_i1, v_i2, v_i3 + T.int64(64)] * T.float16(-1),
            )


@T.prim_func
def squeeze(var_A: T.handle, var_T_squeeze: T.handle):
    T.func_attr({"global_symbol": "squeeze", "tir.noalias": T.bool(True)})
    n = T.int64()
    A = T.match_buffer(var_A, (T.int64(1), n, T.int64(32), T.int64(128)), "float16")
    T_squeeze = T.match_buffer(var_T_squeeze, (n, T.int64(32), T.int64(128)), "float16")
    # with T.block("root"):
    for ax0, ax1, ax2 in T.grid(n, T.int64(32), T.int64(128)):
        with T.block("T_squeeze"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(A[T.int64(0), v_ax0, v_ax1, v_ax2])
            T.writes(T_squeeze[v_ax0, v_ax1, v_ax2])
            T_squeeze[v_ax0, v_ax1, v_ax2] = A[T.int64(0), v_ax0, v_ax1, v_ax2]


@T.prim_func
def reshape3(var_A: T.handle, var_T_reshape: T.handle):
    T.func_attr({"global_symbol": "reshape3", "tir.noalias": T.bool(True)})
    m = T.int64()
    A = T.match_buffer(var_A, (m, T.int64(32), T.int64(128)), "float16")
    T_reshape = T.match_buffer(
        var_T_reshape, (T.int64(1), m, T.int64(32), T.int64(128)), "float16"
    )
    # with T.block("root"):
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), m, T.int64(32), T.int64(128)):
        with T.block("T_reshape"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(
                A[
                    ((v_ax3 // T.int64(128) + v_ax2) // T.int64(32) + v_ax0 * m + v_ax1)
                    % m,
                    (v_ax3 // T.int64(128) + v_ax2) % T.int64(32),
                    v_ax3 % T.int64(128),
                ]
            )
            T.writes(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3])
            T_reshape[v_ax0, v_ax1, v_ax2, v_ax3] = A[
                ((v_ax3 // T.int64(128) + v_ax2) // T.int64(32) + v_ax0 * m + v_ax1)
                % m,
                (v_ax3 // T.int64(128) + v_ax2) % T.int64(32),
                v_ax3 % T.int64(128),
            ]


@T.prim_func
def reshape4(var_A: T.handle, var_T_reshape: T.handle):
    T.func_attr({"global_symbol": "reshape4", "tir.noalias": T.bool(True)})
    n = T.int64()
    A = T.match_buffer(var_A, (T.int64(1), n, T.int64(32), T.int64(128)), "float16")
    T_reshape = T.match_buffer(var_T_reshape, (T.int64(1), n, T.int64(4096)), "float16")
    # with T.block("root"):
    for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(4096)):
        with T.block("T_reshape"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(
                A[
                    T.int64(0),
                    (v_ax2 // T.int64(4096) + v_ax0 * n + v_ax1) % n,
                    v_ax2 % T.int64(4096) // T.int64(128),
                    v_ax2 % T.int64(128),
                ]
            )
            T.writes(T_reshape[v_ax0, v_ax1, v_ax2])
            T_reshape[v_ax0, v_ax1, v_ax2] = A[
                T.int64(0),
                (v_ax2 // T.int64(4096) + v_ax0 * n + v_ax1) % n,
                v_ax2 % T.int64(4096) // T.int64(128),
                v_ax2 % T.int64(128),
            ]


@T.prim_func
def fused_fused_decode3_fused_NT_matmul1_add(
    lv6: T.Buffer((T.int64(4096), T.int64(512)), "uint32"),
    lv7: T.Buffer((T.int64(4096), T.int64(128)), "float16"),
    p_lv41: T.handle,
    p_lv2: T.handle,
    p_output0: T.handle,
):
    T.func_attr(
        {
            "global_symbol": "fused_fused_decode3_fused_NT_matmul1_add",
            "tir.noalias": T.bool(True),
        }
    )
    n = T.int64()
    lv41 = T.match_buffer(p_lv41, (T.int64(1), n, T.int64(4096)), "float16")
    lv2 = T.match_buffer(p_lv2, (T.int64(1), n, T.int64(4096)), "float16")
    p_output0_intermediate = T.match_buffer(
        p_output0, (T.int64(1), n, T.int64(4096)), "float16"
    )
    # with T.block("root"):
    p_output0_intermediate_1 = T.alloc_buffer((T.int64(4096), T.int64(4096)), "float16")
    var_NT_matmul_intermediate = T.alloc_buffer(
        (T.int64(1), n, T.int64(4096)), "float16"
    )
    for i, j in T.grid(T.int64(4096), T.int64(4096)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv6[v_i, v_j // T.int64(8)], lv7[v_i, v_j // T.int64(32)])
            T.writes(p_output0_intermediate_1[v_i, v_j])
            p_output0_intermediate_1[v_i, v_j] = (
                T.Cast(
                    "float16",
                    T.bitwise_and(
                        T.shift_right(
                            lv6[v_i, v_j // T.int64(8)],
                            T.Cast("uint32", v_j % T.int64(8)) * T.uint32(4),
                        ),
                        T.uint32(15),
                    ),
                )
                - T.float16(7)
            ) * lv7[v_i, v_j // T.int64(32)]
    for i0, i1, i2, k in T.grid(T.int64(1), n, T.int64(4096), T.int64(4096)):
        with T.block("NT_matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv41[v_i0, v_i1, v_k], p_output0_intermediate_1[v_i2, v_k])
            T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = (
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2]
                + lv41[v_i0, v_i1, v_k] * p_output0_intermediate_1[v_i2, v_k]
            )
    for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(4096)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(
                lv2[v_ax0, v_ax1, v_ax2],
                var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2],
            )
            T.writes(p_output0_intermediate[v_ax0, v_ax1, v_ax2])
            p_output0_intermediate[v_ax0, v_ax1, v_ax2] = (
                lv2[v_ax0, v_ax1, v_ax2]
                + var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2]
            )


@T.prim_func
def fused_fused_decode4_NT_matmul2(
    lv10: T.Buffer((T.int64(22016), T.int64(512)), "uint32"),
    lv11: T.Buffer((T.int64(22016), T.int64(128)), "float16"),
    p_lv1: T.handle,
    p_output0: T.handle,
):
    T.func_attr(
        {"global_symbol": "fused_fused_decode4_NT_matmul2", "tir.noalias": T.bool(True)}
    )
    n = T.int64()
    lv1 = T.match_buffer(p_lv1, (T.int64(1), n, T.int64(4096)), "float16")
    var_NT_matmul_intermediate = T.match_buffer(
        p_output0, (T.int64(1), n, T.int64(22016)), "float16"
    )
    # with T.block("root"):
    p_output0_intermediate = T.alloc_buffer((T.int64(22016), T.int64(4096)), "float16")
    for i, j in T.grid(T.int64(22016), T.int64(4096)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv10[v_i, v_j // T.int64(8)], lv11[v_i, v_j // T.int64(32)])
            T.writes(p_output0_intermediate[v_i, v_j])
            p_output0_intermediate[v_i, v_j] = (
                T.Cast(
                    "float16",
                    T.bitwise_and(
                        T.shift_right(
                            lv10[v_i, v_j // T.int64(8)],
                            T.Cast("uint32", v_j % T.int64(8)) * T.uint32(4),
                        ),
                        T.uint32(15),
                    ),
                )
                - T.float16(7)
            ) * lv11[v_i, v_j // T.int64(32)]
    for i0, i1, i2, k in T.grid(T.int64(1), n, T.int64(22016), T.int64(4096)):
        with T.block("NT_matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv1[v_i0, v_i1, v_k], p_output0_intermediate[v_i2, v_k])
            T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = (
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2]
                + lv1[v_i0, v_i1, v_k] * p_output0_intermediate[v_i2, v_k]
            )


@T.prim_func
def fused_split1_silu_multiply(p_lv2: T.handle, p_output0: T.handle):
    T.func_attr(
        {"global_symbol": "fused_split1_silu_multiply", "tir.noalias": T.bool(True)}
    )
    n = T.int64()
    lv2 = T.match_buffer(p_lv2, (T.int64(1), n, T.int64(22016)), "float16")
    var_T_multiply_intermediate = T.match_buffer(
        p_output0, (T.int64(1), n, T.int64(11008)), "float16"
    )
    # with T.block("root"):
    var_T_split_sections_intermediate = T.alloc_buffer(
        (T.int64(1), n, T.int64(11008)), "float16"
    )
    var_T_split_sections_intermediate_1 = T.alloc_buffer(
        (T.int64(1), n, T.int64(11008)), "float16"
    )
    compute = T.alloc_buffer((T.int64(1), n, T.int64(11008)), "float16")
    var_T_multiply_intermediate_1 = T.alloc_buffer(
        (T.int64(1), n, T.int64(11008)), "float16"
    )
    for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(11008)):
        with T.block("T_split_sections"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(lv2[v_ax0, v_ax1, v_ax2])
            T.writes(var_T_split_sections_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_split_sections_intermediate[v_ax0, v_ax1, v_ax2] = lv2[
                v_ax0, v_ax1, v_ax2
            ]
    for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(11008)):
        with T.block("T_split_sections_1"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(lv2[v_ax0, v_ax1, v_ax2 + T.int64(11008)])
            T.writes(var_T_split_sections_intermediate_1[v_ax0, v_ax1, v_ax2])
            var_T_split_sections_intermediate_1[v_ax0, v_ax1, v_ax2] = lv2[
                v_ax0, v_ax1, v_ax2 + T.int64(11008)
            ]
    for i0, i1, i2 in T.grid(T.int64(1), n, T.int64(11008)):
        with T.block("compute"):
            v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
            T.reads(var_T_split_sections_intermediate[v_i0, v_i1, v_i2])
            T.writes(compute[v_i0, v_i1, v_i2])
            compute[v_i0, v_i1, v_i2] = T.sigmoid(
                var_T_split_sections_intermediate[v_i0, v_i1, v_i2]
            )
    for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(11008)):
        with T.block("T_multiply"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(
                var_T_split_sections_intermediate[v_ax0, v_ax1, v_ax2],
                compute[v_ax0, v_ax1, v_ax2],
            )
            T.writes(var_T_multiply_intermediate_1[v_ax0, v_ax1, v_ax2])
            var_T_multiply_intermediate_1[v_ax0, v_ax1, v_ax2] = (
                var_T_split_sections_intermediate[v_ax0, v_ax1, v_ax2]
                * compute[v_ax0, v_ax1, v_ax2]
            )
    for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(11008)):
        with T.block("T_multiply_1"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(
                var_T_multiply_intermediate_1[v_ax0, v_ax1, v_ax2],
                var_T_split_sections_intermediate_1[v_ax0, v_ax1, v_ax2],
            )
            T.writes(var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2] = (
                var_T_multiply_intermediate_1[v_ax0, v_ax1, v_ax2]
                * var_T_split_sections_intermediate_1[v_ax0, v_ax1, v_ax2]
            )


@T.prim_func
def fused_fused_decode5_fused_NT_matmul3_add(
    lv14: T.Buffer((T.int64(4096), T.int64(1376)), "uint32"),
    lv15: T.Buffer((T.int64(4096), T.int64(344)), "float16"),
    p_lv13: T.handle,
    p_lv9: T.handle,
    p_output0: T.handle,
):
    T.func_attr(
        {
            "global_symbol": "fused_fused_decode5_fused_NT_matmul3_add",
            "tir.noalias": T.bool(True),
        }
    )
    n = T.int64()
    lv13 = T.match_buffer(p_lv13, (T.int64(1), n, T.int64(11008)), "float16")
    lv9 = T.match_buffer(p_lv9, (T.int64(1), n, T.int64(4096)), "float16")
    p_output0_intermediate = T.match_buffer(
        p_output0, (T.int64(1), n, T.int64(4096)), "float16"
    )
    # with T.block("root"):
    p_output0_intermediate_1 = T.alloc_buffer(
        (T.int64(4096), T.int64(11008)), "float16"
    )
    var_NT_matmul_intermediate = T.alloc_buffer(
        (T.int64(1), n, T.int64(4096)), "float16"
    )
    for i, j in T.grid(T.int64(4096), T.int64(11008)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv14[v_i, v_j // T.int64(8)], lv15[v_i, v_j // T.int64(32)])
            T.writes(p_output0_intermediate_1[v_i, v_j])
            p_output0_intermediate_1[v_i, v_j] = (
                T.Cast(
                    "float16",
                    T.bitwise_and(
                        T.shift_right(
                            lv14[v_i, v_j // T.int64(8)],
                            T.Cast("uint32", v_j % T.int64(8)) * T.uint32(4),
                        ),
                        T.uint32(15),
                    ),
                )
                - T.float16(7)
            ) * lv15[v_i, v_j // T.int64(32)]
    for i0, i1, i2, k in T.grid(T.int64(1), n, T.int64(4096), T.int64(11008)):
        with T.block("NT_matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv13[v_i0, v_i1, v_k], p_output0_intermediate_1[v_i2, v_k])
            T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = (
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2]
                + lv13[v_i0, v_i1, v_k] * p_output0_intermediate_1[v_i2, v_k]
            )
    for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(4096)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(
                lv9[v_ax0, v_ax1, v_ax2],
                var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2],
            )
            T.writes(p_output0_intermediate[v_ax0, v_ax1, v_ax2])
            p_output0_intermediate[v_ax0, v_ax1, v_ax2] = (
                lv9[v_ax0, v_ax1, v_ax2]
                + var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2]
            )


@T.prim_func
def slice(
    var_A: T.handle,
    slice_1: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"),
):
    T.func_attr({"global_symbol": "slice", "tir.noalias": T.bool(True)})
    n = T.int64()
    A = T.match_buffer(var_A, (T.int64(1), n, T.int64(4096)), "float16")
    # with T.block("root"):
    for i, j, k in T.grid(T.int64(1), T.int64(1), T.int64(4096)):
        with T.block("slice"):
            v_i, v_j, v_k = T.axis.remap("SSS", [i, j, k])
            T.reads(A[v_i, n - T.int64(1), v_k])
            T.writes(slice_1[v_i, v_j, v_k])
            slice_1[v_i, v_j, v_k] = A[v_i, n - T.int64(1), v_k]


@T.prim_func
def fused_fused_decode1_fused_NT_matmul4_cast(
    lv483: T.Buffer((T.int64(32000), T.int64(512)), "uint32"),
    lv484: T.Buffer((T.int64(32000), T.int64(128)), "float16"),
    lv1607: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"),
    p_output0_intermediate: T.Buffer(
        (T.int64(1), T.int64(1), T.int64(32000)), "float32"
    ),
):
    T.func_attr(
        {
            "global_symbol": "fused_fused_decode1_fused_NT_matmul4_cast",
            "tir.noalias": T.bool(True),
        }
    )
    # with T.block("root"):
    p_output0_intermediate_1 = T.alloc_buffer(
        (T.int64(32000), T.int64(4096)), "float16"
    )
    var_NT_matmul_intermediate = T.alloc_buffer(
        (T.int64(1), T.int64(1), T.int64(32000)), "float16"
    )
    for i, j in T.grid(T.int64(32000), T.int64(4096)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv483[v_i, v_j // T.int64(8)], lv484[v_i, v_j // T.int64(32)])
            T.writes(p_output0_intermediate_1[v_i, v_j])
            p_output0_intermediate_1[v_i, v_j] = (
                T.Cast(
                    "float16",
                    T.bitwise_and(
                        T.shift_right(
                            lv483[v_i, v_j // T.int64(8)],
                            T.Cast("uint32", v_j % T.int64(8)) * T.uint32(4),
                        ),
                        T.uint32(15),
                    ),
                )
                - T.float16(7)
            ) * lv484[v_i, v_j // T.int64(32)]
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(32000), T.int64(4096)):
        with T.block("NT_matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv1607[v_i0, v_i1, v_k], p_output0_intermediate_1[v_i2, v_k])
            T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = (
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2]
                + lv1607[v_i0, v_i1, v_k] * p_output0_intermediate_1[v_i2, v_k]
            )
    for i0, i1, i2 in T.grid(T.int64(1), T.int64(1), T.int64(32000)):
        with T.block("compute"):
            v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
            T.reads(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
            T.writes(p_output0_intermediate[v_i0, v_i1, v_i2])
            p_output0_intermediate[v_i0, v_i1, v_i2] = T.Cast(
                "float32", var_NT_matmul_intermediate[v_i0, v_i1, v_i2]
            )


@T.prim_func
def reshape5(
    A: T.Buffer((T.int64(1), T.int64(1)), "int32"),
    T_reshape: T.Buffer((T.int64(1),), "int32"),
):
    T.func_attr({"global_symbol": "reshape5", "tir.noalias": T.bool(True)})
    # with T.block("root"):
    for ax0 in range(T.int64(1)):
        with T.block("T_reshape"):
            v_ax0 = T.axis.spatial(T.int64(1), ax0)
            T.reads(A[T.int64(0), T.int64(0)])
            T.writes(T_reshape[v_ax0])
            T_reshape[v_ax0] = A[T.int64(0), T.int64(0)]


@T.prim_func
def fused_fused_decode1_take1(
    lv487: T.Buffer((32000, 512), "uint32"),
    lv488: T.Buffer((32000, 128), "float16"),
    lv1611: T.Buffer((1,), "int32"),
    var_T_take_intermediate: T.Buffer((1, 4096), "float16"),
):
    T.func_attr(
        {"global_symbol": "fused_fused_decode1_take1", "tir.noalias": T.bool(True)}
    )
    # with T.block("root"):
    for ax0, ax1 in T.grid(1, 4096):
        with T.block("T_take"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(
                lv487[lv1611[v_ax0], v_ax1 // 8],
                lv1611[v_ax0],
                lv488[lv1611[v_ax0], v_ax1 // 32],
            )
            T.writes(var_T_take_intermediate[v_ax0, v_ax1])
            var_T_take_intermediate[v_ax0, v_ax1] = (
                T.Cast(
                    "float16",
                    T.bitwise_and(
                        T.shift_right(
                            lv487[lv1611[v_ax0], v_ax1 // 8],
                            T.Cast("uint32", v_ax1 % 8) * T.uint32(4),
                        ),
                        T.uint32(15),
                    ),
                )
                - T.float16(7)
            ) * lv488[lv1611[v_ax0], v_ax1 // 32]


@T.prim_func
def reshape6(
    A: T.Buffer((T.int64(1), T.int64(4096)), "float16"),
    T_reshape: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"),
):
    T.func_attr({"global_symbol": "reshape6", "tir.noalias": T.bool(True)})
    # with T.block("root"):
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(4096)):
        with T.block("T_reshape"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(A[T.int64(0), v_ax2 % T.int64(4096)])
            T.writes(T_reshape[v_ax0, v_ax1, v_ax2])
            T_reshape[v_ax0, v_ax1, v_ax2] = A[T.int64(0), v_ax2 % T.int64(4096)]


@T.prim_func
def fused_fused_decode2_NT_matmul5(
    lv490: T.Buffer((T.int64(12288), T.int64(512)), "uint32"),
    lv491: T.Buffer((T.int64(12288), T.int64(128)), "float16"),
    lv65: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"),
    var_NT_matmul_intermediate: T.Buffer(
        (T.int64(1), T.int64(1), T.int64(12288)), "float16"
    ),
):
    T.func_attr(
        {"global_symbol": "fused_fused_decode2_NT_matmul5", "tir.noalias": T.bool(True)}
    )
    # with T.block("root"):
    p_output0_intermediate = T.alloc_buffer((T.int64(12288), T.int64(4096)), "float16")
    for i, j in T.grid(T.int64(12288), T.int64(4096)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv490[v_i, v_j // T.int64(8)], lv491[v_i, v_j // T.int64(32)])
            T.writes(p_output0_intermediate[v_i, v_j])
            p_output0_intermediate[v_i, v_j] = (
                T.Cast(
                    "float16",
                    T.bitwise_and(
                        T.shift_right(
                            lv490[v_i, v_j // T.int64(8)],
                            T.Cast("uint32", v_j % T.int64(8)) * T.uint32(4),
                        ),
                        T.uint32(15),
                    ),
                )
                - T.float16(7)
            ) * lv491[v_i, v_j // T.int64(32)]
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(12288), T.int64(4096)):
        with T.block("NT_matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv65[v_i0, v_i1, v_k], p_output0_intermediate[v_i2, v_k])
            T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = (
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2]
                + lv65[v_i0, v_i1, v_k] * p_output0_intermediate[v_i2, v_k]
            )


@T.prim_func
def split_rotary(
    A: T.Buffer((1, 1, 12288), "float16"),
    cos: T.Buffer((2048, 128), "float16"),
    sin: T.Buffer((2048, 128), "float16"),
    T_split: T.Buffer((1, 1, 4096), "float16"),
    T_split_1: T.Buffer((1, 1, 4096), "float16"),
    T_split_2: T.Buffer((1, 1, 4096), "float16"),
    n: T.int64,
):
    T.func_attr({"global_symbol": "split_rotary", "tir.noalias": T.bool(True)})
    # with T.block("root"):
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(4096)):
        with T.block("T_split"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(
                A[v_ax0, v_ax1, v_ax2],
                A[v_ax0, v_ax1, v_ax2 + T.int64(4096)],
                A[v_ax0, v_ax1, v_ax2 + T.int64(8192)],
            )
            T.writes(
                T_split[v_ax0, v_ax1, v_ax2],
                T_split_1[v_ax0, v_ax1, v_ax2],
                T_split_2[v_ax0, v_ax1, v_ax2],
            )
            T_split[v_ax0, v_ax1, v_ax2] = cos[
                n - T.int64(1), v_ax2 % T.int64(128)
            ] * A[v_ax0, v_ax1, v_ax2] + sin[
                n - T.int64(1), v_ax2 % T.int64(128)
            ] * T.Select(
                T.int64(64) <= v_ax2 % T.int64(128),
                A[v_ax0, v_ax1, v_ax2 - T.int64(64)],
                A[v_ax0, v_ax1, v_ax2 + T.int64(64)] * T.float16(-1),
            )
            T_split_1[v_ax0, v_ax1, v_ax2] = cos[
                n - T.int64(1), v_ax2 % T.int64(128)
            ] * A[v_ax0, v_ax1, v_ax2 + T.int64(4096)] + sin[
                n - T.int64(1), v_ax2 % T.int64(128)
            ] * T.Select(
                T.int64(64) <= v_ax2 % T.int64(128),
                A[v_ax0, v_ax1, v_ax2 + T.int64(4096) - T.int64(64)],
                A[v_ax0, v_ax1, v_ax2 + T.int64(4096) + T.int64(64)] * T.float16(-1),
            )
            T_split_2[v_ax0, v_ax1, v_ax2] = A[v_ax0, v_ax1, v_ax2 + T.int64(8192)]


@T.prim_func
def fused_reshape7(
    lv_0: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"),
    var_T_reshape_intermediate: T.Buffer(
        (T.int64(1), T.int64(1), T.int64(32), T.int64(128)), "float16"
    ),
):
    T.func_attr({"global_symbol": "fused_reshape7", "tir.noalias": T.bool(True)})
    # with T.block("root"):
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(1), T.int64(32), T.int64(128)):
        with T.block("T_reshape"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(
                lv_0[
                    T.int64(0),
                    T.int64(0),
                    (v_ax2 * T.int64(128) + v_ax3) % T.int64(4096),
                ]
            )
            T.writes(var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = lv_0[
                T.int64(0), T.int64(0), (v_ax2 * T.int64(128) + v_ax3) % T.int64(4096)
            ]


@T.prim_func
def fused_reshape7_squeeze1(
    lv_1: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"),
    var_T_squeeze_intermediate: T.Buffer(
        (T.int64(1), T.int64(32), T.int64(128)), "float16"
    ),
):
    T.func_attr(
        {"global_symbol": "fused_reshape7_squeeze1", "tir.noalias": T.bool(True)}
    )
    # with T.block("root"):
    var_T_reshape_intermediate = T.alloc_buffer(
        (T.int64(1), T.int64(1), T.int64(32), T.int64(128)), "float16"
    )
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(1), T.int64(32), T.int64(128)):
        with T.block("T_reshape"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(
                lv_1[
                    T.int64(0),
                    T.int64(0),
                    (v_ax2 * T.int64(128) + v_ax3) % T.int64(4096),
                ]
            )
            T.writes(var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = lv_1[
                T.int64(0), T.int64(0), (v_ax2 * T.int64(128) + v_ax3) % T.int64(4096)
            ]
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(32), T.int64(128)):
        with T.block("T_squeeze"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_T_reshape_intermediate[T.int64(0), v_ax0, v_ax1, v_ax2])
            T.writes(var_T_squeeze_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_squeeze_intermediate[
                v_ax0, v_ax1, v_ax2
            ] = var_T_reshape_intermediate[T.int64(0), v_ax0, v_ax1, v_ax2]


@T.prim_func
def reshape3(var_A: T.handle, var_T_reshape: T.handle):
    T.func_attr({"global_symbol": "reshape3", "tir.noalias": T.bool(True)})
    m = T.int64()
    A = T.match_buffer(var_A, (m, T.int64(32), T.int64(128)), "float16")
    T_reshape = T.match_buffer(
        var_T_reshape, (T.int64(1), m, T.int64(32), T.int64(128)), "float16"
    )
    # with T.block("root"):
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), m, T.int64(32), T.int64(128)):
        with T.block("T_reshape"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(
                A[
                    ((v_ax3 // T.int64(128) + v_ax2) // T.int64(32) + v_ax0 * m + v_ax1)
                    % m,
                    (v_ax3 // T.int64(128) + v_ax2) % T.int64(32),
                    v_ax3 % T.int64(128),
                ]
            )
            T.writes(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3])
            T_reshape[v_ax0, v_ax1, v_ax2, v_ax3] = A[
                ((v_ax3 // T.int64(128) + v_ax2) // T.int64(32) + v_ax0 * m + v_ax1)
                % m,
                (v_ax3 // T.int64(128) + v_ax2) % T.int64(32),
                v_ax3 % T.int64(128),
            ]


@T.prim_func
def reshape8(
    A: T.Buffer((T.int64(1), T.int64(1), T.int64(32), T.int64(128)), "float16"),
    T_reshape: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"),
):
    T.func_attr({"global_symbol": "reshape8", "tir.noalias": T.bool(True)})
    # with T.block("root"):
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(4096)):
        with T.block("T_reshape"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(
                A[
                    T.int64(0),
                    T.int64(0),
                    v_ax2 % T.int64(4096) // T.int64(128),
                    v_ax2 % T.int64(128),
                ]
            )
            T.writes(T_reshape[v_ax0, v_ax1, v_ax2])
            T_reshape[v_ax0, v_ax1, v_ax2] = A[
                T.int64(0),
                T.int64(0),
                v_ax2 % T.int64(4096) // T.int64(128),
                v_ax2 % T.int64(128),
            ]


@T.prim_func
def fused_fused_decode3_fused_NT_matmul6_add1(
    lv499: T.Buffer((T.int64(4096), T.int64(512)), "uint32"),
    lv500: T.Buffer((T.int64(4096), T.int64(128)), "float16"),
    lv1650: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"),
    lv1613: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"),
    p_output0_intermediate: T.Buffer(
        (T.int64(1), T.int64(1), T.int64(4096)), "float16"
    ),
):
    T.func_attr(
        {
            "global_symbol": "fused_fused_decode3_fused_NT_matmul6_add1",
            "tir.noalias": T.bool(True),
        }
    )
    # with T.block("root"):
    p_output0_intermediate_1 = T.alloc_buffer((T.int64(4096), T.int64(4096)), "float16")
    var_NT_matmul_intermediate = T.alloc_buffer(
        (T.int64(1), T.int64(1), T.int64(4096)), "float16"
    )
    for i, j in T.grid(T.int64(4096), T.int64(4096)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv499[v_i, v_j // T.int64(8)], lv500[v_i, v_j // T.int64(32)])
            T.writes(p_output0_intermediate_1[v_i, v_j])
            p_output0_intermediate_1[v_i, v_j] = (
                T.Cast(
                    "float16",
                    T.bitwise_and(
                        T.shift_right(
                            lv499[v_i, v_j // T.int64(8)],
                            T.Cast("uint32", v_j % T.int64(8)) * T.uint32(4),
                        ),
                        T.uint32(15),
                    ),
                )
                - T.float16(7)
            ) * lv500[v_i, v_j // T.int64(32)]
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(4096), T.int64(4096)):
        with T.block("NT_matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv1650[v_i0, v_i1, v_k], p_output0_intermediate_1[v_i2, v_k])
            T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = (
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2]
                + lv1650[v_i0, v_i1, v_k] * p_output0_intermediate_1[v_i2, v_k]
            )
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(4096)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(
                lv1613[v_ax0, v_ax1, v_ax2],
                var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2],
            )
            T.writes(p_output0_intermediate[v_ax0, v_ax1, v_ax2])
            p_output0_intermediate[v_ax0, v_ax1, v_ax2] = (
                lv1613[v_ax0, v_ax1, v_ax2]
                + var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2]
            )


@T.prim_func
def fused_fused_decode4_NT_matmul7(
    lv503: T.Buffer((T.int64(22016), T.int64(512)), "uint32"),
    lv504: T.Buffer((T.int64(22016), T.int64(128)), "float16"),
    lv66: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"),
    var_NT_matmul_intermediate: T.Buffer(
        (T.int64(1), T.int64(1), T.int64(22016)), "float16"
    ),
):
    T.func_attr(
        {"global_symbol": "fused_fused_decode4_NT_matmul7", "tir.noalias": T.bool(True)}
    )
    # with T.block("root"):
    p_output0_intermediate = T.alloc_buffer((T.int64(22016), T.int64(4096)), "float16")
    for i, j in T.grid(T.int64(22016), T.int64(4096)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv503[v_i, v_j // T.int64(8)], lv504[v_i, v_j // T.int64(32)])
            T.writes(p_output0_intermediate[v_i, v_j])
            p_output0_intermediate[v_i, v_j] = (
                T.Cast(
                    "float16",
                    T.bitwise_and(
                        T.shift_right(
                            lv503[v_i, v_j // T.int64(8)],
                            T.Cast("uint32", v_j % T.int64(8)) * T.uint32(4),
                        ),
                        T.uint32(15),
                    ),
                )
                - T.float16(7)
            ) * lv504[v_i, v_j // T.int64(32)]
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(22016), T.int64(4096)):
        with T.block("NT_matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv66[v_i0, v_i1, v_k], p_output0_intermediate[v_i2, v_k])
            T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = (
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2]
                + lv66[v_i0, v_i1, v_k] * p_output0_intermediate[v_i2, v_k]
            )


@T.prim_func
def fused_split2_silu1_multiply1(
    lv131: T.Buffer((T.int64(1), T.int64(1), T.int64(22016)), "float16"),
    var_T_multiply_intermediate: T.Buffer(
        (T.int64(1), T.int64(1), T.int64(11008)), "float16"
    ),
):
    T.func_attr(
        {"global_symbol": "fused_split2_silu1_multiply1", "tir.noalias": T.bool(True)}
    )
    # with T.block("root"):
    var_T_split_sections_intermediate = T.alloc_buffer(
        (T.int64(1), T.int64(1), T.int64(11008)), "float16"
    )
    var_T_split_sections_intermediate_1 = T.alloc_buffer(
        (T.int64(1), T.int64(1), T.int64(11008)), "float16"
    )
    compute = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(11008)), "float16")
    var_T_multiply_intermediate_1 = T.alloc_buffer(
        (T.int64(1), T.int64(1), T.int64(11008)), "float16"
    )
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(11008)):
        with T.block("T_split_sections"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(lv131[v_ax0, v_ax1, v_ax2])
            T.writes(var_T_split_sections_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_split_sections_intermediate[v_ax0, v_ax1, v_ax2] = lv131[
                v_ax0, v_ax1, v_ax2
            ]
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(11008)):
        with T.block("T_split_sections_1"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(lv131[v_ax0, v_ax1, v_ax2 + T.int64(11008)])
            T.writes(var_T_split_sections_intermediate_1[v_ax0, v_ax1, v_ax2])
            var_T_split_sections_intermediate_1[v_ax0, v_ax1, v_ax2] = lv131[
                v_ax0, v_ax1, v_ax2 + T.int64(11008)
            ]
    for i0, i1, i2 in T.grid(T.int64(1), T.int64(1), T.int64(11008)):
        with T.block("compute"):
            v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
            T.reads(var_T_split_sections_intermediate[v_i0, v_i1, v_i2])
            T.writes(compute[v_i0, v_i1, v_i2])
            compute[v_i0, v_i1, v_i2] = T.sigmoid(
                var_T_split_sections_intermediate[v_i0, v_i1, v_i2]
            )
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(11008)):
        with T.block("T_multiply"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(
                var_T_split_sections_intermediate[v_ax0, v_ax1, v_ax2],
                compute[v_ax0, v_ax1, v_ax2],
            )
            T.writes(var_T_multiply_intermediate_1[v_ax0, v_ax1, v_ax2])
            var_T_multiply_intermediate_1[v_ax0, v_ax1, v_ax2] = (
                var_T_split_sections_intermediate[v_ax0, v_ax1, v_ax2]
                * compute[v_ax0, v_ax1, v_ax2]
            )
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(11008)):
        with T.block("T_multiply_1"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(
                var_T_multiply_intermediate_1[v_ax0, v_ax1, v_ax2],
                var_T_split_sections_intermediate_1[v_ax0, v_ax1, v_ax2],
            )
            T.writes(var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2] = (
                var_T_multiply_intermediate_1[v_ax0, v_ax1, v_ax2]
                * var_T_split_sections_intermediate_1[v_ax0, v_ax1, v_ax2]
            )


@T.prim_func
def fused_fused_decode5_fused_NT_matmul8_add1(
    lv507: T.Buffer((T.int64(4096), T.int64(1376)), "uint32"),
    lv508: T.Buffer((T.int64(4096), T.int64(344)), "float16"),
    lv506: T.Buffer((T.int64(1), T.int64(1), T.int64(11008)), "float16"),
    lv502: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"),
    p_output0_intermediate: T.Buffer(
        (T.int64(1), T.int64(1), T.int64(4096)), "float16"
    ),
):
    T.func_attr(
        {
            "global_symbol": "fused_fused_decode5_fused_NT_matmul8_add1",
            "tir.noalias": T.bool(True),
        }
    )
    # with T.block("root"):
    p_output0_intermediate_1 = T.alloc_buffer(
        (T.int64(4096), T.int64(11008)), "float16"
    )
    var_NT_matmul_intermediate = T.alloc_buffer(
        (T.int64(1), T.int64(1), T.int64(4096)), "float16"
    )
    for i, j in T.grid(T.int64(4096), T.int64(11008)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv507[v_i, v_j // T.int64(8)], lv508[v_i, v_j // T.int64(32)])
            T.writes(p_output0_intermediate_1[v_i, v_j])
            p_output0_intermediate_1[v_i, v_j] = (
                T.Cast(
                    "float16",
                    T.bitwise_and(
                        T.shift_right(
                            lv507[v_i, v_j // T.int64(8)],
                            T.Cast("uint32", v_j % T.int64(8)) * T.uint32(4),
                        ),
                        T.uint32(15),
                    ),
                )
                - T.float16(7)
            ) * lv508[v_i, v_j // T.int64(32)]
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(4096), T.int64(11008)):
        with T.block("NT_matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv506[v_i0, v_i1, v_k], p_output0_intermediate_1[v_i2, v_k])
            T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = (
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2]
                + lv506[v_i0, v_i1, v_k] * p_output0_intermediate_1[v_i2, v_k]
            )
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(4096)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(
                lv502[v_ax0, v_ax1, v_ax2],
                var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2],
            )
            T.writes(p_output0_intermediate[v_ax0, v_ax1, v_ax2])
            p_output0_intermediate[v_ax0, v_ax1, v_ax2] = (
                lv502[v_ax0, v_ax1, v_ax2]
                + var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2]
            )


@T.prim_func
def slice1(
    A: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"),
    slice: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"),
):
    T.func_attr({"global_symbol": "slice1", "tir.noalias": T.bool(True)})
    # with T.block("root"):
    for i, j, k in T.grid(T.int64(1), T.int64(1), T.int64(4096)):
        with T.block("slice"):
            v_i, v_j, v_k = T.axis.remap("SSS", [i, j, k])
            T.reads(A[v_i, T.int64(0), v_k])
            T.writes(slice[v_i, v_j, v_k])
            slice[v_i, v_j, v_k] = A[v_i, T.int64(0), v_k]


@T.prim_func
def fused_fused_decode1_fused_NT_matmul4_cast(
    lv483: T.Buffer((T.int64(32000), T.int64(512)), "uint32"),
    lv484: T.Buffer((T.int64(32000), T.int64(128)), "float16"),
    lv1607: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"),
    p_output0_intermediate: T.Buffer(
        (T.int64(1), T.int64(1), T.int64(32000)), "float32"
    ),
):
    T.func_attr(
        {
            "global_symbol": "fused_fused_decode1_fused_NT_matmul4_cast",
            "tir.noalias": T.bool(True),
        }
    )
    # with T.block("root"):
    p_output0_intermediate_1 = T.alloc_buffer(
        (T.int64(32000), T.int64(4096)), "float16"
    )
    var_NT_matmul_intermediate = T.alloc_buffer(
        (T.int64(1), T.int64(1), T.int64(32000)), "float16"
    )
    for i, j in T.grid(T.int64(32000), T.int64(4096)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv483[v_i, v_j // T.int64(8)], lv484[v_i, v_j // T.int64(32)])
            T.writes(p_output0_intermediate_1[v_i, v_j])
            p_output0_intermediate_1[v_i, v_j] = (
                T.Cast(
                    "float16",
                    T.bitwise_and(
                        T.shift_right(
                            lv483[v_i, v_j // T.int64(8)],
                            T.Cast("uint32", v_j % T.int64(8)) * T.uint32(4),
                        ),
                        T.uint32(15),
                    ),
                )
                - T.float16(7)
            ) * lv484[v_i, v_j // T.int64(32)]
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(32000), T.int64(4096)):
        with T.block("NT_matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv1607[v_i0, v_i1, v_k], p_output0_intermediate_1[v_i2, v_k])
            T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = (
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2]
                + lv1607[v_i0, v_i1, v_k] * p_output0_intermediate_1[v_i2, v_k]
            )
    for i0, i1, i2 in T.grid(T.int64(1), T.int64(1), T.int64(32000)):
        with T.block("compute"):
            v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
            T.reads(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
            T.writes(p_output0_intermediate[v_i0, v_i1, v_i2])
            p_output0_intermediate[v_i0, v_i1, v_i2] = T.Cast(
                "float32", var_NT_matmul_intermediate[v_i0, v_i1, v_i2]
            )


@T.prim_func
def divide(
    A: T.Buffer((T.int64(1), T.int64(1), T.int64(32000)), "float32"),
    B: T.Buffer((), "float32"),
    T_divide: T.Buffer((T.int64(1), T.int64(1), T.int64(32000)), "float32"),
):
    T.func_attr({"global_symbol": "divide", "tir.noalias": T.bool(True)})
    # with T.block("root"):
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(32000)):
        with T.block("T_divide"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(A[v_ax0, v_ax1, v_ax2], B[()])
            T.writes(T_divide[v_ax0, v_ax1, v_ax2])
            T_divide[v_ax0, v_ax1, v_ax2] = A[v_ax0, v_ax1, v_ax2] / B[()]


@T.prim_func
def softmax(
    A: T.Buffer((T.int64(1), T.int64(1), T.int64(32000)), "float32"),
    T_softmax_norm: T.Buffer((T.int64(1), T.int64(1), T.int64(32000)), "float32"),
):
    T.func_attr({"global_symbol": "softmax", "tir.noalias": T.bool(True)})
    # with T.block("root"):
    T_softmax_maxelem = T.alloc_buffer((T.int64(1), T.int64(1)))
    T_softmax_exp = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(32000)))
    T_softmax_expsum = T.alloc_buffer((T.int64(1), T.int64(1)))
    for i0, i1, k in T.grid(T.int64(1), T.int64(1), T.int64(32000)):
        with T.block("T_softmax_maxelem"):
            v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])
            T.reads(A[v_i0, v_i1, v_k])
            T.writes(T_softmax_maxelem[v_i0, v_i1])
            with T.init():
                T_softmax_maxelem[v_i0, v_i1] = T.float32(-3.4028234663852886e38)
            T_softmax_maxelem[v_i0, v_i1] = T.max(
                T_softmax_maxelem[v_i0, v_i1], A[v_i0, v_i1, v_k]
            )
    for i0, i1, i2 in T.grid(T.int64(1), T.int64(1), T.int64(32000)):
        with T.block("T_softmax_exp"):
            v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
            T.reads(A[v_i0, v_i1, v_i2], T_softmax_maxelem[v_i0, v_i1])
            T.writes(T_softmax_exp[v_i0, v_i1, v_i2])
            T_softmax_exp[v_i0, v_i1, v_i2] = T.exp(
                A[v_i0, v_i1, v_i2] - T_softmax_maxelem[v_i0, v_i1]
            )
    for i0, i1, k in T.grid(T.int64(1), T.int64(1), T.int64(32000)):
        with T.block("T_softmax_expsum"):
            v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])
            T.reads(T_softmax_exp[v_i0, v_i1, v_k])
            T.writes(T_softmax_expsum[v_i0, v_i1])
            with T.init():
                T_softmax_expsum[v_i0, v_i1] = T.float32(0)
            T_softmax_expsum[v_i0, v_i1] = (
                T_softmax_expsum[v_i0, v_i1] + T_softmax_exp[v_i0, v_i1, v_k]
            )
    for i0, i1, i2 in T.grid(T.int64(1), T.int64(1), T.int64(32000)):
        with T.block("T_softmax_norm"):
            v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
            T.reads(T_softmax_exp[v_i0, v_i1, v_i2], T_softmax_expsum[v_i0, v_i1])
            T.writes(T_softmax_norm[v_i0, v_i1, v_i2])
            T.block_attr({"axis": 2})
            T_softmax_norm[v_i0, v_i1, v_i2] = (
                T_softmax_exp[v_i0, v_i1, v_i2] / T_softmax_expsum[v_i0, v_i1]
            )


DlightBench.register_bench_workload(reshape, "llama_2_7b_chat_hf_q4f16_1", "reshape")
DlightBench.register_bench_workload(
    fused_fused_decode1_take, "llama_2_7b_chat_hf_q4f16_1", "fused_fused_decode1_take"
)
DlightBench.register_bench_workload(reshape1, "llama_2_7b_chat_hf_q4f16_1", "reshape1")
DlightBench.register_bench_workload(
    fused_fused_decode2_NT_matmul,
    "llama_2_7b_chat_hf_q4f16_1",
    "fused_fused_decode2_NT_matmul",
)
DlightBench.register_bench_workload(split, "llama_2_7b_chat_hf_q4f16_1", "split")
DlightBench.register_bench_workload(reshape2, "llama_2_7b_chat_hf_q4f16_1", "reshape2")
DlightBench.register_bench_workload(
    rotary_embedding, "llama_2_7b_chat_hf_q4f16_1", "rotary_embedding"
)
DlightBench.register_bench_workload(squeeze, "llama_2_7b_chat_hf_q4f16_1", "squeeze")
DlightBench.register_bench_workload(reshape3, "llama_2_7b_chat_hf_q4f16_1", "reshape3")
DlightBench.register_bench_workload(reshape4, "llama_2_7b_chat_hf_q4f16_1", "reshape4")
DlightBench.register_bench_workload(
    fused_fused_decode3_fused_NT_matmul1_add,
    "llama_2_7b_chat_hf_q4f16_1",
    "fused_fused_decode3_fused_NT_matmul1_add",
)
DlightBench.register_bench_workload(
    fused_fused_decode4_NT_matmul2,
    "llama_2_7b_chat_hf_q4f16_1",
    "fused_fused_decode4_NT_matmul2",
)
DlightBench.register_bench_workload(
    fused_split1_silu_multiply,
    "llama_2_7b_chat_hf_q4f16_1",
    "fused_split1_silu_multiply",
)
DlightBench.register_bench_workload(
    fused_fused_decode5_fused_NT_matmul3_add,
    "llama_2_7b_chat_hf_q4f16_1",
    "fused_fused_decode5_fused_NT_matmul3_add",
)
DlightBench.register_bench_workload(slice, "llama_2_7b_chat_hf_q4f16_1", "slice")
DlightBench.register_bench_workload(
    fused_fused_decode1_fused_NT_matmul4_cast,
    "llama_2_7b_chat_hf_q4f16_1",
    "fused_fused_decode1_fused_NT_matmul4_cast",
)
DlightBench.register_bench_workload(reshape5, "llama_2_7b_chat_hf_q4f16_1", "reshape5")
DlightBench.register_bench_workload(
    fused_fused_decode1_take1, "llama_2_7b_chat_hf_q4f16_1", "fused_fused_decode1_take1"
)
DlightBench.register_bench_workload(reshape6, "llama_2_7b_chat_hf_q4f16_1", "reshape6")
DlightBench.register_bench_workload(
    fused_fused_decode2_NT_matmul5,
    "llama_2_7b_chat_hf_q4f16_1",
    "fused_fused_decode2_NT_matmul5",
)
DlightBench.register_bench_workload(
    split_rotary, "llama_2_7b_chat_hf_q4f16_1", "split_rotary"
)
DlightBench.register_bench_workload(
    fused_reshape7, "llama_2_7b_chat_hf_q4f16_1", "fused_reshape7"
)
DlightBench.register_bench_workload(
    fused_reshape7_squeeze1, "llama_2_7b_chat_hf_q4f16_1", "fused_reshape7_squeeze1"
)
DlightBench.register_bench_workload(reshape3, "llama_2_7b_chat_hf_q4f16_1", "reshape3")
DlightBench.register_bench_workload(reshape8, "llama_2_7b_chat_hf_q4f16_1", "reshape8")
DlightBench.register_bench_workload(
    fused_fused_decode3_fused_NT_matmul6_add1,
    "llama_2_7b_chat_hf_q4f16_1",
    "fused_fused_decode3_fused_NT_matmul6_add1",
)
DlightBench.register_bench_workload(
    fused_fused_decode4_NT_matmul7,
    "llama_2_7b_chat_hf_q4f16_1",
    "fused_fused_decode4_NT_matmul7",
)
DlightBench.register_bench_workload(
    fused_split2_silu1_multiply1,
    "llama_2_7b_chat_hf_q4f16_1",
    "fused_split2_silu1_multiply1",
)
DlightBench.register_bench_workload(
    fused_fused_decode5_fused_NT_matmul8_add1,
    "llama_2_7b_chat_hf_q4f16_1",
    "fused_fused_decode5_fused_NT_matmul8_add1",
)
DlightBench.register_bench_workload(slice1, "llama_2_7b_chat_hf_q4f16_1", "slice1")
DlightBench.register_bench_workload(
    fused_fused_decode1_fused_NT_matmul4_cast,
    "llama_2_7b_chat_hf_q4f16_1",
    "fused_fused_decode1_fused_NT_matmul4_cast",
)
DlightBench.register_bench_workload(divide, "llama_2_7b_chat_hf_q4f16_1", "divide")
DlightBench.register_bench_workload(softmax, "llama_2_7b_chat_hf_q4f16_1", "softmax")

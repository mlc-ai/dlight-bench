# Please save this file to dlight_bench/models and add
# `from .llama_2_7b_chat_hf_q4f16_0 import *` to dlight_bench/models/__init__.py
from dlight_bench import DlightBench
from tvm.script import tir as T


@T.prim_func
def reshape(
    A: T.Buffer((T.int64(1), T.int64(1)), "int32"),
    T_reshape: T.Buffer((T.int64(1),), "int32"),
):
    T.func_attr({"global_symbol": "reshape", "tir.noalias": T.bool(True)})
    # with T.block("root"):
    for ax0 in range(T.int64(1)):
        with T.block("T_reshape"):
            v_ax0 = T.axis.spatial(T.int64(1), ax0)
            T.reads(A[T.int64(0), T.int64(0)])
            T.writes(T_reshape[v_ax0])
            T_reshape[v_ax0] = A[T.int64(0), T.int64(0)]


@T.prim_func
def fused_fused_decode1_take(
    lv: T.Buffer((32000, 512), "uint32"),
    lv1: T.Buffer((32000, 128), "float16"),
    lv1611: T.Buffer((1,), "int32"),
    var_T_take_intermediate: T.Buffer((1, 4096), "float16"),
):
    T.func_attr(
        {"global_symbol": "fused_fused_decode1_take", "tir.noalias": T.bool(True)}
    )
    # with T.block("root"):
    for ax0, ax1 in T.grid(1, 4096):
        with T.block("T_take"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(
                lv[lv1611[v_ax0], v_ax1 // 8],
                lv1611[v_ax0],
                lv1[lv1611[v_ax0], v_ax1 // 32],
            )
            T.writes(var_T_take_intermediate[v_ax0, v_ax1])
            var_T_take_intermediate[v_ax0, v_ax1] = (
                T.Cast(
                    "float16",
                    T.bitwise_and(
                        T.shift_right(
                            lv[lv1611[v_ax0], v_ax1 // 8],
                            T.Cast("uint32", v_ax1 % 8) * T.uint32(4),
                        ),
                        T.uint32(15),
                    ),
                )
                - T.float16(7)
            ) * lv1[lv1611[v_ax0], v_ax1 // 32]


@T.prim_func
def reshape1(
    A: T.Buffer((T.int64(1), T.int64(4096)), "float16"),
    T_reshape: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"),
):
    T.func_attr({"global_symbol": "reshape1", "tir.noalias": T.bool(True)})
    # with T.block("root"):
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(4096)):
        with T.block("T_reshape"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(A[T.int64(0), v_ax2 % T.int64(4096)])
            T.writes(T_reshape[v_ax0, v_ax1, v_ax2])
            T_reshape[v_ax0, v_ax1, v_ax2] = A[T.int64(0), v_ax2 % T.int64(4096)]


@T.prim_func
def fused_fused_decode7_matmul2(
    lv3: T.Buffer((T.int64(512), T.int64(12288)), "uint32"),
    lv4: T.Buffer((T.int64(128), T.int64(12288)), "float16"),
    lv: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"),
    var_matmul_intermediate: T.Buffer(
        (T.int64(1), T.int64(1), T.int64(12288)), "float16"
    ),
):
    T.func_attr(
        {"global_symbol": "fused_fused_decode7_matmul2", "tir.noalias": T.bool(True)}
    )
    # with T.block("root"):
    p_output0_intermediate = T.alloc_buffer((T.int64(4096), T.int64(12288)), "float16")
    for i, j in T.grid(T.int64(4096), T.int64(12288)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv3[v_i // T.int64(8), v_j], lv4[v_i // T.int64(32), v_j])
            T.writes(p_output0_intermediate[v_i, v_j])
            p_output0_intermediate[v_i, v_j] = (
                T.Cast(
                    "float16",
                    T.bitwise_and(
                        T.shift_right(
                            lv3[v_i // T.int64(8), v_j],
                            T.Cast("uint32", v_i % T.int64(8)) * T.uint32(4),
                        ),
                        T.uint32(15),
                    ),
                )
                - T.float16(7)
            ) * lv4[v_i // T.int64(32), v_j]
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(12288), T.int64(4096)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv[v_i0, v_i1, v_k], p_output0_intermediate[v_k, v_i2])
            T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            var_matmul_intermediate[v_i0, v_i1, v_i2] = (
                var_matmul_intermediate[v_i0, v_i1, v_i2]
                + lv[v_i0, v_i1, v_k] * p_output0_intermediate[v_k, v_i2]
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
def fused_reshape2(
    lv_0: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"),
    var_T_reshape_intermediate: T.Buffer(
        (T.int64(1), T.int64(1), T.int64(32), T.int64(128)), "float16"
    ),
):
    T.func_attr({"global_symbol": "fused_reshape2", "tir.noalias": T.bool(True)})
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
def fused_reshape2_squeeze(
    lv_1: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"),
    var_T_squeeze_intermediate: T.Buffer(
        (T.int64(1), T.int64(32), T.int64(128)), "float16"
    ),
):
    T.func_attr(
        {"global_symbol": "fused_reshape2_squeeze", "tir.noalias": T.bool(True)}
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
    n = T.int64()
    A = T.match_buffer(var_A, (n, T.int64(32), T.int64(128)), "float16")
    T_reshape = T.match_buffer(
        var_T_reshape, (T.int64(1), n, T.int64(32), T.int64(128)), "float16"
    )
    # with T.block("root"):
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), n, T.int64(32), T.int64(128)):
        with T.block("T_reshape"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(
                A[
                    ((v_ax3 // T.int64(128) + v_ax2) // T.int64(32) + v_ax0 * n + v_ax1)
                    % n,
                    (v_ax3 // T.int64(128) + v_ax2) % T.int64(32),
                    v_ax3 % T.int64(128),
                ]
            )
            T.writes(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3])
            T_reshape[v_ax0, v_ax1, v_ax2, v_ax3] = A[
                ((v_ax3 // T.int64(128) + v_ax2) // T.int64(32) + v_ax0 * n + v_ax1)
                % n,
                (v_ax3 // T.int64(128) + v_ax2) % T.int64(32),
                v_ax3 % T.int64(128),
            ]


@T.prim_func
def reshape4(
    A: T.Buffer((T.int64(1), T.int64(1), T.int64(32), T.int64(128)), "float16"),
    T_reshape: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"),
):
    T.func_attr({"global_symbol": "reshape4", "tir.noalias": T.bool(True)})
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
def fused_fused_decode8_fused_matmul3_add(
    lv12: T.Buffer((T.int64(512), T.int64(4096)), "uint32"),
    lv13: T.Buffer((T.int64(128), T.int64(4096)), "float16"),
    lv1650: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"),
    lv1613: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"),
    p_output0_intermediate: T.Buffer(
        (T.int64(1), T.int64(1), T.int64(4096)), "float16"
    ),
):
    T.func_attr(
        {
            "global_symbol": "fused_fused_decode8_fused_matmul3_add",
            "tir.noalias": T.bool(True),
        }
    )
    # with T.block("root"):
    p_output0_intermediate_1 = T.alloc_buffer((T.int64(4096), T.int64(4096)), "float16")
    var_matmul_intermediate = T.alloc_buffer(
        (T.int64(1), T.int64(1), T.int64(4096)), "float16"
    )
    for i, j in T.grid(T.int64(4096), T.int64(4096)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv12[v_i // T.int64(8), v_j], lv13[v_i // T.int64(32), v_j])
            T.writes(p_output0_intermediate_1[v_i, v_j])
            p_output0_intermediate_1[v_i, v_j] = (
                T.Cast(
                    "float16",
                    T.bitwise_and(
                        T.shift_right(
                            lv12[v_i // T.int64(8), v_j],
                            T.Cast("uint32", v_i % T.int64(8)) * T.uint32(4),
                        ),
                        T.uint32(15),
                    ),
                )
                - T.float16(7)
            ) * lv13[v_i // T.int64(32), v_j]
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(4096), T.int64(4096)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv1650[v_i0, v_i1, v_k], p_output0_intermediate_1[v_k, v_i2])
            T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            var_matmul_intermediate[v_i0, v_i1, v_i2] = (
                var_matmul_intermediate[v_i0, v_i1, v_i2]
                + lv1650[v_i0, v_i1, v_k] * p_output0_intermediate_1[v_k, v_i2]
            )
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(4096)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(
                lv1613[v_ax0, v_ax1, v_ax2],
                var_matmul_intermediate[v_ax0, v_ax1, v_ax2],
            )
            T.writes(p_output0_intermediate[v_ax0, v_ax1, v_ax2])
            p_output0_intermediate[v_ax0, v_ax1, v_ax2] = (
                lv1613[v_ax0, v_ax1, v_ax2]
                + var_matmul_intermediate[v_ax0, v_ax1, v_ax2]
            )


@T.prim_func
def fused_fused_decode9_matmul4(
    lv16: T.Buffer((T.int64(512), T.int64(22016)), "uint32"),
    lv17: T.Buffer((T.int64(128), T.int64(22016)), "float16"),
    lv1: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"),
    var_matmul_intermediate: T.Buffer(
        (T.int64(1), T.int64(1), T.int64(22016)), "float16"
    ),
):
    T.func_attr(
        {"global_symbol": "fused_fused_decode9_matmul4", "tir.noalias": T.bool(True)}
    )
    # with T.block("root"):
    p_output0_intermediate = T.alloc_buffer((T.int64(4096), T.int64(22016)), "float16")
    for i, j in T.grid(T.int64(4096), T.int64(22016)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv16[v_i // T.int64(8), v_j], lv17[v_i // T.int64(32), v_j])
            T.writes(p_output0_intermediate[v_i, v_j])
            p_output0_intermediate[v_i, v_j] = (
                T.Cast(
                    "float16",
                    T.bitwise_and(
                        T.shift_right(
                            lv16[v_i // T.int64(8), v_j],
                            T.Cast("uint32", v_i % T.int64(8)) * T.uint32(4),
                        ),
                        T.uint32(15),
                    ),
                )
                - T.float16(7)
            ) * lv17[v_i // T.int64(32), v_j]
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(22016), T.int64(4096)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv1[v_i0, v_i1, v_k], p_output0_intermediate[v_k, v_i2])
            T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            var_matmul_intermediate[v_i0, v_i1, v_i2] = (
                var_matmul_intermediate[v_i0, v_i1, v_i2]
                + lv1[v_i0, v_i1, v_k] * p_output0_intermediate[v_k, v_i2]
            )


@T.prim_func
def fused_split_silu_multiply(
    lv1656: T.Buffer((T.int64(1), T.int64(1), T.int64(22016)), "float16"),
    var_T_multiply_intermediate: T.Buffer(
        (T.int64(1), T.int64(1), T.int64(11008)), "float16"
    ),
):
    T.func_attr(
        {"global_symbol": "fused_split_silu_multiply", "tir.noalias": T.bool(True)}
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
            T.reads(lv1656[v_ax0, v_ax1, v_ax2])
            T.writes(var_T_split_sections_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_split_sections_intermediate[v_ax0, v_ax1, v_ax2] = lv1656[
                v_ax0, v_ax1, v_ax2
            ]
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(11008)):
        with T.block("T_split_sections_1"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(lv1656[v_ax0, v_ax1, v_ax2 + T.int64(11008)])
            T.writes(var_T_split_sections_intermediate_1[v_ax0, v_ax1, v_ax2])
            var_T_split_sections_intermediate_1[v_ax0, v_ax1, v_ax2] = lv1656[
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
def fused_fused_decode10_fused_matmul5_add(
    lv20: T.Buffer((T.int64(1376), T.int64(4096)), "uint32"),
    lv21: T.Buffer((T.int64(344), T.int64(4096)), "float16"),
    lv19: T.Buffer((T.int64(1), T.int64(1), T.int64(11008)), "float16"),
    lv15: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"),
    p_output0_intermediate: T.Buffer(
        (T.int64(1), T.int64(1), T.int64(4096)), "float16"
    ),
):
    T.func_attr(
        {
            "global_symbol": "fused_fused_decode10_fused_matmul5_add",
            "tir.noalias": T.bool(True),
        }
    )
    # with T.block("root"):
    p_output0_intermediate_1 = T.alloc_buffer(
        (T.int64(11008), T.int64(4096)), "float16"
    )
    var_matmul_intermediate = T.alloc_buffer(
        (T.int64(1), T.int64(1), T.int64(4096)), "float16"
    )
    for i, j in T.grid(T.int64(11008), T.int64(4096)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv20[v_i // T.int64(8), v_j], lv21[v_i // T.int64(32), v_j])
            T.writes(p_output0_intermediate_1[v_i, v_j])
            p_output0_intermediate_1[v_i, v_j] = (
                T.Cast(
                    "float16",
                    T.bitwise_and(
                        T.shift_right(
                            lv20[v_i // T.int64(8), v_j],
                            T.Cast("uint32", v_i % T.int64(8)) * T.uint32(4),
                        ),
                        T.uint32(15),
                    ),
                )
                - T.float16(7)
            ) * lv21[v_i // T.int64(32), v_j]
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(4096), T.int64(11008)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv19[v_i0, v_i1, v_k], p_output0_intermediate_1[v_k, v_i2])
            T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            var_matmul_intermediate[v_i0, v_i1, v_i2] = (
                var_matmul_intermediate[v_i0, v_i1, v_i2]
                + lv19[v_i0, v_i1, v_k] * p_output0_intermediate_1[v_k, v_i2]
            )
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(4096)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(
                lv15[v_ax0, v_ax1, v_ax2], var_matmul_intermediate[v_ax0, v_ax1, v_ax2]
            )
            T.writes(p_output0_intermediate[v_ax0, v_ax1, v_ax2])
            p_output0_intermediate[v_ax0, v_ax1, v_ax2] = (
                lv15[v_ax0, v_ax1, v_ax2] + var_matmul_intermediate[v_ax0, v_ax1, v_ax2]
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
def fused_fused_decode11_fused_matmul6_cast(
    lv1162: T.Buffer((T.int64(512), T.int64(32000)), "uint32"),
    lv1163: T.Buffer((T.int64(128), T.int64(32000)), "float16"),
    lv1607: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"),
    p_output0_intermediate: T.Buffer(
        (T.int64(1), T.int64(1), T.int64(32000)), "float32"
    ),
):
    T.func_attr(
        {
            "global_symbol": "fused_fused_decode11_fused_matmul6_cast",
            "tir.noalias": T.bool(True),
        }
    )
    # with T.block("root"):
    p_output0_intermediate_1 = T.alloc_buffer(
        (T.int64(4096), T.int64(32000)), "float16"
    )
    var_matmul_intermediate = T.alloc_buffer(
        (T.int64(1), T.int64(1), T.int64(32000)), "float16"
    )
    for i, j in T.grid(T.int64(4096), T.int64(32000)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv1162[v_i // T.int64(8), v_j], lv1163[v_i // T.int64(32), v_j])
            T.writes(p_output0_intermediate_1[v_i, v_j])
            p_output0_intermediate_1[v_i, v_j] = (
                T.Cast(
                    "float16",
                    T.bitwise_and(
                        T.shift_right(
                            lv1162[v_i // T.int64(8), v_j],
                            T.Cast("uint32", v_i % T.int64(8)) * T.uint32(4),
                        ),
                        T.uint32(15),
                    ),
                )
                - T.float16(7)
            ) * lv1163[v_i // T.int64(32), v_j]
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(32000), T.int64(4096)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv1607[v_i0, v_i1, v_k], p_output0_intermediate_1[v_k, v_i2])
            T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            var_matmul_intermediate[v_i0, v_i1, v_i2] = (
                var_matmul_intermediate[v_i0, v_i1, v_i2]
                + lv1607[v_i0, v_i1, v_k] * p_output0_intermediate_1[v_k, v_i2]
            )
    for i0, i1, i2 in T.grid(T.int64(1), T.int64(1), T.int64(32000)):
        with T.block("compute"):
            v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
            T.reads(var_matmul_intermediate[v_i0, v_i1, v_i2])
            T.writes(p_output0_intermediate[v_i0, v_i1, v_i2])
            p_output0_intermediate[v_i0, v_i1, v_i2] = T.Cast(
                "float32", var_matmul_intermediate[v_i0, v_i1, v_i2]
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


@T.prim_func
def reshape5(var_A: T.handle, var_T_reshape: T.handle):
    T.func_attr({"global_symbol": "reshape5", "tir.noalias": T.bool(True)})
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
def fused_fused_decode1_take1(
    lv679: T.Buffer((32000, 512), "uint32"),
    lv680: T.Buffer((32000, 128), "float16"),
    p_lv: T.handle,
    p_output0: T.handle,
):
    T.func_attr(
        {"global_symbol": "fused_fused_decode1_take1", "tir.noalias": T.bool(True)}
    )
    n = T.int32()
    lv = T.match_buffer(p_lv, (n,), "int32")
    var_T_take_intermediate = T.match_buffer(p_output0, (n, 4096), "float16")
    # with T.block("root"):
    for ax0, ax1 in T.grid(n, 4096):
        with T.block("T_take"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(
                lv679[lv[v_ax0], v_ax1 // 8], lv[v_ax0], lv680[lv[v_ax0], v_ax1 // 32]
            )
            T.writes(var_T_take_intermediate[v_ax0, v_ax1])
            var_T_take_intermediate[v_ax0, v_ax1] = (
                T.Cast(
                    "float16",
                    T.bitwise_and(
                        T.shift_right(
                            lv679[lv[v_ax0], v_ax1 // 8],
                            T.Cast("uint32", v_ax1 % 8) * T.uint32(4),
                        ),
                        T.uint32(15),
                    ),
                )
                - T.float16(7)
            ) * lv680[lv[v_ax0], v_ax1 // 32]


@T.prim_func
def reshape6(var_A: T.handle, var_T_reshape: T.handle):
    T.func_attr({"global_symbol": "reshape6", "tir.noalias": T.bool(True)})
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
def fused_decode2(
    params_2: T.Buffer((T.int64(512), T.int64(12288)), "uint32"),
    params_3: T.Buffer((T.int64(128), T.int64(12288)), "float16"),
    var_T_transpose_intermediate: T.Buffer((T.int64(12288), T.int64(4096)), "float16"),
):
    T.func_attr({"global_symbol": "fused_decode2", "tir.noalias": T.bool(True)})
    # with T.block("root"):
    decode = T.alloc_buffer((T.int64(4096), T.int64(12288)), "float16")
    for i, j in T.grid(T.int64(4096), T.int64(12288)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(params_2[v_i // T.int64(8), v_j], params_3[v_i // T.int64(32), v_j])
            T.writes(decode[v_i, v_j])
            decode[v_i, v_j] = (
                T.Cast(
                    "float16",
                    T.bitwise_and(
                        T.shift_right(
                            params_2[v_i // T.int64(8), v_j],
                            T.Cast("uint32", v_i % T.int64(8)) * T.uint32(4),
                        ),
                        T.uint32(15),
                    ),
                )
                - T.float16(7)
            ) * params_3[v_i // T.int64(32), v_j]
    for ax0, ax1 in T.grid(T.int64(12288), T.int64(4096)):
        with T.block("T_transpose"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(decode[v_ax1, v_ax0])
            T.writes(var_T_transpose_intermediate[v_ax0, v_ax1])
            var_T_transpose_intermediate[v_ax0, v_ax1] = decode[v_ax1, v_ax0]


@T.prim_func
def NT_matmul(
    var_A: T.handle,
    B: T.Buffer((T.int64(12288), T.int64(4096)), "float16"),
    var_NT_matmul: T.handle,
):
    T.func_attr({"global_symbol": "NT_matmul", "tir.noalias": T.bool(True)})
    n = T.int64()
    A = T.match_buffer(var_A, (T.int64(1), n, T.int64(4096)), "float16")
    NT_matmul_1 = T.match_buffer(
        var_NT_matmul, (T.int64(1), n, T.int64(12288)), "float16"
    )
    # with T.block("root"):
    for i0, i1, i2, k in T.grid(T.int64(1), n, T.int64(12288), T.int64(4096)):
        with T.block("NT_matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(A[v_i0, v_i1, v_k], B[v_i2, v_k])
            T.writes(NT_matmul_1[v_i0, v_i1, v_i2])
            with T.init():
                NT_matmul_1[v_i0, v_i1, v_i2] = T.float16(0)
            NT_matmul_1[v_i0, v_i1, v_i2] = (
                NT_matmul_1[v_i0, v_i1, v_i2] + A[v_i0, v_i1, v_k] * B[v_i2, v_k]
            )


@T.prim_func
def split1(
    var_A: T.handle,
    var_T_split: T.handle,
    var_T_split_1: T.handle,
    var_T_split_2: T.handle,
):
    T.func_attr({"global_symbol": "split1", "tir.noalias": T.bool(True)})
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
def reshape7(var_A: T.handle, var_T_reshape: T.handle):
    T.func_attr({"global_symbol": "reshape7", "tir.noalias": T.bool(True)})
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
def squeeze1(var_A: T.handle, var_T_squeeze: T.handle):
    T.func_attr({"global_symbol": "squeeze1", "tir.noalias": T.bool(True)})
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
    n = T.int64()
    A = T.match_buffer(var_A, (n, T.int64(32), T.int64(128)), "float16")
    T_reshape = T.match_buffer(
        var_T_reshape, (T.int64(1), n, T.int64(32), T.int64(128)), "float16"
    )
    # with T.block("root"):
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), n, T.int64(32), T.int64(128)):
        with T.block("T_reshape"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(
                A[
                    ((v_ax3 // T.int64(128) + v_ax2) // T.int64(32) + v_ax0 * n + v_ax1)
                    % n,
                    (v_ax3 // T.int64(128) + v_ax2) % T.int64(32),
                    v_ax3 % T.int64(128),
                ]
            )
            T.writes(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3])
            T_reshape[v_ax0, v_ax1, v_ax2, v_ax3] = A[
                ((v_ax3 // T.int64(128) + v_ax2) // T.int64(32) + v_ax0 * n + v_ax1)
                % n,
                (v_ax3 // T.int64(128) + v_ax2) % T.int64(32),
                v_ax3 % T.int64(128),
            ]


@T.prim_func
def reshape8(var_A: T.handle, var_T_reshape: T.handle):
    T.func_attr({"global_symbol": "reshape8", "tir.noalias": T.bool(True)})
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
def fused_decode3(
    params_4: T.Buffer((T.int64(512), T.int64(4096)), "uint32"),
    params_5: T.Buffer((T.int64(128), T.int64(4096)), "float16"),
    var_T_transpose_intermediate: T.Buffer((T.int64(4096), T.int64(4096)), "float16"),
):
    T.func_attr({"global_symbol": "fused_decode3", "tir.noalias": T.bool(True)})
    # with T.block("root"):
    decode = T.alloc_buffer((T.int64(4096), T.int64(4096)), "float16")
    for i, j in T.grid(T.int64(4096), T.int64(4096)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(params_4[v_i // T.int64(8), v_j], params_5[v_i // T.int64(32), v_j])
            T.writes(decode[v_i, v_j])
            decode[v_i, v_j] = (
                T.Cast(
                    "float16",
                    T.bitwise_and(
                        T.shift_right(
                            params_4[v_i // T.int64(8), v_j],
                            T.Cast("uint32", v_i % T.int64(8)) * T.uint32(4),
                        ),
                        T.uint32(15),
                    ),
                )
                - T.float16(7)
            ) * params_5[v_i // T.int64(32), v_j]
    for ax0, ax1 in T.grid(T.int64(4096), T.int64(4096)):
        with T.block("T_transpose"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(decode[v_ax1, v_ax0])
            T.writes(var_T_transpose_intermediate[v_ax0, v_ax1])
            var_T_transpose_intermediate[v_ax0, v_ax1] = decode[v_ax1, v_ax0]


@T.prim_func
def fused_NT_matmul1_add1(
    p_lv41: T.handle,
    lv468: T.Buffer((T.int64(4096), T.int64(4096)), "float16"),
    p_lv2: T.handle,
    p_output0: T.handle,
):
    T.func_attr({"global_symbol": "fused_NT_matmul1_add1", "tir.noalias": T.bool(True)})
    n = T.int64()
    lv41 = T.match_buffer(p_lv41, (T.int64(1), n, T.int64(4096)), "float16")
    lv2 = T.match_buffer(p_lv2, (T.int64(1), n, T.int64(4096)), "float16")
    var_T_add_intermediate = T.match_buffer(
        p_output0, (T.int64(1), n, T.int64(4096)), "float16"
    )
    # with T.block("root"):
    var_NT_matmul_intermediate = T.alloc_buffer(
        (T.int64(1), n, T.int64(4096)), "float16"
    )
    for i0, i1, i2, k in T.grid(T.int64(1), n, T.int64(4096), T.int64(4096)):
        with T.block("NT_matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv41[v_i0, v_i1, v_k], lv468[v_i2, v_k])
            T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = (
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2]
                + lv41[v_i0, v_i1, v_k] * lv468[v_i2, v_k]
            )
    for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(4096)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(
                lv2[v_ax0, v_ax1, v_ax2],
                var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2],
            )
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2] = (
                lv2[v_ax0, v_ax1, v_ax2]
                + var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2]
            )


@T.prim_func
def fused_decode4(
    params_6: T.Buffer((T.int64(512), T.int64(22016)), "uint32"),
    params_7: T.Buffer((T.int64(128), T.int64(22016)), "float16"),
    var_T_transpose_intermediate: T.Buffer((T.int64(22016), T.int64(4096)), "float16"),
):
    T.func_attr({"global_symbol": "fused_decode4", "tir.noalias": T.bool(True)})
    # with T.block("root"):
    decode = T.alloc_buffer((T.int64(4096), T.int64(22016)), "float16")
    for i, j in T.grid(T.int64(4096), T.int64(22016)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(params_6[v_i // T.int64(8), v_j], params_7[v_i // T.int64(32), v_j])
            T.writes(decode[v_i, v_j])
            decode[v_i, v_j] = (
                T.Cast(
                    "float16",
                    T.bitwise_and(
                        T.shift_right(
                            params_6[v_i // T.int64(8), v_j],
                            T.Cast("uint32", v_i % T.int64(8)) * T.uint32(4),
                        ),
                        T.uint32(15),
                    ),
                )
                - T.float16(7)
            ) * params_7[v_i // T.int64(32), v_j]
    for ax0, ax1 in T.grid(T.int64(22016), T.int64(4096)):
        with T.block("T_transpose"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(decode[v_ax1, v_ax0])
            T.writes(var_T_transpose_intermediate[v_ax0, v_ax1])
            var_T_transpose_intermediate[v_ax0, v_ax1] = decode[v_ax1, v_ax0]


@T.prim_func
def NT_matmul2(
    var_A: T.handle,
    B: T.Buffer((T.int64(22016), T.int64(4096)), "float16"),
    var_NT_matmul: T.handle,
):
    T.func_attr({"global_symbol": "NT_matmul2", "tir.noalias": T.bool(True)})
    n = T.int64()
    A = T.match_buffer(var_A, (T.int64(1), n, T.int64(4096)), "float16")
    NT_matmul = T.match_buffer(
        var_NT_matmul, (T.int64(1), n, T.int64(22016)), "float16"
    )
    # with T.block("root"):
    for i0, i1, i2, k in T.grid(T.int64(1), n, T.int64(22016), T.int64(4096)):
        with T.block("NT_matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(A[v_i0, v_i1, v_k], B[v_i2, v_k])
            T.writes(NT_matmul[v_i0, v_i1, v_i2])
            with T.init():
                NT_matmul[v_i0, v_i1, v_i2] = T.float16(0)
            NT_matmul[v_i0, v_i1, v_i2] = (
                NT_matmul[v_i0, v_i1, v_i2] + A[v_i0, v_i1, v_k] * B[v_i2, v_k]
            )


@T.prim_func
def fused_split2_silu1_multiply1(p_lv2: T.handle, p_output0: T.handle):
    T.func_attr(
        {"global_symbol": "fused_split2_silu1_multiply1", "tir.noalias": T.bool(True)}
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
def fused_decode5(
    params_8: T.Buffer((T.int64(1376), T.int64(4096)), "uint32"),
    params_9: T.Buffer((T.int64(344), T.int64(4096)), "float16"),
    var_T_transpose_intermediate: T.Buffer((T.int64(4096), T.int64(11008)), "float16"),
):
    T.func_attr({"global_symbol": "fused_decode5", "tir.noalias": T.bool(True)})
    # with T.block("root"):
    decode = T.alloc_buffer((T.int64(11008), T.int64(4096)), "float16")
    for i, j in T.grid(T.int64(11008), T.int64(4096)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(params_8[v_i // T.int64(8), v_j], params_9[v_i // T.int64(32), v_j])
            T.writes(decode[v_i, v_j])
            decode[v_i, v_j] = (
                T.Cast(
                    "float16",
                    T.bitwise_and(
                        T.shift_right(
                            params_8[v_i // T.int64(8), v_j],
                            T.Cast("uint32", v_i % T.int64(8)) * T.uint32(4),
                        ),
                        T.uint32(15),
                    ),
                )
                - T.float16(7)
            ) * params_9[v_i // T.int64(32), v_j]
    for ax0, ax1 in T.grid(T.int64(4096), T.int64(11008)):
        with T.block("T_transpose"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(decode[v_ax1, v_ax0])
            T.writes(var_T_transpose_intermediate[v_ax0, v_ax1])
            var_T_transpose_intermediate[v_ax0, v_ax1] = decode[v_ax1, v_ax0]


@T.prim_func
def fused_NT_matmul3_add1(
    p_lv52: T.handle,
    lv475: T.Buffer((T.int64(4096), T.int64(11008)), "float16"),
    p_lv44: T.handle,
    p_output0: T.handle,
):
    T.func_attr({"global_symbol": "fused_NT_matmul3_add1", "tir.noalias": T.bool(True)})
    n = T.int64()
    lv52 = T.match_buffer(p_lv52, (T.int64(1), n, T.int64(11008)), "float16")
    lv44 = T.match_buffer(p_lv44, (T.int64(1), n, T.int64(4096)), "float16")
    var_T_add_intermediate = T.match_buffer(
        p_output0, (T.int64(1), n, T.int64(4096)), "float16"
    )
    # with T.block("root"):
    var_NT_matmul_intermediate = T.alloc_buffer(
        (T.int64(1), n, T.int64(4096)), "float16"
    )
    for i0, i1, i2, k in T.grid(T.int64(1), n, T.int64(4096), T.int64(11008)):
        with T.block("NT_matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv52[v_i0, v_i1, v_k], lv475[v_i2, v_k])
            T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = (
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2]
                + lv52[v_i0, v_i1, v_k] * lv475[v_i2, v_k]
            )
    for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(4096)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(
                lv44[v_ax0, v_ax1, v_ax2],
                var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2],
            )
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2] = (
                lv44[v_ax0, v_ax1, v_ax2]
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
def fused_fused_decode11_fused_matmul6_cast(
    lv1162: T.Buffer((T.int64(512), T.int64(32000)), "uint32"),
    lv1163: T.Buffer((T.int64(128), T.int64(32000)), "float16"),
    lv1607: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"),
    p_output0_intermediate: T.Buffer(
        (T.int64(1), T.int64(1), T.int64(32000)), "float32"
    ),
):
    T.func_attr(
        {
            "global_symbol": "fused_fused_decode11_fused_matmul6_cast",
            "tir.noalias": T.bool(True),
        }
    )
    # with T.block("root"):
    p_output0_intermediate_1 = T.alloc_buffer(
        (T.int64(4096), T.int64(32000)), "float16"
    )
    var_matmul_intermediate = T.alloc_buffer(
        (T.int64(1), T.int64(1), T.int64(32000)), "float16"
    )
    for i, j in T.grid(T.int64(4096), T.int64(32000)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv1162[v_i // T.int64(8), v_j], lv1163[v_i // T.int64(32), v_j])
            T.writes(p_output0_intermediate_1[v_i, v_j])
            p_output0_intermediate_1[v_i, v_j] = (
                T.Cast(
                    "float16",
                    T.bitwise_and(
                        T.shift_right(
                            lv1162[v_i // T.int64(8), v_j],
                            T.Cast("uint32", v_i % T.int64(8)) * T.uint32(4),
                        ),
                        T.uint32(15),
                    ),
                )
                - T.float16(7)
            ) * lv1163[v_i // T.int64(32), v_j]
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(32000), T.int64(4096)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv1607[v_i0, v_i1, v_k], p_output0_intermediate_1[v_k, v_i2])
            T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            var_matmul_intermediate[v_i0, v_i1, v_i2] = (
                var_matmul_intermediate[v_i0, v_i1, v_i2]
                + lv1607[v_i0, v_i1, v_k] * p_output0_intermediate_1[v_k, v_i2]
            )
    for i0, i1, i2 in T.grid(T.int64(1), T.int64(1), T.int64(32000)):
        with T.block("compute"):
            v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
            T.reads(var_matmul_intermediate[v_i0, v_i1, v_i2])
            T.writes(p_output0_intermediate[v_i0, v_i1, v_i2])
            p_output0_intermediate[v_i0, v_i1, v_i2] = T.Cast(
                "float32", var_matmul_intermediate[v_i0, v_i1, v_i2]
            )


DlightBench.register_bench_workload(reshape, "llama_2_7b_chat_hf_q4f16_0", "reshape")
DlightBench.register_bench_workload(
    fused_fused_decode1_take, "llama_2_7b_chat_hf_q4f16_0", "fused_fused_decode1_take"
)
DlightBench.register_bench_workload(reshape1, "llama_2_7b_chat_hf_q4f16_0", "reshape1")
DlightBench.register_bench_workload(
    fused_fused_decode7_matmul2,
    "llama_2_7b_chat_hf_q4f16_0",
    "fused_fused_decode7_matmul2",
)
DlightBench.register_bench_workload(
    split_rotary, "llama_2_7b_chat_hf_q4f16_0", "split_rotary"
)
DlightBench.register_bench_workload(
    fused_reshape2, "llama_2_7b_chat_hf_q4f16_0", "fused_reshape2"
)
DlightBench.register_bench_workload(
    fused_reshape2_squeeze, "llama_2_7b_chat_hf_q4f16_0", "fused_reshape2_squeeze"
)
DlightBench.register_bench_workload(reshape3, "llama_2_7b_chat_hf_q4f16_0", "reshape3")
DlightBench.register_bench_workload(reshape4, "llama_2_7b_chat_hf_q4f16_0", "reshape4")
DlightBench.register_bench_workload(
    fused_fused_decode8_fused_matmul3_add,
    "llama_2_7b_chat_hf_q4f16_0",
    "fused_fused_decode8_fused_matmul3_add",
)
DlightBench.register_bench_workload(
    fused_fused_decode9_matmul4,
    "llama_2_7b_chat_hf_q4f16_0",
    "fused_fused_decode9_matmul4",
)
DlightBench.register_bench_workload(
    fused_split_silu_multiply, "llama_2_7b_chat_hf_q4f16_0", "fused_split_silu_multiply"
)
DlightBench.register_bench_workload(
    fused_fused_decode10_fused_matmul5_add,
    "llama_2_7b_chat_hf_q4f16_0",
    "fused_fused_decode10_fused_matmul5_add",
)
DlightBench.register_bench_workload(slice1, "llama_2_7b_chat_hf_q4f16_0", "slice1")
DlightBench.register_bench_workload(
    fused_fused_decode11_fused_matmul6_cast,
    "llama_2_7b_chat_hf_q4f16_0",
    "fused_fused_decode11_fused_matmul6_cast",
)
DlightBench.register_bench_workload(divide, "llama_2_7b_chat_hf_q4f16_0", "divide")
DlightBench.register_bench_workload(softmax, "llama_2_7b_chat_hf_q4f16_0", "softmax")
DlightBench.register_bench_workload(reshape5, "llama_2_7b_chat_hf_q4f16_0", "reshape5")
DlightBench.register_bench_workload(
    fused_fused_decode1_take1, "llama_2_7b_chat_hf_q4f16_0", "fused_fused_decode1_take1"
)
DlightBench.register_bench_workload(reshape6, "llama_2_7b_chat_hf_q4f16_0", "reshape6")
DlightBench.register_bench_workload(
    fused_decode2, "llama_2_7b_chat_hf_q4f16_0", "fused_decode2"
)
DlightBench.register_bench_workload(
    NT_matmul, "llama_2_7b_chat_hf_q4f16_0", "NT_matmul"
)
DlightBench.register_bench_workload(split1, "llama_2_7b_chat_hf_q4f16_0", "split1")
DlightBench.register_bench_workload(reshape7, "llama_2_7b_chat_hf_q4f16_0", "reshape7")
DlightBench.register_bench_workload(
    rotary_embedding, "llama_2_7b_chat_hf_q4f16_0", "rotary_embedding"
)
DlightBench.register_bench_workload(squeeze1, "llama_2_7b_chat_hf_q4f16_0", "squeeze1")
DlightBench.register_bench_workload(reshape3, "llama_2_7b_chat_hf_q4f16_0", "reshape3")
DlightBench.register_bench_workload(reshape8, "llama_2_7b_chat_hf_q4f16_0", "reshape8")
DlightBench.register_bench_workload(
    fused_decode3, "llama_2_7b_chat_hf_q4f16_0", "fused_decode3"
)
DlightBench.register_bench_workload(
    fused_NT_matmul1_add1, "llama_2_7b_chat_hf_q4f16_0", "fused_NT_matmul1_add1"
)
DlightBench.register_bench_workload(
    fused_decode4, "llama_2_7b_chat_hf_q4f16_0", "fused_decode4"
)
DlightBench.register_bench_workload(
    NT_matmul2, "llama_2_7b_chat_hf_q4f16_0", "NT_matmul2"
)
DlightBench.register_bench_workload(
    fused_split2_silu1_multiply1,
    "llama_2_7b_chat_hf_q4f16_0",
    "fused_split2_silu1_multiply1",
)
DlightBench.register_bench_workload(
    fused_decode5, "llama_2_7b_chat_hf_q4f16_0", "fused_decode5"
)
DlightBench.register_bench_workload(
    fused_NT_matmul3_add1, "llama_2_7b_chat_hf_q4f16_0", "fused_NT_matmul3_add1"
)
DlightBench.register_bench_workload(slice, "llama_2_7b_chat_hf_q4f16_0", "slice")
DlightBench.register_bench_workload(
    fused_fused_decode11_fused_matmul6_cast,
    "llama_2_7b_chat_hf_q4f16_0",
    "fused_fused_decode11_fused_matmul6_cast",
)

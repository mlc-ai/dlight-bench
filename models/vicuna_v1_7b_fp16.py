from dlight_bench import DlightBench
from tvm.script import tir as T



@T.prim_func
def reshape5(A: T.Buffer((T.int64(1), T.int64(1)), "int32"), T_reshape: T.Buffer((T.int64(1),), "int32")):
    T.func_attr({"op_pattern": 1, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    for ax0 in range(T.int64(1)):
        with T.block("T_reshape"):
            v_ax0 = T.axis.spatial(T.int64(1), ax0)
            T.reads(A[T.int64(0), T.int64(0)])
            T.writes(T_reshape[v_ax0])
            T_reshape[v_ax0] = A[T.int64(0), T.int64(0)]


@T.prim_func
def take1(A: T.Buffer((T.int64(32000), T.int64(4096)), "float16"), B: T.Buffer((T.int64(1),), "int32"), T_take: T.Buffer((T.int64(1), T.int64(4096)), "float16")):
    T.func_attr({"op_pattern": 8, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    for ax0, ax1 in T.grid(T.int64(1), T.int64(4096)):
        with T.block("T_take"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(A[B[v_ax0], v_ax1], B[v_ax0])
            T.writes(T_take[v_ax0, v_ax1])
            T_take[v_ax0, v_ax1] = A[B[v_ax0], v_ax1]


@T.prim_func
def reshape6(A: T.Buffer((T.int64(1), T.int64(4096)), "float16"), T_reshape: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16")):
    T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(4096)):
        with T.block("T_reshape"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(A[T.int64(0), v_ax2 % T.int64(4096)])
            T.writes(T_reshape[v_ax0, v_ax1, v_ax2])
            T_reshape[v_ax0, v_ax1, v_ax2] = A[T.int64(0), v_ax2 % T.int64(4096)]


@T.prim_func
def full(var_T_full: T.handle):
    T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
    n = T.int64()
    T_full = T.match_buffer(var_T_full, (T.int64(1), T.int64(1), T.int64(1), n), "float16")
    # with T.block("root"):
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(1), T.int64(1), n):
        with T.block("T_full"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads()
            T.writes(T_full[v_ax0, v_ax1, v_ax2, v_ax3])
            T_full[v_ax0, v_ax1, v_ax2, v_ax3] = T.float16(0)


@T.prim_func
def rms_norm1(A: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"), B: T.Buffer((T.int64(4096),), "float16"), rms_norm: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16")):
    T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    Ared_temp = T.alloc_buffer((T.int64(1), T.int64(1)))
    for bsz, i, k in T.grid(T.int64(1), T.int64(1), T.int64(4096)):
        with T.block("Ared_temp"):
            v_bsz, v_i, v_k = T.axis.remap("SSR", [bsz, i, k])
            T.reads(A[v_bsz, v_i, v_k])
            T.writes(Ared_temp[v_bsz, v_i])
            with T.init():
                Ared_temp[v_bsz, v_i] = T.float32(0)
            Ared_temp[v_bsz, v_i] = Ared_temp[v_bsz, v_i] + T.Cast("float32", A[v_bsz, v_i, v_k]) * T.Cast("float32", A[v_bsz, v_i, v_k])
    for bsz, i, k in T.grid(T.int64(1), T.int64(1), T.int64(4096)):
        with T.block("rms_norm"):
            v_bsz, v_i, v_k = T.axis.remap("SSS", [bsz, i, k])
            T.reads(B[v_k], A[v_bsz, v_i, v_k], Ared_temp[v_bsz, v_i])
            T.writes(rms_norm[v_bsz, v_i, v_k])
            rms_norm[v_bsz, v_i, v_k] = T.Cast("float16", T.Cast("float32", B[v_k]) * (T.Cast("float32", A[v_bsz, v_i, v_k]) / T.sqrt(Ared_temp[v_bsz, v_i] * T.float32(0.000244140625) + T.float32(9.9999999999999995e-07))))


@T.prim_func
def reshape7(A: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"), T_reshape: T.Buffer((T.int64(1), T.int64(1), T.int64(32), T.int64(128)), "float16")):
    T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(1), T.int64(32), T.int64(128)):
        with T.block("T_reshape"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(A[T.int64(0), T.int64(0), (v_ax2 * T.int64(128) + v_ax3) % T.int64(4096)])
            T.writes(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3])
            T_reshape[v_ax0, v_ax1, v_ax2, v_ax3] = A[T.int64(0), T.int64(0), (v_ax2 * T.int64(128) + v_ax3) % T.int64(4096)]


@T.prim_func
def fused_reshape7_squeeze1(lv195: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"), var_T_squeeze_intermediate: T.Buffer((T.int64(1), T.int64(32), T.int64(128)), "float16")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_T_reshape_intermediate = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(32), T.int64(128)), "float16")
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(1), T.int64(32), T.int64(128)):
        with T.block("T_reshape"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(lv195[T.int64(0), T.int64(0), (v_ax2 * T.int64(128) + v_ax3) % T.int64(4096)])
            T.writes(var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = lv195[T.int64(0), T.int64(0), (v_ax2 * T.int64(128) + v_ax3) % T.int64(4096)]
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(32), T.int64(128)):
        with T.block("T_squeeze"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_T_reshape_intermediate[T.int64(0), v_ax0, v_ax1, v_ax2])
            T.writes(var_T_squeeze_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_squeeze_intermediate[v_ax0, v_ax1, v_ax2] = var_T_reshape_intermediate[T.int64(0), v_ax0, v_ax1, v_ax2]


@T.prim_func
def rotary_embedding1(A: T.Buffer((T.int64(1), T.int64(1), T.int64(32), T.int64(128)), "float16"), B: T.Buffer((T.int64(2048), T.int64(128)), "float16"), C: T.Buffer((T.int64(2048), T.int64(128)), "float16"), rotary: T.Buffer((T.int64(1), T.int64(1), T.int64(32), T.int64(128)), "float16"), n: T.int64):
    T.func_attr({"op_pattern": 8, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(1), T.int64(32), T.int64(128)):
        with T.block("rotary"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(B[n + v_i1 - T.int64(1), v_i3], A[v_i0, v_i1, v_i2, v_i3 - T.int64(64):v_i3 - T.int64(64) + T.int64(129)], C[n + v_i1 - T.int64(1), v_i3])
            T.writes(rotary[v_i0, v_i1, v_i2, v_i3])
            rotary[v_i0, v_i1, v_i2, v_i3] = B[n + v_i1 - T.int64(1), v_i3] * A[v_i0, v_i1, v_i2, v_i3] + C[n + v_i1 - T.int64(1), v_i3] * T.Select(T.int64(64) <= v_i3, A[v_i0, v_i1, v_i2, v_i3 - T.int64(64)], A[v_i0, v_i1, v_i2, v_i3 + T.int64(64)] * T.float16(-1))


@T.prim_func
def squeeze1(A: T.Buffer((T.int64(1), T.int64(1), T.int64(32), T.int64(128)), "float16"), T_squeeze: T.Buffer((T.int64(1), T.int64(32), T.int64(128)), "float16")):
    T.func_attr({"op_pattern": 1, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(32), T.int64(128)):
        with T.block("T_squeeze"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(A[T.int64(0), v_ax0, v_ax1, v_ax2])
            T.writes(T_squeeze[v_ax0, v_ax1, v_ax2])
            T_squeeze[v_ax0, v_ax1, v_ax2] = A[T.int64(0), v_ax0, v_ax1, v_ax2]


@T.prim_func
def reshape3(var_A: T.handle, var_T_reshape: T.handle):
    T.func_attr({"op_pattern": 8, "tir.noalias": T.bool(True)})
    m = T.int64()
    A = T.match_buffer(var_A, (m, T.int64(32), T.int64(128)), "float16")
    T_reshape = T.match_buffer(var_T_reshape, (T.int64(1), m, T.int64(32), T.int64(128)), "float16")
    # with T.block("root"):
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), m, T.int64(32), T.int64(128)):
        with T.block("T_reshape"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(A[((v_ax3 // T.int64(128) + v_ax2) // T.int64(32) + v_ax0 * m + v_ax1) % m, (v_ax3 // T.int64(128) + v_ax2) % T.int64(32), v_ax3 % T.int64(128)])
            T.writes(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3])
            T_reshape[v_ax0, v_ax1, v_ax2, v_ax3] = A[((v_ax3 // T.int64(128) + v_ax2) // T.int64(32) + v_ax0 * m + v_ax1) % m, (v_ax3 // T.int64(128) + v_ax2) % T.int64(32), v_ax3 % T.int64(128)]


@T.prim_func
def transpose2(A: T.Buffer((T.int64(1), T.int64(1), T.int64(32), T.int64(128)), "float16"), T_transpose: T.Buffer((T.int64(1), T.int64(32), T.int64(1), T.int64(128)), "float16")):
    T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(32), T.int64(1), T.int64(128)):
        with T.block("T_transpose"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(A[v_ax0, v_ax2, v_ax1, v_ax3])
            T.writes(T_transpose[v_ax0, v_ax1, v_ax2, v_ax3])
            T_transpose[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax2, v_ax1, v_ax3]


@T.prim_func
def transpose(var_A: T.handle, var_T_transpose: T.handle):
    T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})
    n = T.int64()
    A = T.match_buffer(var_A, (T.int64(1), n, T.int64(32), T.int64(128)), "float16")
    T_transpose = T.match_buffer(var_T_transpose, (T.int64(1), T.int64(32), n, T.int64(128)), "float16")
    # with T.block("root"):
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(32), n, T.int64(128)):
        with T.block("T_transpose"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(A[v_ax0, v_ax2, v_ax1, v_ax3])
            T.writes(T_transpose[v_ax0, v_ax1, v_ax2, v_ax3])
            T_transpose[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax2, v_ax1, v_ax3]


@T.prim_func
def fused_divide1_add1(p_lv196: T.handle, p_lv1486: T.handle, p_output0: T.handle):
    T.func_attr({"tir.noalias": T.bool(True)})
    n = T.int64()
    lv196 = T.match_buffer(p_lv196, (T.int64(1), T.int64(32), T.int64(1), n), "float16")
    lv1486 = T.match_buffer(p_lv1486, (T.int64(1), T.int64(1), T.int64(1), n), "float16")
    var_T_add_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(32), T.int64(1), n), "float16")
    # with T.block("root"):
    var_T_divide_intermediate = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(1), n), "float16")
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(32), T.int64(1), n):
        with T.block("T_divide"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(lv196[v_ax0, v_ax1, v_ax2, v_ax3])
            T.writes(var_T_divide_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_divide_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = lv196[v_ax0, v_ax1, v_ax2, v_ax3] * T.float16(0.088397790055248615)
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(32), T.int64(1), n):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_divide_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], lv1486[v_ax0, T.int64(0), v_ax2, v_ax3])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_T_divide_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] + lv1486[v_ax0, T.int64(0), v_ax2, v_ax3]


@T.prim_func
def softmax1(var_A: T.handle, var_T_softmax_norm: T.handle):
    T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True)})
    n = T.int64()
    A = T.match_buffer(var_A, (T.int64(1), T.int64(32), T.int64(1), n), "float16")
    T_softmax_norm = T.match_buffer(var_T_softmax_norm, (T.int64(1), T.int64(32), T.int64(1), n), "float16")
    # with T.block("root"):
    T_softmax_maxelem = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(1)), "float16")
    T_softmax_exp = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(1), n), "float16")
    T_softmax_expsum = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(1)), "float16")
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(32), T.int64(1), n):
        with T.block("T_softmax_maxelem"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(A[v_i0, v_i1, v_i2, v_k])
            T.writes(T_softmax_maxelem[v_i0, v_i1, v_i2])
            with T.init():
                T_softmax_maxelem[v_i0, v_i1, v_i2] = T.float16(-65504)
            T_softmax_maxelem[v_i0, v_i1, v_i2] = T.max(T_softmax_maxelem[v_i0, v_i1, v_i2], A[v_i0, v_i1, v_i2, v_k])
    for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(32), T.int64(1), n):
        with T.block("T_softmax_exp"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(A[v_i0, v_i1, v_i2, v_i3], T_softmax_maxelem[v_i0, v_i1, v_i2])
            T.writes(T_softmax_exp[v_i0, v_i1, v_i2, v_i3])
            T_softmax_exp[v_i0, v_i1, v_i2, v_i3] = T.exp(A[v_i0, v_i1, v_i2, v_i3] - T_softmax_maxelem[v_i0, v_i1, v_i2])
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(32), T.int64(1), n):
        with T.block("T_softmax_expsum"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(T_softmax_exp[v_i0, v_i1, v_i2, v_k])
            T.writes(T_softmax_expsum[v_i0, v_i1, v_i2])
            with T.init():
                T_softmax_expsum[v_i0, v_i1, v_i2] = T.float16(0)
            T_softmax_expsum[v_i0, v_i1, v_i2] = T_softmax_expsum[v_i0, v_i1, v_i2] + T_softmax_exp[v_i0, v_i1, v_i2, v_k]
    for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(32), T.int64(1), n):
        with T.block("T_softmax_norm"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(T_softmax_exp[v_i0, v_i1, v_i2, v_i3], T_softmax_expsum[v_i0, v_i1, v_i2])
            T.writes(T_softmax_norm[v_i0, v_i1, v_i2, v_i3])
            T.block_attr({"axis": 3})
            T_softmax_norm[v_i0, v_i1, v_i2, v_i3] = T_softmax_exp[v_i0, v_i1, v_i2, v_i3] / T_softmax_expsum[v_i0, v_i1, v_i2]


@T.prim_func
def matmul1(var_A: T.handle, var_B: T.handle, matmul: T.Buffer((T.int64(1), T.int64(32), T.int64(1), T.int64(128)), "float16")):
    T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True)})
    n = T.int64()
    A = T.match_buffer(var_A, (T.int64(1), T.int64(32), T.int64(1), n), "float16")
    B = T.match_buffer(var_B, (T.int64(1), T.int64(32), n, T.int64(128)), "float16")
    # with T.block("root"):
    for i0, i1, i2, i3, k in T.grid(T.int64(1), T.int64(32), T.int64(1), T.int64(128), n):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_i3, v_k = T.axis.remap("SSSSR", [i0, i1, i2, i3, k])
            T.reads(A[v_i0, v_i1, v_i2, v_k], B[v_i0, v_i1, v_k, v_i3])
            T.writes(matmul[v_i0, v_i1, v_i2, v_i3])
            with T.init():
                matmul[v_i0, v_i1, v_i2, v_i3] = T.float16(0)
            matmul[v_i0, v_i1, v_i2, v_i3] = matmul[v_i0, v_i1, v_i2, v_i3] + A[v_i0, v_i1, v_i2, v_k] * B[v_i0, v_i1, v_k, v_i3]


@T.prim_func
def fused_transpose3_reshape8(lv1517: T.Buffer((T.int64(1), T.int64(32), T.int64(1), T.int64(128)), "float16"), var_T_reshape_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_T_transpose_intermediate = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(32), T.int64(128)), "float16")
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(1), T.int64(32), T.int64(128)):
        with T.block("T_transpose"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(lv1517[v_ax0, v_ax2, v_ax1, v_ax3])
            T.writes(var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = lv1517[v_ax0, v_ax2, v_ax1, v_ax3]
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(4096)):
        with T.block("T_reshape"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_T_transpose_intermediate[T.int64(0), T.int64(0), v_ax2 % T.int64(4096) // T.int64(128), v_ax2 % T.int64(128)])
            T.writes(var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2] = var_T_transpose_intermediate[T.int64(0), T.int64(0), v_ax2 % T.int64(4096) // T.int64(128), v_ax2 % T.int64(128)]


@T.prim_func
def fused_silu1_multiply1(lv197: T.Buffer((T.int64(1), T.int64(1), T.int64(11008)), "float16"), lv198: T.Buffer((T.int64(1), T.int64(1), T.int64(11008)), "float16"), var_T_multiply_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(11008)), "float16")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    compute = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(11008)), "float16")
    var_T_multiply_intermediate_1 = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(11008)), "float16")
    for i0, i1, i2 in T.grid(T.int64(1), T.int64(1), T.int64(11008)):
        with T.block("compute"):
            v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
            T.reads(lv197[v_i0, v_i1, v_i2])
            T.writes(compute[v_i0, v_i1, v_i2])
            compute[v_i0, v_i1, v_i2] = T.sigmoid(lv197[v_i0, v_i1, v_i2])
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(11008)):
        with T.block("T_multiply"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(lv197[v_ax0, v_ax1, v_ax2], compute[v_ax0, v_ax1, v_ax2])
            T.writes(var_T_multiply_intermediate_1[v_ax0, v_ax1, v_ax2])
            var_T_multiply_intermediate_1[v_ax0, v_ax1, v_ax2] = lv197[v_ax0, v_ax1, v_ax2] * compute[v_ax0, v_ax1, v_ax2]
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(1), T.int64(11008)):
        with T.block("T_multiply_1"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_T_multiply_intermediate_1[v_ax0, v_ax1, v_ax2], lv198[v_ax0, v_ax1, v_ax2])
            T.writes(var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2] = var_T_multiply_intermediate_1[v_ax0, v_ax1, v_ax2] * lv198[v_ax0, v_ax1, v_ax2]


@T.prim_func
def slice1(A: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"), slice: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16")):
    T.func_attr({"op_pattern": 1, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    for i, j, k in T.grid(T.int64(1), T.int64(1), T.int64(4096)):
        with T.block("slice"):
            v_i, v_j, v_k = T.axis.remap("SSS", [i, j, k])
            T.reads(A[v_i, T.int64(0), v_k])
            T.writes(slice[v_i, v_j, v_k])
            slice[v_i, v_j, v_k] = A[v_i, T.int64(0), v_k]


@T.prim_func
def cast(A: T.Buffer((T.int64(1), T.int64(1), T.int64(32000)), "float16"), compute: T.Buffer((T.int64(1), T.int64(1), T.int64(32000)), "float32")):
    T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    for i0, i1, i2 in T.grid(T.int64(1), T.int64(1), T.int64(32000)):
        with T.block("compute"):
            v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
            T.reads(A[v_i0, v_i1, v_i2])
            T.writes(compute[v_i0, v_i1, v_i2])
            compute[v_i0, v_i1, v_i2] = T.Cast("float32", A[v_i0, v_i1, v_i2])


@T.prim_func
def reshape(var_A: T.handle, var_T_reshape: T.handle):
    T.func_attr({"op_pattern": 8, "tir.noalias": T.bool(True)})
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
def take(A: T.Buffer((T.int64(32000), T.int64(4096)), "float16"), var_B: T.handle, var_T_take: T.handle):
    T.func_attr({"op_pattern": 8, "tir.noalias": T.bool(True)})
    n = T.int64()
    B = T.match_buffer(var_B, (n,), "int32")
    T_take = T.match_buffer(var_T_take, (n, T.int64(4096)), "float16")
    # with T.block("root"):
    for ax0, ax1 in T.grid(n, T.int64(4096)):
        with T.block("T_take"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(A[B[v_ax0], v_ax1], B[v_ax0])
            T.writes(T_take[v_ax0, v_ax1])
            T_take[v_ax0, v_ax1] = A[B[v_ax0], v_ax1]


@T.prim_func
def reshape1(var_A: T.handle, var_T_reshape: T.handle):
    T.func_attr({"op_pattern": 8, "tir.noalias": T.bool(True)})
    n = T.int64()
    A = T.match_buffer(var_A, (n, T.int64(4096)), "float16")
    T_reshape = T.match_buffer(var_T_reshape, (T.int64(1), n, T.int64(4096)), "float16")
    # with T.block("root"):
    for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(4096)):
        with T.block("T_reshape"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(A[(v_ax2 // T.int64(4096) + v_ax0 * n + v_ax1) % n, v_ax2 % T.int64(4096)])
            T.writes(T_reshape[v_ax0, v_ax1, v_ax2])
            T_reshape[v_ax0, v_ax1, v_ax2] = A[(v_ax2 // T.int64(4096) + v_ax0 * n + v_ax1) % n, v_ax2 % T.int64(4096)]


@T.prim_func
def fused_min_max_triu_te_broadcast_to(p_output0: T.handle, n: T.int64):
    T.func_attr({"tir.noalias": T.bool(True)})
    var_T_broadcast_to_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(1), n, n), "float16")
    # with T.block("root"):
    var_make_diag_mask_te_intermediate = T.alloc_buffer((n, n), "float16")
    for i, j in T.grid(n, n):
        with T.block("make_diag_mask_te"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads()
            T.writes(var_make_diag_mask_te_intermediate[v_i, v_j])
            var_make_diag_mask_te_intermediate[v_i, v_j] = T.Select(v_i < v_j, T.float16(-65504), T.float16(0))
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(1), n, n):
        with T.block("T_broadcast_to"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_make_diag_mask_te_intermediate[v_ax2, v_ax3])
            T.writes(var_T_broadcast_to_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_broadcast_to_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_make_diag_mask_te_intermediate[v_ax2, v_ax3]


@T.prim_func
def extend_te(var_A: T.handle, var_concat_te: T.handle):
    T.func_attr({"op_pattern": 8, "tir.noalias": T.bool(True)})
    n = T.int64()
    A = T.match_buffer(var_A, (T.int64(1), T.int64(1), n, n), "float16")
    m = T.int64()
    concat_te = T.match_buffer(var_concat_te, (T.int64(1), T.int64(1), n, m), "float16")
    # with T.block("root"):
    for b, _, i, j in T.grid(T.int64(1), T.int64(1), n, m):
        with T.block("concat_te"):
            v_b, v__, v_i, v_j = T.axis.remap("SSSS", [b, _, i, j])
            T.reads(A[v_b, v__, v_i, v_j + n - m])
            T.writes(concat_te[v_b, v__, v_i, v_j])
            concat_te[v_b, v__, v_i, v_j] = T.if_then_else(v_j < m - n, T.float16(0), A[v_b, v__, v_i, v_j + n - m])


@T.prim_func
def rms_norm(var_A: T.handle, B: T.Buffer((T.int64(4096),), "float16"), var_rms_norm: T.handle):
    T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True)})
    n = T.int64()
    A = T.match_buffer(var_A, (T.int64(1), n, T.int64(4096)), "float16")
    rms_norm_1 = T.match_buffer(var_rms_norm, (T.int64(1), n, T.int64(4096)), "float16")
    # with T.block("root"):
    Ared_temp = T.alloc_buffer((T.int64(1), n))
    for bsz, i, k in T.grid(T.int64(1), n, T.int64(4096)):
        with T.block("Ared_temp"):
            v_bsz, v_i, v_k = T.axis.remap("SSR", [bsz, i, k])
            T.reads(A[v_bsz, v_i, v_k])
            T.writes(Ared_temp[v_bsz, v_i])
            with T.init():
                Ared_temp[v_bsz, v_i] = T.float32(0)
            Ared_temp[v_bsz, v_i] = Ared_temp[v_bsz, v_i] + T.Cast("float32", A[v_bsz, v_i, v_k]) * T.Cast("float32", A[v_bsz, v_i, v_k])
    for bsz, i, k in T.grid(T.int64(1), n, T.int64(4096)):
        with T.block("rms_norm"):
            v_bsz, v_i, v_k = T.axis.remap("SSS", [bsz, i, k])
            T.reads(B[v_k], A[v_bsz, v_i, v_k], Ared_temp[v_bsz, v_i])
            T.writes(rms_norm_1[v_bsz, v_i, v_k])
            rms_norm_1[v_bsz, v_i, v_k] = T.Cast("float16", T.Cast("float32", B[v_k]) * (T.Cast("float32", A[v_bsz, v_i, v_k]) / T.sqrt(Ared_temp[v_bsz, v_i] * T.float32(0.000244140625) + T.float32(9.9999999999999995e-07))))


@T.prim_func
def reshape2(var_A: T.handle, var_T_reshape: T.handle):
    T.func_attr({"op_pattern": 8, "tir.noalias": T.bool(True)})
    n = T.int64()
    A = T.match_buffer(var_A, (T.int64(1), n, T.int64(4096)), "float16")
    T_reshape = T.match_buffer(var_T_reshape, (T.int64(1), n, T.int64(32), T.int64(128)), "float16")
    # with T.block("root"):
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), n, T.int64(32), T.int64(128)):
        with T.block("T_reshape"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(A[T.int64(0), ((v_ax2 * T.int64(128) + v_ax3) // T.int64(4096) + v_ax0 * n + v_ax1) % n, (v_ax2 * T.int64(128) + v_ax3) % T.int64(4096)])
            T.writes(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3])
            T_reshape[v_ax0, v_ax1, v_ax2, v_ax3] = A[T.int64(0), ((v_ax2 * T.int64(128) + v_ax3) // T.int64(4096) + v_ax0 * n + v_ax1) % n, (v_ax2 * T.int64(128) + v_ax3) % T.int64(4096)]


@T.prim_func
def rotary_embedding(var_A: T.handle, B: T.Buffer((T.int64(2048), T.int64(128)), "float16"), C: T.Buffer((T.int64(2048), T.int64(128)), "float16"), var_rotary: T.handle, m: T.int64):
    T.func_attr({"op_pattern": 8, "tir.noalias": T.bool(True)})
    n = T.int64()
    A = T.match_buffer(var_A, (T.int64(1), n, T.int64(32), T.int64(128)), "float16")
    rotary = T.match_buffer(var_rotary, (T.int64(1), n, T.int64(32), T.int64(128)), "float16")
    # with T.block("root"):
    for i0, i1, i2, i3 in T.grid(T.int64(1), n, T.int64(32), T.int64(128)):
        with T.block("rotary"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(B[m + v_i1 - n, v_i3], A[v_i0, v_i1, v_i2, v_i3 - T.int64(64):v_i3 - T.int64(64) + T.int64(129)], C[m + v_i1 - n, v_i3])
            T.writes(rotary[v_i0, v_i1, v_i2, v_i3])
            rotary[v_i0, v_i1, v_i2, v_i3] = B[m + v_i1 - n, v_i3] * A[v_i0, v_i1, v_i2, v_i3] + C[m + v_i1 - n, v_i3] * T.Select(T.int64(64) <= v_i3, A[v_i0, v_i1, v_i2, v_i3 - T.int64(64)], A[v_i0, v_i1, v_i2, v_i3 + T.int64(64)] * T.float16(-1))


@T.prim_func
def squeeze(var_A: T.handle, var_T_squeeze: T.handle):
    T.func_attr({"op_pattern": 1, "tir.noalias": T.bool(True)})
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
    T.func_attr({"op_pattern": 8, "tir.noalias": T.bool(True)})
    m = T.int64()
    A = T.match_buffer(var_A, (m, T.int64(32), T.int64(128)), "float16")
    T_reshape = T.match_buffer(var_T_reshape, (T.int64(1), m, T.int64(32), T.int64(128)), "float16")
    # with T.block("root"):
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), m, T.int64(32), T.int64(128)):
        with T.block("T_reshape"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(A[((v_ax3 // T.int64(128) + v_ax2) // T.int64(32) + v_ax0 * m + v_ax1) % m, (v_ax3 // T.int64(128) + v_ax2) % T.int64(32), v_ax3 % T.int64(128)])
            T.writes(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3])
            T_reshape[v_ax0, v_ax1, v_ax2, v_ax3] = A[((v_ax3 // T.int64(128) + v_ax2) // T.int64(32) + v_ax0 * m + v_ax1) % m, (v_ax3 // T.int64(128) + v_ax2) % T.int64(32), v_ax3 % T.int64(128)]


@T.prim_func
def transpose(var_A: T.handle, var_T_transpose: T.handle):
    T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})
    n = T.int64()
    A = T.match_buffer(var_A, (T.int64(1), n, T.int64(32), T.int64(128)), "float16")
    T_transpose = T.match_buffer(var_T_transpose, (T.int64(1), T.int64(32), n, T.int64(128)), "float16")
    # with T.block("root"):
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(32), n, T.int64(128)):
        with T.block("T_transpose"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(A[v_ax0, v_ax2, v_ax1, v_ax3])
            T.writes(T_transpose[v_ax0, v_ax1, v_ax2, v_ax3])
            T_transpose[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax2, v_ax1, v_ax3]


@T.prim_func
def fused_divide_add(p_lv3: T.handle, p_lv5: T.handle, p_output0: T.handle):
    T.func_attr({"tir.noalias": T.bool(True)})
    n, m = T.int64(), T.int64()
    lv3 = T.match_buffer(p_lv3, (T.int64(1), T.int64(32), n, m), "float16")
    lv5 = T.match_buffer(p_lv5, (T.int64(1), T.int64(1), n, m), "float16")
    var_T_add_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(32), n, m), "float16")
    # with T.block("root"):
    var_T_divide_intermediate = T.alloc_buffer((T.int64(1), T.int64(32), n, m), "float16")
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(32), n, m):
        with T.block("T_divide"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(lv3[v_ax0, v_ax1, v_ax2, v_ax3])
            T.writes(var_T_divide_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_divide_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = lv3[v_ax0, v_ax1, v_ax2, v_ax3] * T.float16(0.088397790055248615)
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(32), n, m):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_divide_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], lv5[v_ax0, T.int64(0), v_ax2, v_ax3])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_T_divide_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] + lv5[v_ax0, T.int64(0), v_ax2, v_ax3]


@T.prim_func
def softmax(var_A: T.handle, var_T_softmax_norm: T.handle):
    T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True)})
    n, m = T.int64(), T.int64()
    A = T.match_buffer(var_A, (T.int64(1), T.int64(32), n, m), "float16")
    T_softmax_norm = T.match_buffer(var_T_softmax_norm, (T.int64(1), T.int64(32), n, m), "float16")
    # with T.block("root"):
    T_softmax_maxelem = T.alloc_buffer((T.int64(1), T.int64(32), n), "float16")
    T_softmax_exp = T.alloc_buffer((T.int64(1), T.int64(32), n, m), "float16")
    T_softmax_expsum = T.alloc_buffer((T.int64(1), T.int64(32), n), "float16")
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(32), n, m):
        with T.block("T_softmax_maxelem"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(A[v_i0, v_i1, v_i2, v_k])
            T.writes(T_softmax_maxelem[v_i0, v_i1, v_i2])
            with T.init():
                T_softmax_maxelem[v_i0, v_i1, v_i2] = T.float16(-65504)
            T_softmax_maxelem[v_i0, v_i1, v_i2] = T.max(T_softmax_maxelem[v_i0, v_i1, v_i2], A[v_i0, v_i1, v_i2, v_k])
    for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(32), n, m):
        with T.block("T_softmax_exp"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(A[v_i0, v_i1, v_i2, v_i3], T_softmax_maxelem[v_i0, v_i1, v_i2])
            T.writes(T_softmax_exp[v_i0, v_i1, v_i2, v_i3])
            T_softmax_exp[v_i0, v_i1, v_i2, v_i3] = T.exp(A[v_i0, v_i1, v_i2, v_i3] - T_softmax_maxelem[v_i0, v_i1, v_i2])
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(32), n, m):
        with T.block("T_softmax_expsum"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(T_softmax_exp[v_i0, v_i1, v_i2, v_k])
            T.writes(T_softmax_expsum[v_i0, v_i1, v_i2])
            with T.init():
                T_softmax_expsum[v_i0, v_i1, v_i2] = T.float16(0)
            T_softmax_expsum[v_i0, v_i1, v_i2] = T_softmax_expsum[v_i0, v_i1, v_i2] + T_softmax_exp[v_i0, v_i1, v_i2, v_k]
    for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(32), n, m):
        with T.block("T_softmax_norm"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(T_softmax_exp[v_i0, v_i1, v_i2, v_i3], T_softmax_expsum[v_i0, v_i1, v_i2])
            T.writes(T_softmax_norm[v_i0, v_i1, v_i2, v_i3])
            T.block_attr({"axis": 3})
            T_softmax_norm[v_i0, v_i1, v_i2, v_i3] = T_softmax_exp[v_i0, v_i1, v_i2, v_i3] / T_softmax_expsum[v_i0, v_i1, v_i2]


@T.prim_func
def matmul(var_A: T.handle, var_B: T.handle, var_matmul: T.handle):
    T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True)})
    n, m = T.int64(), T.int64()
    A = T.match_buffer(var_A, (T.int64(1), T.int64(32), n, m), "float16")
    B = T.match_buffer(var_B, (T.int64(1), T.int64(32), m, T.int64(128)), "float16")
    matmul_1 = T.match_buffer(var_matmul, (T.int64(1), T.int64(32), n, T.int64(128)), "float16")
    # with T.block("root"):
    for i0, i1, i2, i3, k in T.grid(T.int64(1), T.int64(32), n, T.int64(128), m):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_i3, v_k = T.axis.remap("SSSSR", [i0, i1, i2, i3, k])
            T.reads(A[v_i0, v_i1, v_i2, v_k], B[v_i0, v_i1, v_k, v_i3])
            T.writes(matmul_1[v_i0, v_i1, v_i2, v_i3])
            with T.init():
                matmul_1[v_i0, v_i1, v_i2, v_i3] = T.float16(0)
            matmul_1[v_i0, v_i1, v_i2, v_i3] = matmul_1[v_i0, v_i1, v_i2, v_i3] + A[v_i0, v_i1, v_i2, v_k] * B[v_i0, v_i1, v_k, v_i3]


@T.prim_func
def transpose1(var_A: T.handle, var_T_transpose: T.handle):
    T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})
    n = T.int64()
    A = T.match_buffer(var_A, (T.int64(1), T.int64(32), n, T.int64(128)), "float16")
    T_transpose = T.match_buffer(var_T_transpose, (T.int64(1), n, T.int64(32), T.int64(128)), "float16")
    # with T.block("root"):
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), n, T.int64(32), T.int64(128)):
        with T.block("T_transpose"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(A[v_ax0, v_ax2, v_ax1, v_ax3])
            T.writes(T_transpose[v_ax0, v_ax1, v_ax2, v_ax3])
            T_transpose[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax2, v_ax1, v_ax3]


@T.prim_func
def reshape4(var_A: T.handle, var_T_reshape: T.handle):
    T.func_attr({"op_pattern": 8, "tir.noalias": T.bool(True)})
    n = T.int64()
    A = T.match_buffer(var_A, (T.int64(1), n, T.int64(32), T.int64(128)), "float16")
    T_reshape = T.match_buffer(var_T_reshape, (T.int64(1), n, T.int64(4096)), "float16")
    # with T.block("root"):
    for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(4096)):
        with T.block("T_reshape"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(A[T.int64(0), (v_ax2 // T.int64(4096) + v_ax0 * n + v_ax1) % n, v_ax2 % T.int64(4096) // T.int64(128), v_ax2 % T.int64(128)])
            T.writes(T_reshape[v_ax0, v_ax1, v_ax2])
            T_reshape[v_ax0, v_ax1, v_ax2] = A[T.int64(0), (v_ax2 // T.int64(4096) + v_ax0 * n + v_ax1) % n, v_ax2 % T.int64(4096) // T.int64(128), v_ax2 % T.int64(128)]


@T.prim_func
def fused_silu_multiply(p_lv4: T.handle, p_lv5: T.handle, p_output0: T.handle):
    T.func_attr({"tir.noalias": T.bool(True)})
    n = T.int64()
    lv4 = T.match_buffer(p_lv4, (T.int64(1), n, T.int64(11008)), "float16")
    lv5 = T.match_buffer(p_lv5, (T.int64(1), n, T.int64(11008)), "float16")
    var_T_multiply_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(11008)), "float16")
    # with T.block("root"):
    compute = T.alloc_buffer((T.int64(1), n, T.int64(11008)), "float16")
    var_T_multiply_intermediate_1 = T.alloc_buffer((T.int64(1), n, T.int64(11008)), "float16")
    for i0, i1, i2 in T.grid(T.int64(1), n, T.int64(11008)):
        with T.block("compute"):
            v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
            T.reads(lv4[v_i0, v_i1, v_i2])
            T.writes(compute[v_i0, v_i1, v_i2])
            compute[v_i0, v_i1, v_i2] = T.sigmoid(lv4[v_i0, v_i1, v_i2])
    for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(11008)):
        with T.block("T_multiply"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(lv4[v_ax0, v_ax1, v_ax2], compute[v_ax0, v_ax1, v_ax2])
            T.writes(var_T_multiply_intermediate_1[v_ax0, v_ax1, v_ax2])
            var_T_multiply_intermediate_1[v_ax0, v_ax1, v_ax2] = lv4[v_ax0, v_ax1, v_ax2] * compute[v_ax0, v_ax1, v_ax2]
    for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(11008)):
        with T.block("T_multiply_1"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_T_multiply_intermediate_1[v_ax0, v_ax1, v_ax2], lv5[v_ax0, v_ax1, v_ax2])
            T.writes(var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2] = var_T_multiply_intermediate_1[v_ax0, v_ax1, v_ax2] * lv5[v_ax0, v_ax1, v_ax2]


@T.prim_func
def slice(var_A: T.handle, slice_1: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16")):
    T.func_attr({"op_pattern": 8, "tir.noalias": T.bool(True)})
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
def cast(A: T.Buffer((T.int64(1), T.int64(1), T.int64(32000)), "float16"), compute: T.Buffer((T.int64(1), T.int64(1), T.int64(32000)), "float32")):
    T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    for i0, i1, i2 in T.grid(T.int64(1), T.int64(1), T.int64(32000)):
        with T.block("compute"):
            v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
            T.reads(A[v_i0, v_i1, v_i2])
            T.writes(compute[v_i0, v_i1, v_i2])
            compute[v_i0, v_i1, v_i2] = T.Cast("float32", A[v_i0, v_i1, v_i2])

DlightBench.register_bench_workload(reshape5, 'vicuna_v1_7b_fp16', 'reshape5')
DlightBench.register_bench_workload(take1, 'vicuna_v1_7b_fp16', 'take1')
DlightBench.register_bench_workload(reshape6, 'vicuna_v1_7b_fp16', 'reshape6')
DlightBench.register_bench_workload(full, 'vicuna_v1_7b_fp16', 'full')
DlightBench.register_bench_workload(rms_norm1, 'vicuna_v1_7b_fp16', 'rms_norm1')
DlightBench.register_bench_workload(reshape7, 'vicuna_v1_7b_fp16', 'reshape7')
DlightBench.register_bench_workload(fused_reshape7_squeeze1, 'vicuna_v1_7b_fp16', 'fused_reshape7_squeeze1')
DlightBench.register_bench_workload(rotary_embedding1, 'vicuna_v1_7b_fp16', 'rotary_embedding1')
DlightBench.register_bench_workload(squeeze1, 'vicuna_v1_7b_fp16', 'squeeze1')
DlightBench.register_bench_workload(reshape3, 'vicuna_v1_7b_fp16', 'reshape3')
DlightBench.register_bench_workload(transpose2, 'vicuna_v1_7b_fp16', 'transpose2')
DlightBench.register_bench_workload(transpose, 'vicuna_v1_7b_fp16', 'transpose')
DlightBench.register_bench_workload(fused_divide1_add1, 'vicuna_v1_7b_fp16', 'fused_divide1_add1')
DlightBench.register_bench_workload(softmax1, 'vicuna_v1_7b_fp16', 'softmax1')
DlightBench.register_bench_workload(matmul1, 'vicuna_v1_7b_fp16', 'matmul1')
DlightBench.register_bench_workload(fused_transpose3_reshape8, 'vicuna_v1_7b_fp16', 'fused_transpose3_reshape8')
DlightBench.register_bench_workload(fused_silu1_multiply1, 'vicuna_v1_7b_fp16', 'fused_silu1_multiply1')
DlightBench.register_bench_workload(slice1, 'vicuna_v1_7b_fp16', 'slice1')
DlightBench.register_bench_workload(cast, 'vicuna_v1_7b_fp16', 'cast')
DlightBench.register_bench_workload(reshape, 'vicuna_v1_7b_fp16', 'reshape')
DlightBench.register_bench_workload(take, 'vicuna_v1_7b_fp16', 'take')
DlightBench.register_bench_workload(reshape1, 'vicuna_v1_7b_fp16', 'reshape1')
DlightBench.register_bench_workload(fused_min_max_triu_te_broadcast_to, 'vicuna_v1_7b_fp16', 'fused_min_max_triu_te_broadcast_to')
DlightBench.register_bench_workload(extend_te, 'vicuna_v1_7b_fp16', 'extend_te')
DlightBench.register_bench_workload(rms_norm, 'vicuna_v1_7b_fp16', 'rms_norm')
DlightBench.register_bench_workload(reshape2, 'vicuna_v1_7b_fp16', 'reshape2')
DlightBench.register_bench_workload(rotary_embedding, 'vicuna_v1_7b_fp16', 'rotary_embedding')
DlightBench.register_bench_workload(squeeze, 'vicuna_v1_7b_fp16', 'squeeze')
DlightBench.register_bench_workload(reshape3, 'vicuna_v1_7b_fp16', 'reshape3')
DlightBench.register_bench_workload(transpose, 'vicuna_v1_7b_fp16', 'transpose')
DlightBench.register_bench_workload(fused_divide_add, 'vicuna_v1_7b_fp16', 'fused_divide_add')
DlightBench.register_bench_workload(softmax, 'vicuna_v1_7b_fp16', 'softmax')
DlightBench.register_bench_workload(matmul, 'vicuna_v1_7b_fp16', 'matmul')
DlightBench.register_bench_workload(transpose1, 'vicuna_v1_7b_fp16', 'transpose1')
DlightBench.register_bench_workload(reshape4, 'vicuna_v1_7b_fp16', 'reshape4')
DlightBench.register_bench_workload(fused_silu_multiply, 'vicuna_v1_7b_fp16', 'fused_silu_multiply')
DlightBench.register_bench_workload(slice, 'vicuna_v1_7b_fp16', 'slice')
DlightBench.register_bench_workload(cast, 'vicuna_v1_7b_fp16', 'cast')

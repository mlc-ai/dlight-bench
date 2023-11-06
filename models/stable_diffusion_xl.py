from dlight_bench import DlightBench
from tvm.script import tir as T

@T.prim_func
def add29(A: T.Buffer((T.int64(1), T.int64(4), T.int64(128), T.int64(128)), "float32"), B: T.Buffer((T.int64(1), T.int64(4), T.int64(128), T.int64(128)), "float32"), T_add: T.Buffer((T.int64(1), T.int64(4), T.int64(128), T.int64(128)), "float32")):
    T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(4), T.int64(128), T.int64(128)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3], B[v_ax0, v_ax1, v_ax2, v_ax3])
            T.writes(T_add[v_ax0, v_ax1, v_ax2, v_ax3])
            T_add[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax1, v_ax2, v_ax3] + B[v_ax0, v_ax1, v_ax2, v_ax3]

@T.prim_func
def add48(A: T.Buffer((), "float32"), T_add: T.Buffer((), "float32")):
    T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    with T.block("T_add"):
        vi = T.axis.spatial(1, T.int64(0))
        T.reads(A[()])
        T.writes(T_add[()])
        T_add[()] = A[()] + T.float32(1)

@T.prim_func
def argmax(A: T.Buffer((T.int64(1), T.int64(77)), "int32"), A_red: T.Buffer((T.int64(1),), "int64")):
    T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    A_red_temp_v0 = T.alloc_buffer((T.int64(1),), "int64")
    A_red_temp_v1 = T.alloc_buffer((T.int64(1),), "int32")
    for ax0, k1 in T.grid(T.int64(1), T.int64(77)):
        with T.block("A_red_temp"):
            v_ax0, v_k1 = T.axis.remap("SR", [ax0, k1])
            T.reads(A[v_ax0, v_k1])
            T.writes(A_red_temp_v0[v_ax0], A_red_temp_v1[v_ax0])
            with T.init():
                A_red_temp_v0[v_ax0] = T.int64(-1)
                A_red_temp_v1[v_ax0] = -2147483648
            v_A_red_temp_v0: T.int64 = T.Select(A_red_temp_v1[v_ax0] > A[v_ax0, v_k1] or A_red_temp_v1[v_ax0] == A[v_ax0, v_k1] and A_red_temp_v0[v_ax0] < v_k1, A_red_temp_v0[v_ax0], v_k1)
            v_A_red_temp_v1: T.int32 = T.Select(A_red_temp_v1[v_ax0] > A[v_ax0, v_k1], A_red_temp_v1[v_ax0], A[v_ax0, v_k1])
            A_red_temp_v0[v_ax0] = v_A_red_temp_v0
            A_red_temp_v1[v_ax0] = v_A_red_temp_v1
    for ax0 in range(T.int64(1)):
        with T.block("A_red"):
            v_ax0 = T.axis.spatial(T.int64(1), ax0)
            T.reads(A_red_temp_v0[v_ax0])
            T.writes(A_red[v_ax0])
            A_red[v_ax0] = A_red_temp_v0[v_ax0]

@T.prim_func
def cast(A: T.Buffer((T.int64(1), T.int64(77)), "int32"), compute: T.Buffer((T.int64(1), T.int64(77)), "int32")):
    T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    for i0, i1 in T.grid(T.int64(1), T.int64(77)):
        with T.block("compute"):
            v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
            T.reads(A[v_i0, v_i1])
            T.writes(compute[v_i0, v_i1])
            compute[v_i0, v_i1] = A[v_i0, v_i1]

@T.prim_func
def concatenate(A: T.Buffer((T.int64(1), T.int64(77), T.int64(768)), "float32"), B: T.Buffer((T.int64(1), T.int64(77), T.int64(1280)), "float32"), T_concat: T.Buffer((T.int64(1), T.int64(77), T.int64(2048)), "float32")):
    T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(77), T.int64(2048)):
        with T.block("T_concat"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(B[v_ax0, v_ax1, v_ax2 - T.int64(768)], A[v_ax0, v_ax1, v_ax2])
            T.writes(T_concat[v_ax0, v_ax1, v_ax2])
            T_concat[v_ax0, v_ax1, v_ax2] = T.if_then_else(T.int64(768) <= v_ax2, B[v_ax0, v_ax1, v_ax2 - T.int64(768)], A[v_ax0, v_ax1, v_ax2])

@T.prim_func
def concatenate10(A: T.Buffer((T.int64(2), T.int64(320), T.int64(128), T.int64(128)), "float32"), B: T.Buffer((T.int64(2), T.int64(320), T.int64(128), T.int64(128)), "float32"), T_concat: T.Buffer((T.int64(2), T.int64(640), T.int64(128), T.int64(128)), "float32")):
    T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(640), T.int64(128), T.int64(128)):
        with T.block("T_concat"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(B[v_ax0, v_ax1 - T.int64(320), v_ax2, v_ax3], A[v_ax0, v_ax1, v_ax2, v_ax3])
            T.writes(T_concat[v_ax0, v_ax1, v_ax2, v_ax3])
            T_concat[v_ax0, v_ax1, v_ax2, v_ax3] = T.if_then_else(T.int64(320) <= v_ax1, B[v_ax0, v_ax1 - T.int64(320), v_ax2, v_ax3], A[v_ax0, v_ax1, v_ax2, v_ax3])

@T.prim_func
def concatenate11(A: T.Buffer((T.int64(1), T.int64(4), T.int64(128), T.int64(128)), "float32"), B: T.Buffer((T.int64(1), T.int64(4), T.int64(128), T.int64(128)), "float32"), T_concat: T.Buffer((T.int64(2), T.int64(4), T.int64(128), T.int64(128)), "float32")):
    T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(4), T.int64(128), T.int64(128)):
        with T.block("T_concat"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(B[v_ax0 - T.int64(1), v_ax1, v_ax2, v_ax3], A[v_ax0, v_ax1, v_ax2, v_ax3])
            T.writes(T_concat[v_ax0, v_ax1, v_ax2, v_ax3])
            T_concat[v_ax0, v_ax1, v_ax2, v_ax3] = T.if_then_else(T.int64(1) <= v_ax0, B[v_ax0 - T.int64(1), v_ax1, v_ax2, v_ax3], A[v_ax0, v_ax1, v_ax2, v_ax3])

@T.prim_func
def concatenate12(A: T.Buffer((T.int64(1), T.int64(77), T.int64(2048)), "float32"), B: T.Buffer((T.int64(1), T.int64(77), T.int64(2048)), "float32"), T_concat: T.Buffer((T.int64(2), T.int64(77), T.int64(2048)), "float32")):
    T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    for ax0, ax1, ax2 in T.grid(T.int64(2), T.int64(77), T.int64(2048)):
        with T.block("T_concat"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(B[v_ax0 - T.int64(1), v_ax1, v_ax2], A[v_ax0, v_ax1, v_ax2])
            T.writes(T_concat[v_ax0, v_ax1, v_ax2])
            T_concat[v_ax0, v_ax1, v_ax2] = T.if_then_else(T.int64(1) <= v_ax0, B[v_ax0 - T.int64(1), v_ax1, v_ax2], A[v_ax0, v_ax1, v_ax2])

@T.prim_func
def concatenate13(A: T.Buffer((T.int64(1), T.int64(1280)), "float32"), B: T.Buffer((T.int64(1), T.int64(1280)), "float32"), T_concat: T.Buffer((T.int64(2), T.int64(1280)), "float32")):
    T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    for ax0, ax1 in T.grid(T.int64(2), T.int64(1280)):
        with T.block("T_concat"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(B[v_ax0 - T.int64(1), v_ax1], A[v_ax0, v_ax1])
            T.writes(T_concat[v_ax0, v_ax1])
            T_concat[v_ax0, v_ax1] = T.if_then_else(T.int64(1) <= v_ax0, B[v_ax0 - T.int64(1), v_ax1], A[v_ax0, v_ax1])

@T.prim_func
def concatenate4(A: T.Buffer((T.int64(2), T.int64(1280), T.int64(32), T.int64(32)), "float32"), B: T.Buffer((T.int64(2), T.int64(1280), T.int64(32), T.int64(32)), "float32"), T_concat: T.Buffer((T.int64(2), T.int64(2560), T.int64(32), T.int64(32)), "float32")):
    T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(2560), T.int64(32), T.int64(32)):
        with T.block("T_concat"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(B[v_ax0, v_ax1 - T.int64(1280), v_ax2, v_ax3], A[v_ax0, v_ax1, v_ax2, v_ax3])
            T.writes(T_concat[v_ax0, v_ax1, v_ax2, v_ax3])
            T_concat[v_ax0, v_ax1, v_ax2, v_ax3] = T.if_then_else(T.int64(1280) <= v_ax1, B[v_ax0, v_ax1 - T.int64(1280), v_ax2, v_ax3], A[v_ax0, v_ax1, v_ax2, v_ax3])

@T.prim_func
def concatenate6(A: T.Buffer((T.int64(2), T.int64(1280), T.int64(64), T.int64(64)), "float32"), B: T.Buffer((T.int64(2), T.int64(640), T.int64(64), T.int64(64)), "float32"), T_concat: T.Buffer((T.int64(2), T.int64(1920), T.int64(64), T.int64(64)), "float32")):
    T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(1920), T.int64(64), T.int64(64)):
        with T.block("T_concat"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(B[v_ax0, v_ax1 - T.int64(1280), v_ax2, v_ax3], A[v_ax0, v_ax1, v_ax2, v_ax3])
            T.writes(T_concat[v_ax0, v_ax1, v_ax2, v_ax3])
            T_concat[v_ax0, v_ax1, v_ax2, v_ax3] = T.if_then_else(T.int64(1280) <= v_ax1, B[v_ax0, v_ax1 - T.int64(1280), v_ax2, v_ax3], A[v_ax0, v_ax1, v_ax2, v_ax3])

@T.prim_func
def concatenate9(A: T.Buffer((T.int64(2), T.int64(640), T.int64(128), T.int64(128)), "float32"), B: T.Buffer((T.int64(2), T.int64(320), T.int64(128), T.int64(128)), "float32"), T_concat: T.Buffer((T.int64(2), T.int64(960), T.int64(128), T.int64(128)), "float32")):
    T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(960), T.int64(128), T.int64(128)):
        with T.block("T_concat"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(B[v_ax0, v_ax1 - T.int64(640), v_ax2, v_ax3], A[v_ax0, v_ax1, v_ax2, v_ax3])
            T.writes(T_concat[v_ax0, v_ax1, v_ax2, v_ax3])
            T_concat[v_ax0, v_ax1, v_ax2, v_ax3] = T.if_then_else(T.int64(640) <= v_ax1, B[v_ax0, v_ax1 - T.int64(640), v_ax2, v_ax3], A[v_ax0, v_ax1, v_ax2, v_ax3])

@T.prim_func
def divide11(A: T.Buffer((T.int64(2), T.int64(4), T.int64(128), T.int64(128)), "float32"), B: T.Buffer((), "float32"), T_divide: T.Buffer((T.int64(2), T.int64(4), T.int64(128), T.int64(128)), "float32")):
    T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(4), T.int64(128), T.int64(128)):
        with T.block("T_divide"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3], B[()])
            T.writes(T_divide[v_ax0, v_ax1, v_ax2, v_ax3])
            T_divide[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax1, v_ax2, v_ax3] / B[()]

@T.prim_func
def fused_broadcast_to1_strided_slice1_reshape12_cast3_multiply1_multiply2_tir_sin_tir_cos_concatenate1_strided_slice2_reshape13_strided_slice3_reshape13_concatenate1_cast4(inp_1: T.Buffer((), "int32"), param_0: T.Buffer((T.int64(1), T.int64(160)), "float32"), var_compute_intermediate: T.Buffer((T.int64(2), T.int64(320)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_T_broadcast_to_intermediate = T.alloc_buffer((T.int64(2),), "int32")
    var_T_strided_slice_with_axes_intermediate = T.alloc_buffer((T.int64(2),), "int32")
    var_T_reshape_intermediate = T.alloc_buffer((T.int64(2), T.int64(1)), "int32")
    var_compute_intermediate_1 = T.alloc_buffer((T.int64(2), T.int64(1)))
    var_T_multiply_intermediate = T.alloc_buffer((T.int64(2), T.int64(160)))
    var_T_multiply_intermediate_1 = T.alloc_buffer((T.int64(2), T.int64(160)))
    var_compute_intermediate_2 = T.alloc_buffer((T.int64(2), T.int64(160)))
    var_compute_intermediate_3 = T.alloc_buffer((T.int64(2), T.int64(160)))
    var_T_concat_intermediate = T.alloc_buffer((T.int64(2), T.int64(320)))
    var_T_strided_slice_with_axes_intermediate_1 = T.alloc_buffer((T.int64(2), T.int64(160)))
    var_T_reshape_intermediate_1 = T.alloc_buffer((T.int64(2), T.int64(160)))
    var_T_strided_slice_with_axes_intermediate_2 = T.alloc_buffer((T.int64(2), T.int64(160)))
    var_T_reshape_intermediate_2 = T.alloc_buffer((T.int64(2), T.int64(160)))
    var_T_concat_intermediate_1 = T.alloc_buffer((T.int64(2), T.int64(320)))
    for ax0 in range(T.int64(2)):
        with T.block("T_broadcast_to"):
            v_ax0 = T.axis.spatial(T.int64(2), ax0)
            T.reads(inp_1[()])
            T.writes(var_T_broadcast_to_intermediate[v_ax0])
            var_T_broadcast_to_intermediate[v_ax0] = inp_1[()]
    for ax0 in range(T.int64(2)):
        with T.block("T_strided_slice_with_axes"):
            v_ax0 = T.axis.spatial(T.int64(2), ax0)
            T.reads(var_T_broadcast_to_intermediate[v_ax0])
            T.writes(var_T_strided_slice_with_axes_intermediate[v_ax0])
            var_T_strided_slice_with_axes_intermediate[v_ax0] = var_T_broadcast_to_intermediate[v_ax0]
    for ax0, ax1 in T.grid(T.int64(2), T.int64(1)):
        with T.block("T_reshape"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(var_T_strided_slice_with_axes_intermediate[(v_ax0 + v_ax1) % T.int64(2)])
            T.writes(var_T_reshape_intermediate[v_ax0, v_ax1])
            var_T_reshape_intermediate[v_ax0, v_ax1] = var_T_strided_slice_with_axes_intermediate[(v_ax0 + v_ax1) % T.int64(2)]
    for i0, i1 in T.grid(T.int64(2), T.int64(1)):
        with T.block("compute"):
            v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
            T.reads(var_T_reshape_intermediate[v_i0, v_i1])
            T.writes(var_compute_intermediate_1[v_i0, v_i1])
            var_compute_intermediate_1[v_i0, v_i1] = T.Cast("float32", var_T_reshape_intermediate[v_i0, v_i1])
    for ax0, ax1 in T.grid(T.int64(2), T.int64(160)):
        with T.block("T_multiply"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(var_compute_intermediate_1[v_ax0, T.int64(0)], param_0[T.int64(0), v_ax1])
            T.writes(var_T_multiply_intermediate[v_ax0, v_ax1])
            var_T_multiply_intermediate[v_ax0, v_ax1] = var_compute_intermediate_1[v_ax0, T.int64(0)] * param_0[T.int64(0), v_ax1]
    for ax0, ax1 in T.grid(T.int64(2), T.int64(160)):
        with T.block("T_multiply_1"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(var_T_multiply_intermediate[v_ax0, v_ax1])
            T.writes(var_T_multiply_intermediate_1[v_ax0, v_ax1])
            var_T_multiply_intermediate_1[v_ax0, v_ax1] = var_T_multiply_intermediate[v_ax0, v_ax1]
    for i0, i1 in T.grid(T.int64(2), T.int64(160)):
        with T.block("compute_1"):
            v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
            T.reads(var_T_multiply_intermediate_1[v_i0, v_i1])
            T.writes(var_compute_intermediate_2[v_i0, v_i1])
            var_compute_intermediate_2[v_i0, v_i1] = T.sin(var_T_multiply_intermediate_1[v_i0, v_i1])
    for i0, i1 in T.grid(T.int64(2), T.int64(160)):
        with T.block("compute_2"):
            v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
            T.reads(var_T_multiply_intermediate_1[v_i0, v_i1])
            T.writes(var_compute_intermediate_3[v_i0, v_i1])
            var_compute_intermediate_3[v_i0, v_i1] = T.cos(var_T_multiply_intermediate_1[v_i0, v_i1])
    for ax0, ax1 in T.grid(T.int64(2), T.int64(320)):
        with T.block("T_concat"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(var_compute_intermediate_3[v_ax0, v_ax1 - T.int64(160)], var_compute_intermediate_2[v_ax0, v_ax1])
            T.writes(var_T_concat_intermediate[v_ax0, v_ax1])
            var_T_concat_intermediate[v_ax0, v_ax1] = T.if_then_else(T.int64(160) <= v_ax1, var_compute_intermediate_3[v_ax0, v_ax1 - T.int64(160)], var_compute_intermediate_2[v_ax0, v_ax1])
    for ax0, ax1 in T.grid(T.int64(2), T.int64(160)):
        with T.block("T_strided_slice_with_axes_1"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(var_T_concat_intermediate[v_ax0, v_ax1 + T.int64(160)])
            T.writes(var_T_strided_slice_with_axes_intermediate_1[v_ax0, v_ax1])
            var_T_strided_slice_with_axes_intermediate_1[v_ax0, v_ax1] = var_T_concat_intermediate[v_ax0, v_ax1 + T.int64(160)]
    for ax0, ax1 in T.grid(T.int64(2), T.int64(160)):
        with T.block("T_reshape_1"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(var_T_strided_slice_with_axes_intermediate_1[(v_ax1 // T.int64(160) + v_ax0) % T.int64(2), v_ax1 % T.int64(160)])
            T.writes(var_T_reshape_intermediate_1[v_ax0, v_ax1])
            var_T_reshape_intermediate_1[v_ax0, v_ax1] = var_T_strided_slice_with_axes_intermediate_1[(v_ax1 // T.int64(160) + v_ax0) % T.int64(2), v_ax1 % T.int64(160)]
    for ax0, ax1 in T.grid(T.int64(2), T.int64(160)):
        with T.block("T_strided_slice_with_axes_2"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(var_T_concat_intermediate[v_ax0, v_ax1])
            T.writes(var_T_strided_slice_with_axes_intermediate_2[v_ax0, v_ax1])
            var_T_strided_slice_with_axes_intermediate_2[v_ax0, v_ax1] = var_T_concat_intermediate[v_ax0, v_ax1]
    for ax0, ax1 in T.grid(T.int64(2), T.int64(160)):
        with T.block("T_reshape_2"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(var_T_strided_slice_with_axes_intermediate_2[(v_ax1 // T.int64(160) + v_ax0) % T.int64(2), v_ax1 % T.int64(160)])
            T.writes(var_T_reshape_intermediate_2[v_ax0, v_ax1])
            var_T_reshape_intermediate_2[v_ax0, v_ax1] = var_T_strided_slice_with_axes_intermediate_2[(v_ax1 // T.int64(160) + v_ax0) % T.int64(2), v_ax1 % T.int64(160)]
    for ax0, ax1 in T.grid(T.int64(2), T.int64(320)):
        with T.block("T_concat_1"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(var_T_reshape_intermediate_2[v_ax0, v_ax1 - T.int64(160)], var_T_reshape_intermediate_1[v_ax0, v_ax1])
            T.writes(var_T_concat_intermediate_1[v_ax0, v_ax1])
            var_T_concat_intermediate_1[v_ax0, v_ax1] = T.if_then_else(T.int64(160) <= v_ax1, var_T_reshape_intermediate_2[v_ax0, v_ax1 - T.int64(160)], var_T_reshape_intermediate_1[v_ax0, v_ax1])
    for i0, i1 in T.grid(T.int64(2), T.int64(320)):
        with T.block("compute_3"):
            v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
            T.reads(var_T_concat_intermediate_1[v_i0, v_i1])
            T.writes(var_compute_intermediate[v_i0, v_i1])
            var_compute_intermediate[v_i0, v_i1] = var_T_concat_intermediate_1[v_i0, v_i1]

@T.prim_func
def fused_cast_reshape1(lv: T.Buffer((T.int64(1), T.int64(77)), "int32"), var_T_reshape_intermediate: T.Buffer((T.int64(77),), "int32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_compute_intermediate = T.alloc_buffer((T.int64(1), T.int64(77)), "int32")
    for i0, i1 in T.grid(T.int64(1), T.int64(77)):
        with T.block("compute"):
            v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
            T.reads(lv[v_i0, v_i1])
            T.writes(var_compute_intermediate[v_i0, v_i1])
            var_compute_intermediate[v_i0, v_i1] = lv[v_i0, v_i1]
    for ax0 in range(T.int64(77)):
        with T.block("T_reshape"):
            v_ax0 = T.axis.spatial(T.int64(77), ax0)
            T.reads(var_compute_intermediate[T.int64(0), v_ax0 % T.int64(77)])
            T.writes(var_T_reshape_intermediate[v_ax0])
            var_T_reshape_intermediate[v_ax0] = var_compute_intermediate[T.int64(0), v_ax0 % T.int64(77)]

@T.prim_func
def fused_conv2d10_add20_add21(lv2553: T.Buffer((T.int64(2), T.int64(2560), T.int64(32), T.int64(32)), "float32"), unet_up_blocks_0_resnets_0_conv1_weight: T.Buffer((T.int64(1280), T.int64(2560), T.int64(3), T.int64(3)), "float32"), lv2555: T.Buffer((T.int64(1), T.int64(1280), T.int64(1), T.int64(1)), "float32"), lv2562: T.Buffer((T.int64(2), T.int64(1280), T.int64(1), T.int64(1)), "float32"), var_T_add_intermediate: T.Buffer((T.int64(2), T.int64(1280), T.int64(32), T.int64(32)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    pad_temp = T.alloc_buffer((T.int64(2), T.int64(2560), T.int64(34), T.int64(34)))
    var_conv2d_nchw_intermediate = T.alloc_buffer((T.int64(2), T.int64(1280), T.int64(32), T.int64(32)))
    var_T_add_intermediate_1 = T.alloc_buffer((T.int64(2), T.int64(1280), T.int64(32), T.int64(32)))
    for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(2560), T.int64(34), T.int64(34)):
        with T.block("pad_temp"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(lv2553[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)])
            T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
            pad_temp[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(T.int64(1) <= v_i2 and v_i2 < T.int64(33) and T.int64(1) <= v_i3 and v_i3 < T.int64(33), lv2553[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)], T.float32(0))
    for nn, ff, yy, xx, rc, ry, rx in T.grid(T.int64(2), T.int64(1280), T.int64(32), T.int64(32), T.int64(2560), T.int64(3), T.int64(3)):
        with T.block("conv2d_nchw"):
            v_nn, v_ff, v_yy, v_xx, v_rc, v_ry, v_rx = T.axis.remap("SSSSRRR", [nn, ff, yy, xx, rc, ry, rx])
            T.reads(pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx], unet_up_blocks_0_resnets_0_conv1_weight[v_ff, v_rc, v_ry, v_rx])
            T.writes(var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx])
            with T.init():
                var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = T.float32(0)
            var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] + pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx] * unet_up_blocks_0_resnets_0_conv1_weight[v_ff, v_rc, v_ry, v_rx]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(1280), T.int64(32), T.int64(32)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], lv2555[T.int64(0), v_ax1, T.int64(0), T.int64(0)])
            T.writes(var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3] = var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] + lv2555[T.int64(0), v_ax1, T.int64(0), T.int64(0)]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(1280), T.int64(32), T.int64(32)):
        with T.block("T_add_1"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3], lv2562[v_ax0, v_ax1, T.int64(0), T.int64(0)])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3] + lv2562[v_ax0, v_ax1, T.int64(0), T.int64(0)]

@T.prim_func
def fused_conv2d11_add20(lv2551: T.Buffer((T.int64(2), T.int64(2560), T.int64(32), T.int64(32)), "float32"), unet_up_blocks_0_resnets_0_conv_shortcut_weight: T.Buffer((T.int64(1280), T.int64(2560), T.int64(1), T.int64(1)), "float32"), lv2570: T.Buffer((T.int64(1), T.int64(1280), T.int64(1), T.int64(1)), "float32"), var_T_add_intermediate: T.Buffer((T.int64(2), T.int64(1280), T.int64(32), T.int64(32)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    pad_temp = T.alloc_buffer((T.int64(2), T.int64(2560), T.int64(32), T.int64(32)))
    var_conv2d_nchw_intermediate = T.alloc_buffer((T.int64(2), T.int64(1280), T.int64(32), T.int64(32)))
    for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(2560), T.int64(32), T.int64(32)):
        with T.block("pad_temp"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(lv2551[v_i0, v_i1, v_i2, v_i3])
            T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
            pad_temp[v_i0, v_i1, v_i2, v_i3] = lv2551[v_i0, v_i1, v_i2, v_i3]
    for nn, ff, yy, xx, rc, ry, rx in T.grid(T.int64(2), T.int64(1280), T.int64(32), T.int64(32), T.int64(2560), T.int64(1), T.int64(1)):
        with T.block("conv2d_nchw"):
            v_nn, v_ff, v_yy, v_xx, v_rc, v_ry, v_rx = T.axis.remap("SSSSRRR", [nn, ff, yy, xx, rc, ry, rx])
            T.reads(pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx], unet_up_blocks_0_resnets_0_conv_shortcut_weight[v_ff, v_rc, v_ry, v_rx])
            T.writes(var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx])
            with T.init():
                var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = T.float32(0)
            var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] + pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx] * unet_up_blocks_0_resnets_0_conv_shortcut_weight[v_ff, v_rc, v_ry, v_rx]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(1280), T.int64(32), T.int64(32)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], lv2570[T.int64(0), v_ax1, T.int64(0), T.int64(0)])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] + lv2570[T.int64(0), v_ax1, T.int64(0), T.int64(0)]

@T.prim_func
def fused_conv2d12_add20_add21(lv3963: T.Buffer((T.int64(2), T.int64(1920), T.int64(32), T.int64(32)), "float32"), unet_up_blocks_0_resnets_2_conv1_weight: T.Buffer((T.int64(1280), T.int64(1920), T.int64(3), T.int64(3)), "float32"), lv3965: T.Buffer((T.int64(1), T.int64(1280), T.int64(1), T.int64(1)), "float32"), lv3972: T.Buffer((T.int64(2), T.int64(1280), T.int64(1), T.int64(1)), "float32"), var_T_add_intermediate: T.Buffer((T.int64(2), T.int64(1280), T.int64(32), T.int64(32)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    pad_temp = T.alloc_buffer((T.int64(2), T.int64(1920), T.int64(34), T.int64(34)))
    var_conv2d_nchw_intermediate = T.alloc_buffer((T.int64(2), T.int64(1280), T.int64(32), T.int64(32)))
    var_T_add_intermediate_1 = T.alloc_buffer((T.int64(2), T.int64(1280), T.int64(32), T.int64(32)))
    for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(1920), T.int64(34), T.int64(34)):
        with T.block("pad_temp"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(lv3963[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)])
            T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
            pad_temp[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(T.int64(1) <= v_i2 and v_i2 < T.int64(33) and T.int64(1) <= v_i3 and v_i3 < T.int64(33), lv3963[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)], T.float32(0))
    for nn, ff, yy, xx, rc, ry, rx in T.grid(T.int64(2), T.int64(1280), T.int64(32), T.int64(32), T.int64(1920), T.int64(3), T.int64(3)):
        with T.block("conv2d_nchw"):
            v_nn, v_ff, v_yy, v_xx, v_rc, v_ry, v_rx = T.axis.remap("SSSSRRR", [nn, ff, yy, xx, rc, ry, rx])
            T.reads(pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx], unet_up_blocks_0_resnets_2_conv1_weight[v_ff, v_rc, v_ry, v_rx])
            T.writes(var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx])
            with T.init():
                var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = T.float32(0)
            var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] + pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx] * unet_up_blocks_0_resnets_2_conv1_weight[v_ff, v_rc, v_ry, v_rx]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(1280), T.int64(32), T.int64(32)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], lv3965[T.int64(0), v_ax1, T.int64(0), T.int64(0)])
            T.writes(var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3] = var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] + lv3965[T.int64(0), v_ax1, T.int64(0), T.int64(0)]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(1280), T.int64(32), T.int64(32)):
        with T.block("T_add_1"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3], lv3972[v_ax0, v_ax1, T.int64(0), T.int64(0)])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3] + lv3972[v_ax0, v_ax1, T.int64(0), T.int64(0)]

@T.prim_func
def fused_conv2d13_add20(lv3961: T.Buffer((T.int64(2), T.int64(1920), T.int64(32), T.int64(32)), "float32"), unet_up_blocks_0_resnets_2_conv_shortcut_weight: T.Buffer((T.int64(1280), T.int64(1920), T.int64(1), T.int64(1)), "float32"), lv3980: T.Buffer((T.int64(1), T.int64(1280), T.int64(1), T.int64(1)), "float32"), var_T_add_intermediate: T.Buffer((T.int64(2), T.int64(1280), T.int64(32), T.int64(32)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    pad_temp = T.alloc_buffer((T.int64(2), T.int64(1920), T.int64(32), T.int64(32)))
    var_conv2d_nchw_intermediate = T.alloc_buffer((T.int64(2), T.int64(1280), T.int64(32), T.int64(32)))
    for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(1920), T.int64(32), T.int64(32)):
        with T.block("pad_temp"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(lv3961[v_i0, v_i1, v_i2, v_i3])
            T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
            pad_temp[v_i0, v_i1, v_i2, v_i3] = lv3961[v_i0, v_i1, v_i2, v_i3]
    for nn, ff, yy, xx, rc, ry, rx in T.grid(T.int64(2), T.int64(1280), T.int64(32), T.int64(32), T.int64(1920), T.int64(1), T.int64(1)):
        with T.block("conv2d_nchw"):
            v_nn, v_ff, v_yy, v_xx, v_rc, v_ry, v_rx = T.axis.remap("SSSSRRR", [nn, ff, yy, xx, rc, ry, rx])
            T.reads(pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx], unet_up_blocks_0_resnets_2_conv_shortcut_weight[v_ff, v_rc, v_ry, v_rx])
            T.writes(var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx])
            with T.init():
                var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = T.float32(0)
            var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] + pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx] * unet_up_blocks_0_resnets_2_conv_shortcut_weight[v_ff, v_rc, v_ry, v_rx]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(1280), T.int64(32), T.int64(32)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], lv3980[T.int64(0), v_ax1, T.int64(0), T.int64(0)])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] + lv3980[T.int64(0), v_ax1, T.int64(0), T.int64(0)]

@T.prim_func
def fused_conv2d14_add26(lv4666: T.Buffer((T.int64(2), T.int64(1280), T.int64(64), T.int64(64)), "float32"), unet_up_blocks_0_upsamplers_0_conv_weight: T.Buffer((T.int64(1280), T.int64(1280), T.int64(3), T.int64(3)), "float32"), lv4668: T.Buffer((T.int64(1), T.int64(1280), T.int64(1), T.int64(1)), "float32"), var_T_add_intermediate: T.Buffer((T.int64(2), T.int64(1280), T.int64(64), T.int64(64)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    pad_temp = T.alloc_buffer((T.int64(2), T.int64(1280), T.int64(66), T.int64(66)))
    var_conv2d_nchw_intermediate = T.alloc_buffer((T.int64(2), T.int64(1280), T.int64(64), T.int64(64)))
    for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(1280), T.int64(66), T.int64(66)):
        with T.block("pad_temp"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(lv4666[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)])
            T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
            pad_temp[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(T.int64(1) <= v_i2 and v_i2 < T.int64(65) and T.int64(1) <= v_i3 and v_i3 < T.int64(65), lv4666[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)], T.float32(0))
    for nn, ff, yy, xx, rc, ry, rx in T.grid(T.int64(2), T.int64(1280), T.int64(64), T.int64(64), T.int64(1280), T.int64(3), T.int64(3)):
        with T.block("conv2d_nchw"):
            v_nn, v_ff, v_yy, v_xx, v_rc, v_ry, v_rx = T.axis.remap("SSSSRRR", [nn, ff, yy, xx, rc, ry, rx])
            T.reads(pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx], unet_up_blocks_0_upsamplers_0_conv_weight[v_ff, v_rc, v_ry, v_rx])
            T.writes(var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx])
            with T.init():
                var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = T.float32(0)
            var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] + pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx] * unet_up_blocks_0_upsamplers_0_conv_weight[v_ff, v_rc, v_ry, v_rx]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(1280), T.int64(64), T.int64(64)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], lv4668[T.int64(0), v_ax1, T.int64(0), T.int64(0)])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] + lv4668[T.int64(0), v_ax1, T.int64(0), T.int64(0)]

@T.prim_func
def fused_conv2d15_add12_add14(lv4672: T.Buffer((T.int64(2), T.int64(1920), T.int64(64), T.int64(64)), "float32"), unet_up_blocks_1_resnets_0_conv1_weight: T.Buffer((T.int64(640), T.int64(1920), T.int64(3), T.int64(3)), "float32"), lv4674: T.Buffer((T.int64(1), T.int64(640), T.int64(1), T.int64(1)), "float32"), lv4681: T.Buffer((T.int64(2), T.int64(640), T.int64(1), T.int64(1)), "float32"), var_T_add_intermediate: T.Buffer((T.int64(2), T.int64(640), T.int64(64), T.int64(64)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    pad_temp = T.alloc_buffer((T.int64(2), T.int64(1920), T.int64(66), T.int64(66)))
    var_conv2d_nchw_intermediate = T.alloc_buffer((T.int64(2), T.int64(640), T.int64(64), T.int64(64)))
    var_T_add_intermediate_1 = T.alloc_buffer((T.int64(2), T.int64(640), T.int64(64), T.int64(64)))
    for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(1920), T.int64(66), T.int64(66)):
        with T.block("pad_temp"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(lv4672[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)])
            T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
            pad_temp[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(T.int64(1) <= v_i2 and v_i2 < T.int64(65) and T.int64(1) <= v_i3 and v_i3 < T.int64(65), lv4672[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)], T.float32(0))
    for nn, ff, yy, xx, rc, ry, rx in T.grid(T.int64(2), T.int64(640), T.int64(64), T.int64(64), T.int64(1920), T.int64(3), T.int64(3)):
        with T.block("conv2d_nchw"):
            v_nn, v_ff, v_yy, v_xx, v_rc, v_ry, v_rx = T.axis.remap("SSSSRRR", [nn, ff, yy, xx, rc, ry, rx])
            T.reads(pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx], unet_up_blocks_1_resnets_0_conv1_weight[v_ff, v_rc, v_ry, v_rx])
            T.writes(var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx])
            with T.init():
                var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = T.float32(0)
            var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] + pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx] * unet_up_blocks_1_resnets_0_conv1_weight[v_ff, v_rc, v_ry, v_rx]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(640), T.int64(64), T.int64(64)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], lv4674[T.int64(0), v_ax1, T.int64(0), T.int64(0)])
            T.writes(var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3] = var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] + lv4674[T.int64(0), v_ax1, T.int64(0), T.int64(0)]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(640), T.int64(64), T.int64(64)):
        with T.block("T_add_1"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3], lv4681[v_ax0, v_ax1, T.int64(0), T.int64(0)])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3] + lv4681[v_ax0, v_ax1, T.int64(0), T.int64(0)]

@T.prim_func
def fused_conv2d16_add12(lv4670: T.Buffer((T.int64(2), T.int64(1920), T.int64(64), T.int64(64)), "float32"), unet_up_blocks_1_resnets_0_conv_shortcut_weight: T.Buffer((T.int64(640), T.int64(1920), T.int64(1), T.int64(1)), "float32"), lv4689: T.Buffer((T.int64(1), T.int64(640), T.int64(1), T.int64(1)), "float32"), var_T_add_intermediate: T.Buffer((T.int64(2), T.int64(640), T.int64(64), T.int64(64)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    pad_temp = T.alloc_buffer((T.int64(2), T.int64(1920), T.int64(64), T.int64(64)))
    var_conv2d_nchw_intermediate = T.alloc_buffer((T.int64(2), T.int64(640), T.int64(64), T.int64(64)))
    for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(1920), T.int64(64), T.int64(64)):
        with T.block("pad_temp"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(lv4670[v_i0, v_i1, v_i2, v_i3])
            T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
            pad_temp[v_i0, v_i1, v_i2, v_i3] = lv4670[v_i0, v_i1, v_i2, v_i3]
    for nn, ff, yy, xx, rc, ry, rx in T.grid(T.int64(2), T.int64(640), T.int64(64), T.int64(64), T.int64(1920), T.int64(1), T.int64(1)):
        with T.block("conv2d_nchw"):
            v_nn, v_ff, v_yy, v_xx, v_rc, v_ry, v_rx = T.axis.remap("SSSSRRR", [nn, ff, yy, xx, rc, ry, rx])
            T.reads(pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx], unet_up_blocks_1_resnets_0_conv_shortcut_weight[v_ff, v_rc, v_ry, v_rx])
            T.writes(var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx])
            with T.init():
                var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = T.float32(0)
            var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] + pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx] * unet_up_blocks_1_resnets_0_conv_shortcut_weight[v_ff, v_rc, v_ry, v_rx]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(640), T.int64(64), T.int64(64)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], lv4689[T.int64(0), v_ax1, T.int64(0), T.int64(0)])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] + lv4689[T.int64(0), v_ax1, T.int64(0), T.int64(0)]

@T.prim_func
def fused_conv2d17_add12_add14(lv4841: T.Buffer((T.int64(2), T.int64(1280), T.int64(64), T.int64(64)), "float32"), unet_up_blocks_1_resnets_1_conv1_weight: T.Buffer((T.int64(640), T.int64(1280), T.int64(3), T.int64(3)), "float32"), lv4843: T.Buffer((T.int64(1), T.int64(640), T.int64(1), T.int64(1)), "float32"), lv4850: T.Buffer((T.int64(2), T.int64(640), T.int64(1), T.int64(1)), "float32"), var_T_add_intermediate: T.Buffer((T.int64(2), T.int64(640), T.int64(64), T.int64(64)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    pad_temp = T.alloc_buffer((T.int64(2), T.int64(1280), T.int64(66), T.int64(66)))
    var_conv2d_nchw_intermediate = T.alloc_buffer((T.int64(2), T.int64(640), T.int64(64), T.int64(64)))
    var_T_add_intermediate_1 = T.alloc_buffer((T.int64(2), T.int64(640), T.int64(64), T.int64(64)))
    for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(1280), T.int64(66), T.int64(66)):
        with T.block("pad_temp"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(lv4841[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)])
            T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
            pad_temp[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(T.int64(1) <= v_i2 and v_i2 < T.int64(65) and T.int64(1) <= v_i3 and v_i3 < T.int64(65), lv4841[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)], T.float32(0))
    for nn, ff, yy, xx, rc, ry, rx in T.grid(T.int64(2), T.int64(640), T.int64(64), T.int64(64), T.int64(1280), T.int64(3), T.int64(3)):
        with T.block("conv2d_nchw"):
            v_nn, v_ff, v_yy, v_xx, v_rc, v_ry, v_rx = T.axis.remap("SSSSRRR", [nn, ff, yy, xx, rc, ry, rx])
            T.reads(pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx], unet_up_blocks_1_resnets_1_conv1_weight[v_ff, v_rc, v_ry, v_rx])
            T.writes(var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx])
            with T.init():
                var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = T.float32(0)
            var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] + pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx] * unet_up_blocks_1_resnets_1_conv1_weight[v_ff, v_rc, v_ry, v_rx]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(640), T.int64(64), T.int64(64)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], lv4843[T.int64(0), v_ax1, T.int64(0), T.int64(0)])
            T.writes(var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3] = var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] + lv4843[T.int64(0), v_ax1, T.int64(0), T.int64(0)]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(640), T.int64(64), T.int64(64)):
        with T.block("T_add_1"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3], lv4850[v_ax0, v_ax1, T.int64(0), T.int64(0)])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3] + lv4850[v_ax0, v_ax1, T.int64(0), T.int64(0)]

@T.prim_func
def fused_conv2d18_add12(lv4839: T.Buffer((T.int64(2), T.int64(1280), T.int64(64), T.int64(64)), "float32"), unet_up_blocks_1_resnets_1_conv_shortcut_weight: T.Buffer((T.int64(640), T.int64(1280), T.int64(1), T.int64(1)), "float32"), lv4858: T.Buffer((T.int64(1), T.int64(640), T.int64(1), T.int64(1)), "float32"), var_T_add_intermediate: T.Buffer((T.int64(2), T.int64(640), T.int64(64), T.int64(64)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    pad_temp = T.alloc_buffer((T.int64(2), T.int64(1280), T.int64(64), T.int64(64)))
    var_conv2d_nchw_intermediate = T.alloc_buffer((T.int64(2), T.int64(640), T.int64(64), T.int64(64)))
    for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(1280), T.int64(64), T.int64(64)):
        with T.block("pad_temp"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(lv4839[v_i0, v_i1, v_i2, v_i3])
            T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
            pad_temp[v_i0, v_i1, v_i2, v_i3] = lv4839[v_i0, v_i1, v_i2, v_i3]
    for nn, ff, yy, xx, rc, ry, rx in T.grid(T.int64(2), T.int64(640), T.int64(64), T.int64(64), T.int64(1280), T.int64(1), T.int64(1)):
        with T.block("conv2d_nchw"):
            v_nn, v_ff, v_yy, v_xx, v_rc, v_ry, v_rx = T.axis.remap("SSSSRRR", [nn, ff, yy, xx, rc, ry, rx])
            T.reads(pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx], unet_up_blocks_1_resnets_1_conv_shortcut_weight[v_ff, v_rc, v_ry, v_rx])
            T.writes(var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx])
            with T.init():
                var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = T.float32(0)
            var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] + pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx] * unet_up_blocks_1_resnets_1_conv_shortcut_weight[v_ff, v_rc, v_ry, v_rx]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(640), T.int64(64), T.int64(64)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], lv4858[T.int64(0), v_ax1, T.int64(0), T.int64(0)])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] + lv4858[T.int64(0), v_ax1, T.int64(0), T.int64(0)]

@T.prim_func
def fused_conv2d19_add12_add14(lv5010: T.Buffer((T.int64(2), T.int64(960), T.int64(64), T.int64(64)), "float32"), unet_up_blocks_1_resnets_2_conv1_weight: T.Buffer((T.int64(640), T.int64(960), T.int64(3), T.int64(3)), "float32"), lv5012: T.Buffer((T.int64(1), T.int64(640), T.int64(1), T.int64(1)), "float32"), lv5019: T.Buffer((T.int64(2), T.int64(640), T.int64(1), T.int64(1)), "float32"), var_T_add_intermediate: T.Buffer((T.int64(2), T.int64(640), T.int64(64), T.int64(64)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    pad_temp = T.alloc_buffer((T.int64(2), T.int64(960), T.int64(66), T.int64(66)))
    var_conv2d_nchw_intermediate = T.alloc_buffer((T.int64(2), T.int64(640), T.int64(64), T.int64(64)))
    var_T_add_intermediate_1 = T.alloc_buffer((T.int64(2), T.int64(640), T.int64(64), T.int64(64)))
    for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(960), T.int64(66), T.int64(66)):
        with T.block("pad_temp"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(lv5010[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)])
            T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
            pad_temp[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(T.int64(1) <= v_i2 and v_i2 < T.int64(65) and T.int64(1) <= v_i3 and v_i3 < T.int64(65), lv5010[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)], T.float32(0))
    for nn, ff, yy, xx, rc, ry, rx in T.grid(T.int64(2), T.int64(640), T.int64(64), T.int64(64), T.int64(960), T.int64(3), T.int64(3)):
        with T.block("conv2d_nchw"):
            v_nn, v_ff, v_yy, v_xx, v_rc, v_ry, v_rx = T.axis.remap("SSSSRRR", [nn, ff, yy, xx, rc, ry, rx])
            T.reads(pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx], unet_up_blocks_1_resnets_2_conv1_weight[v_ff, v_rc, v_ry, v_rx])
            T.writes(var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx])
            with T.init():
                var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = T.float32(0)
            var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] + pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx] * unet_up_blocks_1_resnets_2_conv1_weight[v_ff, v_rc, v_ry, v_rx]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(640), T.int64(64), T.int64(64)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], lv5012[T.int64(0), v_ax1, T.int64(0), T.int64(0)])
            T.writes(var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3] = var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] + lv5012[T.int64(0), v_ax1, T.int64(0), T.int64(0)]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(640), T.int64(64), T.int64(64)):
        with T.block("T_add_1"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3], lv5019[v_ax0, v_ax1, T.int64(0), T.int64(0)])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3] + lv5019[v_ax0, v_ax1, T.int64(0), T.int64(0)]

@T.prim_func
def fused_conv2d1_add7_add10_divide(lv62: T.Buffer((T.int64(2), T.int64(320), T.int64(128), T.int64(128)), "float32"), unet_down_blocks_0_resnets_0_conv2_weight: T.Buffer((T.int64(320), T.int64(320), T.int64(3), T.int64(3)), "float32"), lv64: T.Buffer((T.int64(1), T.int64(320), T.int64(1), T.int64(1)), "float32"), lv48: T.Buffer((T.int64(2), T.int64(320), T.int64(128), T.int64(128)), "float32"), var_T_divide_intermediate: T.Buffer((T.int64(2), T.int64(320), T.int64(128), T.int64(128)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    pad_temp = T.alloc_buffer((T.int64(2), T.int64(320), T.int64(130), T.int64(130)))
    var_conv2d_nchw_intermediate = T.alloc_buffer((T.int64(2), T.int64(320), T.int64(128), T.int64(128)))
    var_T_add_intermediate = T.alloc_buffer((T.int64(2), T.int64(320), T.int64(128), T.int64(128)))
    var_T_add_intermediate_1 = T.alloc_buffer((T.int64(2), T.int64(320), T.int64(128), T.int64(128)))
    for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(320), T.int64(130), T.int64(130)):
        with T.block("pad_temp"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(lv62[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)])
            T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
            pad_temp[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(T.int64(1) <= v_i2 and v_i2 < T.int64(129) and T.int64(1) <= v_i3 and v_i3 < T.int64(129), lv62[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)], T.float32(0))
    for nn, ff, yy, xx, rc, ry, rx in T.grid(T.int64(2), T.int64(320), T.int64(128), T.int64(128), T.int64(320), T.int64(3), T.int64(3)):
        with T.block("conv2d_nchw"):
            v_nn, v_ff, v_yy, v_xx, v_rc, v_ry, v_rx = T.axis.remap("SSSSRRR", [nn, ff, yy, xx, rc, ry, rx])
            T.reads(pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx], unet_down_blocks_0_resnets_0_conv2_weight[v_ff, v_rc, v_ry, v_rx])
            T.writes(var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx])
            with T.init():
                var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = T.float32(0)
            var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] + pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx] * unet_down_blocks_0_resnets_0_conv2_weight[v_ff, v_rc, v_ry, v_rx]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(320), T.int64(128), T.int64(128)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], lv64[T.int64(0), v_ax1, T.int64(0), T.int64(0)])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] + lv64[T.int64(0), v_ax1, T.int64(0), T.int64(0)]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(320), T.int64(128), T.int64(128)):
        with T.block("T_add_1"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(lv48[v_ax0, v_ax1, v_ax2, v_ax3], var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            T.writes(var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3] = lv48[v_ax0, v_ax1, v_ax2, v_ax3] + var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(320), T.int64(128), T.int64(128)):
        with T.block("T_divide"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3])
            T.writes(var_T_divide_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_divide_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3]

@T.prim_func
def fused_conv2d1_add7_add9(lv50: T.Buffer((T.int64(2), T.int64(320), T.int64(128), T.int64(128)), "float32"), unet_down_blocks_0_resnets_0_conv1_weight: T.Buffer((T.int64(320), T.int64(320), T.int64(3), T.int64(3)), "float32"), lv52: T.Buffer((T.int64(1), T.int64(320), T.int64(1), T.int64(1)), "float32"), lv59: T.Buffer((T.int64(2), T.int64(320), T.int64(1), T.int64(1)), "float32"), var_T_add_intermediate: T.Buffer((T.int64(2), T.int64(320), T.int64(128), T.int64(128)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    pad_temp = T.alloc_buffer((T.int64(2), T.int64(320), T.int64(130), T.int64(130)))
    var_conv2d_nchw_intermediate = T.alloc_buffer((T.int64(2), T.int64(320), T.int64(128), T.int64(128)))
    var_T_add_intermediate_1 = T.alloc_buffer((T.int64(2), T.int64(320), T.int64(128), T.int64(128)))
    for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(320), T.int64(130), T.int64(130)):
        with T.block("pad_temp"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(lv50[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)])
            T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
            pad_temp[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(T.int64(1) <= v_i2 and v_i2 < T.int64(129) and T.int64(1) <= v_i3 and v_i3 < T.int64(129), lv50[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)], T.float32(0))
    for nn, ff, yy, xx, rc, ry, rx in T.grid(T.int64(2), T.int64(320), T.int64(128), T.int64(128), T.int64(320), T.int64(3), T.int64(3)):
        with T.block("conv2d_nchw"):
            v_nn, v_ff, v_yy, v_xx, v_rc, v_ry, v_rx = T.axis.remap("SSSSRRR", [nn, ff, yy, xx, rc, ry, rx])
            T.reads(pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx], unet_down_blocks_0_resnets_0_conv1_weight[v_ff, v_rc, v_ry, v_rx])
            T.writes(var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx])
            with T.init():
                var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = T.float32(0)
            var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] + pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx] * unet_down_blocks_0_resnets_0_conv1_weight[v_ff, v_rc, v_ry, v_rx]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(320), T.int64(128), T.int64(128)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], lv52[T.int64(0), v_ax1, T.int64(0), T.int64(0)])
            T.writes(var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3] = var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] + lv52[T.int64(0), v_ax1, T.int64(0), T.int64(0)]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(320), T.int64(128), T.int64(128)):
        with T.block("T_add_1"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3], lv59[v_ax0, v_ax1, T.int64(0), T.int64(0)])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3] + lv59[v_ax0, v_ax1, T.int64(0), T.int64(0)]

@T.prim_func
def fused_conv2d20_add12(lv5008: T.Buffer((T.int64(2), T.int64(960), T.int64(64), T.int64(64)), "float32"), unet_up_blocks_1_resnets_2_conv_shortcut_weight: T.Buffer((T.int64(640), T.int64(960), T.int64(1), T.int64(1)), "float32"), lv5027: T.Buffer((T.int64(1), T.int64(640), T.int64(1), T.int64(1)), "float32"), var_T_add_intermediate: T.Buffer((T.int64(2), T.int64(640), T.int64(64), T.int64(64)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    pad_temp = T.alloc_buffer((T.int64(2), T.int64(960), T.int64(64), T.int64(64)))
    var_conv2d_nchw_intermediate = T.alloc_buffer((T.int64(2), T.int64(640), T.int64(64), T.int64(64)))
    for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(960), T.int64(64), T.int64(64)):
        with T.block("pad_temp"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(lv5008[v_i0, v_i1, v_i2, v_i3])
            T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
            pad_temp[v_i0, v_i1, v_i2, v_i3] = lv5008[v_i0, v_i1, v_i2, v_i3]
    for nn, ff, yy, xx, rc, ry, rx in T.grid(T.int64(2), T.int64(640), T.int64(64), T.int64(64), T.int64(960), T.int64(1), T.int64(1)):
        with T.block("conv2d_nchw"):
            v_nn, v_ff, v_yy, v_xx, v_rc, v_ry, v_rx = T.axis.remap("SSSSRRR", [nn, ff, yy, xx, rc, ry, rx])
            T.reads(pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx], unet_up_blocks_1_resnets_2_conv_shortcut_weight[v_ff, v_rc, v_ry, v_rx])
            T.writes(var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx])
            with T.init():
                var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = T.float32(0)
            var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] + pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx] * unet_up_blocks_1_resnets_2_conv_shortcut_weight[v_ff, v_rc, v_ry, v_rx]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(640), T.int64(64), T.int64(64)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], lv5027[T.int64(0), v_ax1, T.int64(0), T.int64(0)])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] + lv5027[T.int64(0), v_ax1, T.int64(0), T.int64(0)]

@T.prim_func
def fused_conv2d21_add27(lv5177: T.Buffer((T.int64(2), T.int64(640), T.int64(128), T.int64(128)), "float32"), unet_up_blocks_1_upsamplers_0_conv_weight: T.Buffer((T.int64(640), T.int64(640), T.int64(3), T.int64(3)), "float32"), lv5179: T.Buffer((T.int64(1), T.int64(640), T.int64(1), T.int64(1)), "float32"), var_T_add_intermediate: T.Buffer((T.int64(2), T.int64(640), T.int64(128), T.int64(128)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    pad_temp = T.alloc_buffer((T.int64(2), T.int64(640), T.int64(130), T.int64(130)))
    var_conv2d_nchw_intermediate = T.alloc_buffer((T.int64(2), T.int64(640), T.int64(128), T.int64(128)))
    for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(640), T.int64(130), T.int64(130)):
        with T.block("pad_temp"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(lv5177[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)])
            T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
            pad_temp[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(T.int64(1) <= v_i2 and v_i2 < T.int64(129) and T.int64(1) <= v_i3 and v_i3 < T.int64(129), lv5177[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)], T.float32(0))
    for nn, ff, yy, xx, rc, ry, rx in T.grid(T.int64(2), T.int64(640), T.int64(128), T.int64(128), T.int64(640), T.int64(3), T.int64(3)):
        with T.block("conv2d_nchw"):
            v_nn, v_ff, v_yy, v_xx, v_rc, v_ry, v_rx = T.axis.remap("SSSSRRR", [nn, ff, yy, xx, rc, ry, rx])
            T.reads(pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx], unet_up_blocks_1_upsamplers_0_conv_weight[v_ff, v_rc, v_ry, v_rx])
            T.writes(var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx])
            with T.init():
                var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = T.float32(0)
            var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] + pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx] * unet_up_blocks_1_upsamplers_0_conv_weight[v_ff, v_rc, v_ry, v_rx]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(640), T.int64(128), T.int64(128)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], lv5179[T.int64(0), v_ax1, T.int64(0), T.int64(0)])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] + lv5179[T.int64(0), v_ax1, T.int64(0), T.int64(0)]

@T.prim_func
def fused_conv2d22_add7_add9(lv5183: T.Buffer((T.int64(2), T.int64(960), T.int64(128), T.int64(128)), "float32"), unet_up_blocks_2_resnets_0_conv1_weight: T.Buffer((T.int64(320), T.int64(960), T.int64(3), T.int64(3)), "float32"), lv5185: T.Buffer((T.int64(1), T.int64(320), T.int64(1), T.int64(1)), "float32"), lv5192: T.Buffer((T.int64(2), T.int64(320), T.int64(1), T.int64(1)), "float32"), var_T_add_intermediate: T.Buffer((T.int64(2), T.int64(320), T.int64(128), T.int64(128)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    pad_temp = T.alloc_buffer((T.int64(2), T.int64(960), T.int64(130), T.int64(130)))
    var_conv2d_nchw_intermediate = T.alloc_buffer((T.int64(2), T.int64(320), T.int64(128), T.int64(128)))
    var_T_add_intermediate_1 = T.alloc_buffer((T.int64(2), T.int64(320), T.int64(128), T.int64(128)))
    for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(960), T.int64(130), T.int64(130)):
        with T.block("pad_temp"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(lv5183[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)])
            T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
            pad_temp[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(T.int64(1) <= v_i2 and v_i2 < T.int64(129) and T.int64(1) <= v_i3 and v_i3 < T.int64(129), lv5183[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)], T.float32(0))
    for nn, ff, yy, xx, rc, ry, rx in T.grid(T.int64(2), T.int64(320), T.int64(128), T.int64(128), T.int64(960), T.int64(3), T.int64(3)):
        with T.block("conv2d_nchw"):
            v_nn, v_ff, v_yy, v_xx, v_rc, v_ry, v_rx = T.axis.remap("SSSSRRR", [nn, ff, yy, xx, rc, ry, rx])
            T.reads(pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx], unet_up_blocks_2_resnets_0_conv1_weight[v_ff, v_rc, v_ry, v_rx])
            T.writes(var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx])
            with T.init():
                var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = T.float32(0)
            var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] + pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx] * unet_up_blocks_2_resnets_0_conv1_weight[v_ff, v_rc, v_ry, v_rx]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(320), T.int64(128), T.int64(128)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], lv5185[T.int64(0), v_ax1, T.int64(0), T.int64(0)])
            T.writes(var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3] = var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] + lv5185[T.int64(0), v_ax1, T.int64(0), T.int64(0)]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(320), T.int64(128), T.int64(128)):
        with T.block("T_add_1"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3], lv5192[v_ax0, v_ax1, T.int64(0), T.int64(0)])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3] + lv5192[v_ax0, v_ax1, T.int64(0), T.int64(0)]

@T.prim_func
def fused_conv2d23_add7(lv5181: T.Buffer((T.int64(2), T.int64(960), T.int64(128), T.int64(128)), "float32"), unet_up_blocks_2_resnets_0_conv_shortcut_weight: T.Buffer((T.int64(320), T.int64(960), T.int64(1), T.int64(1)), "float32"), lv5200: T.Buffer((T.int64(1), T.int64(320), T.int64(1), T.int64(1)), "float32"), var_T_add_intermediate: T.Buffer((T.int64(2), T.int64(320), T.int64(128), T.int64(128)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    pad_temp = T.alloc_buffer((T.int64(2), T.int64(960), T.int64(128), T.int64(128)))
    var_conv2d_nchw_intermediate = T.alloc_buffer((T.int64(2), T.int64(320), T.int64(128), T.int64(128)))
    for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(960), T.int64(128), T.int64(128)):
        with T.block("pad_temp"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(lv5181[v_i0, v_i1, v_i2, v_i3])
            T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
            pad_temp[v_i0, v_i1, v_i2, v_i3] = lv5181[v_i0, v_i1, v_i2, v_i3]
    for nn, ff, yy, xx, rc, ry, rx in T.grid(T.int64(2), T.int64(320), T.int64(128), T.int64(128), T.int64(960), T.int64(1), T.int64(1)):
        with T.block("conv2d_nchw"):
            v_nn, v_ff, v_yy, v_xx, v_rc, v_ry, v_rx = T.axis.remap("SSSSRRR", [nn, ff, yy, xx, rc, ry, rx])
            T.reads(pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx], unet_up_blocks_2_resnets_0_conv_shortcut_weight[v_ff, v_rc, v_ry, v_rx])
            T.writes(var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx])
            with T.init():
                var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = T.float32(0)
            var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] + pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx] * unet_up_blocks_2_resnets_0_conv_shortcut_weight[v_ff, v_rc, v_ry, v_rx]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(320), T.int64(128), T.int64(128)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], lv5200[T.int64(0), v_ax1, T.int64(0), T.int64(0)])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] + lv5200[T.int64(0), v_ax1, T.int64(0), T.int64(0)]

@T.prim_func
def fused_conv2d24_add7_add9(lv5206: T.Buffer((T.int64(2), T.int64(640), T.int64(128), T.int64(128)), "float32"), unet_up_blocks_2_resnets_1_conv1_weight: T.Buffer((T.int64(320), T.int64(640), T.int64(3), T.int64(3)), "float32"), lv5208: T.Buffer((T.int64(1), T.int64(320), T.int64(1), T.int64(1)), "float32"), lv5215: T.Buffer((T.int64(2), T.int64(320), T.int64(1), T.int64(1)), "float32"), var_T_add_intermediate: T.Buffer((T.int64(2), T.int64(320), T.int64(128), T.int64(128)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    pad_temp = T.alloc_buffer((T.int64(2), T.int64(640), T.int64(130), T.int64(130)))
    var_conv2d_nchw_intermediate = T.alloc_buffer((T.int64(2), T.int64(320), T.int64(128), T.int64(128)))
    var_T_add_intermediate_1 = T.alloc_buffer((T.int64(2), T.int64(320), T.int64(128), T.int64(128)))
    for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(640), T.int64(130), T.int64(130)):
        with T.block("pad_temp"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(lv5206[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)])
            T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
            pad_temp[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(T.int64(1) <= v_i2 and v_i2 < T.int64(129) and T.int64(1) <= v_i3 and v_i3 < T.int64(129), lv5206[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)], T.float32(0))
    for nn, ff, yy, xx, rc, ry, rx in T.grid(T.int64(2), T.int64(320), T.int64(128), T.int64(128), T.int64(640), T.int64(3), T.int64(3)):
        with T.block("conv2d_nchw"):
            v_nn, v_ff, v_yy, v_xx, v_rc, v_ry, v_rx = T.axis.remap("SSSSRRR", [nn, ff, yy, xx, rc, ry, rx])
            T.reads(pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx], unet_up_blocks_2_resnets_1_conv1_weight[v_ff, v_rc, v_ry, v_rx])
            T.writes(var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx])
            with T.init():
                var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = T.float32(0)
            var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] + pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx] * unet_up_blocks_2_resnets_1_conv1_weight[v_ff, v_rc, v_ry, v_rx]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(320), T.int64(128), T.int64(128)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], lv5208[T.int64(0), v_ax1, T.int64(0), T.int64(0)])
            T.writes(var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3] = var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] + lv5208[T.int64(0), v_ax1, T.int64(0), T.int64(0)]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(320), T.int64(128), T.int64(128)):
        with T.block("T_add_1"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3], lv5215[v_ax0, v_ax1, T.int64(0), T.int64(0)])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3] + lv5215[v_ax0, v_ax1, T.int64(0), T.int64(0)]

@T.prim_func
def fused_conv2d25_add7(lv5204: T.Buffer((T.int64(2), T.int64(640), T.int64(128), T.int64(128)), "float32"), unet_up_blocks_2_resnets_1_conv_shortcut_weight: T.Buffer((T.int64(320), T.int64(640), T.int64(1), T.int64(1)), "float32"), lv5223: T.Buffer((T.int64(1), T.int64(320), T.int64(1), T.int64(1)), "float32"), var_T_add_intermediate: T.Buffer((T.int64(2), T.int64(320), T.int64(128), T.int64(128)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    pad_temp = T.alloc_buffer((T.int64(2), T.int64(640), T.int64(128), T.int64(128)))
    var_conv2d_nchw_intermediate = T.alloc_buffer((T.int64(2), T.int64(320), T.int64(128), T.int64(128)))
    for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(640), T.int64(128), T.int64(128)):
        with T.block("pad_temp"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(lv5204[v_i0, v_i1, v_i2, v_i3])
            T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
            pad_temp[v_i0, v_i1, v_i2, v_i3] = lv5204[v_i0, v_i1, v_i2, v_i3]
    for nn, ff, yy, xx, rc, ry, rx in T.grid(T.int64(2), T.int64(320), T.int64(128), T.int64(128), T.int64(640), T.int64(1), T.int64(1)):
        with T.block("conv2d_nchw"):
            v_nn, v_ff, v_yy, v_xx, v_rc, v_ry, v_rx = T.axis.remap("SSSSRRR", [nn, ff, yy, xx, rc, ry, rx])
            T.reads(pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx], unet_up_blocks_2_resnets_1_conv_shortcut_weight[v_ff, v_rc, v_ry, v_rx])
            T.writes(var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx])
            with T.init():
                var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = T.float32(0)
            var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] + pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx] * unet_up_blocks_2_resnets_1_conv_shortcut_weight[v_ff, v_rc, v_ry, v_rx]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(320), T.int64(128), T.int64(128)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], lv5223[T.int64(0), v_ax1, T.int64(0), T.int64(0)])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] + lv5223[T.int64(0), v_ax1, T.int64(0), T.int64(0)]

@T.prim_func
def fused_conv2d26_add28(lv5251: T.Buffer((T.int64(2), T.int64(320), T.int64(128), T.int64(128)), "float32"), unet_conv_out_weight: T.Buffer((T.int64(4), T.int64(320), T.int64(3), T.int64(3)), "float32"), lv5253: T.Buffer((T.int64(1), T.int64(4), T.int64(1), T.int64(1)), "float32"), var_T_add_intermediate: T.Buffer((T.int64(2), T.int64(4), T.int64(128), T.int64(128)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    pad_temp = T.alloc_buffer((T.int64(2), T.int64(320), T.int64(130), T.int64(130)))
    var_conv2d_nchw_intermediate = T.alloc_buffer((T.int64(2), T.int64(4), T.int64(128), T.int64(128)))
    for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(320), T.int64(130), T.int64(130)):
        with T.block("pad_temp"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(lv5251[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)])
            T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
            pad_temp[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(T.int64(1) <= v_i2 and v_i2 < T.int64(129) and T.int64(1) <= v_i3 and v_i3 < T.int64(129), lv5251[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)], T.float32(0))
    for nn, ff, yy, xx, rc, ry, rx in T.grid(T.int64(2), T.int64(4), T.int64(128), T.int64(128), T.int64(320), T.int64(3), T.int64(3)):
        with T.block("conv2d_nchw"):
            v_nn, v_ff, v_yy, v_xx, v_rc, v_ry, v_rx = T.axis.remap("SSSSRRR", [nn, ff, yy, xx, rc, ry, rx])
            T.reads(pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx], unet_conv_out_weight[v_ff, v_rc, v_ry, v_rx])
            T.writes(var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx])
            with T.init():
                var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = T.float32(0)
            var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] + pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx] * unet_conv_out_weight[v_ff, v_rc, v_ry, v_rx]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(4), T.int64(128), T.int64(128)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], lv5253[T.int64(0), v_ax1, T.int64(0), T.int64(0)])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] + lv5253[T.int64(0), v_ax1, T.int64(0), T.int64(0)]

@T.prim_func
def fused_conv2d27_add30(lv: T.Buffer((T.int64(1), T.int64(4), T.int64(128), T.int64(128)), "float32"), vae_post_quant_conv_weight: T.Buffer((T.int64(4), T.int64(4), T.int64(1), T.int64(1)), "float32"), lv2: T.Buffer((T.int64(1), T.int64(4), T.int64(1), T.int64(1)), "float32"), var_T_add_intermediate: T.Buffer((T.int64(1), T.int64(4), T.int64(128), T.int64(128)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    pad_temp = T.alloc_buffer((T.int64(1), T.int64(4), T.int64(128), T.int64(128)))
    var_conv2d_nchw_intermediate = T.alloc_buffer((T.int64(1), T.int64(4), T.int64(128), T.int64(128)))
    for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(4), T.int64(128), T.int64(128)):
        with T.block("pad_temp"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(lv[v_i0, v_i1, v_i2, v_i3])
            T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
            pad_temp[v_i0, v_i1, v_i2, v_i3] = lv[v_i0, v_i1, v_i2, v_i3]
    for nn, ff, yy, xx, rc, ry, rx in T.grid(T.int64(1), T.int64(4), T.int64(128), T.int64(128), T.int64(4), T.int64(1), T.int64(1)):
        with T.block("conv2d_nchw"):
            v_nn, v_ff, v_yy, v_xx, v_rc, v_ry, v_rx = T.axis.remap("SSSSRRR", [nn, ff, yy, xx, rc, ry, rx])
            T.reads(pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx], vae_post_quant_conv_weight[v_ff, v_rc, v_ry, v_rx])
            T.writes(var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx])
            with T.init():
                var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = T.float32(0)
            var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] + pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx] * vae_post_quant_conv_weight[v_ff, v_rc, v_ry, v_rx]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(4), T.int64(128), T.int64(128)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], lv2[v_ax0, v_ax1, T.int64(0), T.int64(0)])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] + lv2[v_ax0, v_ax1, T.int64(0), T.int64(0)]

@T.prim_func
def fused_conv2d28_add31(lv3: T.Buffer((T.int64(1), T.int64(4), T.int64(128), T.int64(128)), "float32"), vae_decoder_conv_in_weight: T.Buffer((T.int64(512), T.int64(4), T.int64(3), T.int64(3)), "float32"), lv5: T.Buffer((T.int64(1), T.int64(512), T.int64(1), T.int64(1)), "float32"), var_T_add_intermediate: T.Buffer((T.int64(1), T.int64(512), T.int64(128), T.int64(128)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    pad_temp = T.alloc_buffer((T.int64(1), T.int64(4), T.int64(130), T.int64(130)))
    var_conv2d_nchw_intermediate = T.alloc_buffer((T.int64(1), T.int64(512), T.int64(128), T.int64(128)))
    for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(4), T.int64(130), T.int64(130)):
        with T.block("pad_temp"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(lv3[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)])
            T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
            pad_temp[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(T.int64(1) <= v_i2 and v_i2 < T.int64(129) and T.int64(1) <= v_i3 and v_i3 < T.int64(129), lv3[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)], T.float32(0))
    for nn, ff, yy, xx, rc, ry, rx in T.grid(T.int64(1), T.int64(512), T.int64(128), T.int64(128), T.int64(4), T.int64(3), T.int64(3)):
        with T.block("conv2d_nchw"):
            v_nn, v_ff, v_yy, v_xx, v_rc, v_ry, v_rx = T.axis.remap("SSSSRRR", [nn, ff, yy, xx, rc, ry, rx])
            T.reads(pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx], vae_decoder_conv_in_weight[v_ff, v_rc, v_ry, v_rx])
            T.writes(var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx])
            with T.init():
                var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = T.float32(0)
            var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] + pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx] * vae_decoder_conv_in_weight[v_ff, v_rc, v_ry, v_rx]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(512), T.int64(128), T.int64(128)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], lv5[v_ax0, v_ax1, T.int64(0), T.int64(0)])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] + lv5[v_ax0, v_ax1, T.int64(0), T.int64(0)]

@T.prim_func
def fused_conv2d29_add31(lv8: T.Buffer((T.int64(1), T.int64(512), T.int64(128), T.int64(128)), "float32"), vae_decoder_mid_block_resnets_0_conv1_weight: T.Buffer((T.int64(512), T.int64(512), T.int64(3), T.int64(3)), "float32"), lv10: T.Buffer((T.int64(1), T.int64(512), T.int64(1), T.int64(1)), "float32"), var_T_add_intermediate: T.Buffer((T.int64(1), T.int64(512), T.int64(128), T.int64(128)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    pad_temp = T.alloc_buffer((T.int64(1), T.int64(512), T.int64(130), T.int64(130)))
    var_conv2d_nchw_intermediate = T.alloc_buffer((T.int64(1), T.int64(512), T.int64(128), T.int64(128)))
    for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(512), T.int64(130), T.int64(130)):
        with T.block("pad_temp"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(lv8[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)])
            T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
            pad_temp[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(T.int64(1) <= v_i2 and v_i2 < T.int64(129) and T.int64(1) <= v_i3 and v_i3 < T.int64(129), lv8[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)], T.float32(0))
    for nn, ff, yy, xx, rc, ry, rx in T.grid(T.int64(1), T.int64(512), T.int64(128), T.int64(128), T.int64(512), T.int64(3), T.int64(3)):
        with T.block("conv2d_nchw"):
            v_nn, v_ff, v_yy, v_xx, v_rc, v_ry, v_rx = T.axis.remap("SSSSRRR", [nn, ff, yy, xx, rc, ry, rx])
            T.reads(pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx], vae_decoder_mid_block_resnets_0_conv1_weight[v_ff, v_rc, v_ry, v_rx])
            T.writes(var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx])
            with T.init():
                var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = T.float32(0)
            var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] + pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx] * vae_decoder_mid_block_resnets_0_conv1_weight[v_ff, v_rc, v_ry, v_rx]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(512), T.int64(128), T.int64(128)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], lv10[v_ax0, v_ax1, T.int64(0), T.int64(0)])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] + lv10[v_ax0, v_ax1, T.int64(0), T.int64(0)]

@T.prim_func
def fused_conv2d29_add31_add32_divide6(lv13: T.Buffer((T.int64(1), T.int64(512), T.int64(128), T.int64(128)), "float32"), vae_decoder_mid_block_resnets_0_conv2_weight: T.Buffer((T.int64(512), T.int64(512), T.int64(3), T.int64(3)), "float32"), lv15: T.Buffer((T.int64(1), T.int64(512), T.int64(1), T.int64(1)), "float32"), lv6: T.Buffer((T.int64(1), T.int64(512), T.int64(128), T.int64(128)), "float32"), var_T_divide_intermediate: T.Buffer((T.int64(1), T.int64(512), T.int64(128), T.int64(128)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    pad_temp = T.alloc_buffer((T.int64(1), T.int64(512), T.int64(130), T.int64(130)))
    var_conv2d_nchw_intermediate = T.alloc_buffer((T.int64(1), T.int64(512), T.int64(128), T.int64(128)))
    var_T_add_intermediate = T.alloc_buffer((T.int64(1), T.int64(512), T.int64(128), T.int64(128)))
    var_T_add_intermediate_1 = T.alloc_buffer((T.int64(1), T.int64(512), T.int64(128), T.int64(128)))
    for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(512), T.int64(130), T.int64(130)):
        with T.block("pad_temp"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(lv13[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)])
            T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
            pad_temp[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(T.int64(1) <= v_i2 and v_i2 < T.int64(129) and T.int64(1) <= v_i3 and v_i3 < T.int64(129), lv13[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)], T.float32(0))
    for nn, ff, yy, xx, rc, ry, rx in T.grid(T.int64(1), T.int64(512), T.int64(128), T.int64(128), T.int64(512), T.int64(3), T.int64(3)):
        with T.block("conv2d_nchw"):
            v_nn, v_ff, v_yy, v_xx, v_rc, v_ry, v_rx = T.axis.remap("SSSSRRR", [nn, ff, yy, xx, rc, ry, rx])
            T.reads(pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx], vae_decoder_mid_block_resnets_0_conv2_weight[v_ff, v_rc, v_ry, v_rx])
            T.writes(var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx])
            with T.init():
                var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = T.float32(0)
            var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] + pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx] * vae_decoder_mid_block_resnets_0_conv2_weight[v_ff, v_rc, v_ry, v_rx]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(512), T.int64(128), T.int64(128)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], lv15[v_ax0, v_ax1, T.int64(0), T.int64(0)])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] + lv15[v_ax0, v_ax1, T.int64(0), T.int64(0)]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(512), T.int64(128), T.int64(128)):
        with T.block("T_add_1"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(lv6[v_ax0, v_ax1, v_ax2, v_ax3], var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            T.writes(var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3] = lv6[v_ax0, v_ax1, v_ax2, v_ax3] + var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(512), T.int64(128), T.int64(128)):
        with T.block("T_divide"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3])
            T.writes(var_T_divide_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_divide_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3]

@T.prim_func
def fused_conv2d29_add31_add32_divide6_divide6(lv61: T.Buffer((T.int64(1), T.int64(512), T.int64(128), T.int64(128)), "float32"), vae_decoder_mid_block_resnets_1_conv2_weight: T.Buffer((T.int64(512), T.int64(512), T.int64(3), T.int64(3)), "float32"), lv63: T.Buffer((T.int64(1), T.int64(512), T.int64(1), T.int64(1)), "float32"), lv54: T.Buffer((T.int64(1), T.int64(512), T.int64(128), T.int64(128)), "float32"), var_T_divide_intermediate: T.Buffer((T.int64(1), T.int64(512), T.int64(128), T.int64(128)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    pad_temp = T.alloc_buffer((T.int64(1), T.int64(512), T.int64(130), T.int64(130)))
    var_conv2d_nchw_intermediate = T.alloc_buffer((T.int64(1), T.int64(512), T.int64(128), T.int64(128)))
    var_T_add_intermediate = T.alloc_buffer((T.int64(1), T.int64(512), T.int64(128), T.int64(128)))
    var_T_add_intermediate_1 = T.alloc_buffer((T.int64(1), T.int64(512), T.int64(128), T.int64(128)))
    var_T_divide_intermediate_1 = T.alloc_buffer((T.int64(1), T.int64(512), T.int64(128), T.int64(128)))
    for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(512), T.int64(130), T.int64(130)):
        with T.block("pad_temp"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(lv61[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)])
            T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
            pad_temp[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(T.int64(1) <= v_i2 and v_i2 < T.int64(129) and T.int64(1) <= v_i3 and v_i3 < T.int64(129), lv61[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)], T.float32(0))
    for nn, ff, yy, xx, rc, ry, rx in T.grid(T.int64(1), T.int64(512), T.int64(128), T.int64(128), T.int64(512), T.int64(3), T.int64(3)):
        with T.block("conv2d_nchw"):
            v_nn, v_ff, v_yy, v_xx, v_rc, v_ry, v_rx = T.axis.remap("SSSSRRR", [nn, ff, yy, xx, rc, ry, rx])
            T.reads(pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx], vae_decoder_mid_block_resnets_1_conv2_weight[v_ff, v_rc, v_ry, v_rx])
            T.writes(var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx])
            with T.init():
                var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = T.float32(0)
            var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] + pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx] * vae_decoder_mid_block_resnets_1_conv2_weight[v_ff, v_rc, v_ry, v_rx]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(512), T.int64(128), T.int64(128)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], lv63[v_ax0, v_ax1, T.int64(0), T.int64(0)])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] + lv63[v_ax0, v_ax1, T.int64(0), T.int64(0)]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(512), T.int64(128), T.int64(128)):
        with T.block("T_add_1"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(lv54[v_ax0, v_ax1, v_ax2, v_ax3], var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            T.writes(var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3] = lv54[v_ax0, v_ax1, v_ax2, v_ax3] + var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(512), T.int64(128), T.int64(128)):
        with T.block("T_divide"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3])
            T.writes(var_T_divide_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_divide_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3] = var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(512), T.int64(128), T.int64(128)):
        with T.block("T_divide_1"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_divide_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3])
            T.writes(var_T_divide_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_divide_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_T_divide_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3]

@T.prim_func
def fused_conv2d2_add11(lv86: T.Buffer((T.int64(2), T.int64(320), T.int64(128), T.int64(128)), "float32"), unet_down_blocks_0_downsamplers_0_conv_weight: T.Buffer((T.int64(320), T.int64(320), T.int64(3), T.int64(3)), "float32"), lv88: T.Buffer((T.int64(1), T.int64(320), T.int64(1), T.int64(1)), "float32"), var_T_add_intermediate: T.Buffer((T.int64(2), T.int64(320), T.int64(64), T.int64(64)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    pad_temp = T.alloc_buffer((T.int64(2), T.int64(320), T.int64(130), T.int64(130)))
    var_conv2d_nchw_intermediate = T.alloc_buffer((T.int64(2), T.int64(320), T.int64(64), T.int64(64)))
    for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(320), T.int64(130), T.int64(130)):
        with T.block("pad_temp"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(lv86[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)])
            T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
            pad_temp[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(T.int64(1) <= v_i2 and v_i2 < T.int64(129) and T.int64(1) <= v_i3 and v_i3 < T.int64(129), lv86[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)], T.float32(0))
    for nn, ff, yy, xx, rc, ry, rx in T.grid(T.int64(2), T.int64(320), T.int64(64), T.int64(64), T.int64(320), T.int64(3), T.int64(3)):
        with T.block("conv2d_nchw"):
            v_nn, v_ff, v_yy, v_xx, v_rc, v_ry, v_rx = T.axis.remap("SSSSRRR", [nn, ff, yy, xx, rc, ry, rx])
            T.reads(pad_temp[v_nn, v_rc, v_yy * T.int64(2) + v_ry, v_xx * T.int64(2) + v_rx], unet_down_blocks_0_downsamplers_0_conv_weight[v_ff, v_rc, v_ry, v_rx])
            T.writes(var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx])
            with T.init():
                var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = T.float32(0)
            var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] + pad_temp[v_nn, v_rc, v_yy * T.int64(2) + v_ry, v_xx * T.int64(2) + v_rx] * unet_down_blocks_0_downsamplers_0_conv_weight[v_ff, v_rc, v_ry, v_rx]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(320), T.int64(64), T.int64(64)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], lv88[T.int64(0), v_ax1, T.int64(0), T.int64(0)])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] + lv88[T.int64(0), v_ax1, T.int64(0), T.int64(0)]

@T.prim_func
def fused_conv2d30_add34(lv104: T.Buffer((T.int64(1), T.int64(512), T.int64(256), T.int64(256)), "float32"), vae_decoder_up_blocks_0_upsamplers_0_conv_weight: T.Buffer((T.int64(512), T.int64(512), T.int64(3), T.int64(3)), "float32"), lv106: T.Buffer((T.int64(1), T.int64(512), T.int64(1), T.int64(1)), "float32"), var_T_add_intermediate: T.Buffer((T.int64(1), T.int64(512), T.int64(256), T.int64(256)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    pad_temp = T.alloc_buffer((T.int64(1), T.int64(512), T.int64(258), T.int64(258)))
    var_conv2d_nchw_intermediate = T.alloc_buffer((T.int64(1), T.int64(512), T.int64(256), T.int64(256)))
    for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(512), T.int64(258), T.int64(258)):
        with T.block("pad_temp"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(lv104[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)])
            T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
            pad_temp[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(T.int64(1) <= v_i2 and v_i2 < T.int64(257) and T.int64(1) <= v_i3 and v_i3 < T.int64(257), lv104[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)], T.float32(0))
    for nn, ff, yy, xx, rc, ry, rx in T.grid(T.int64(1), T.int64(512), T.int64(256), T.int64(256), T.int64(512), T.int64(3), T.int64(3)):
        with T.block("conv2d_nchw"):
            v_nn, v_ff, v_yy, v_xx, v_rc, v_ry, v_rx = T.axis.remap("SSSSRRR", [nn, ff, yy, xx, rc, ry, rx])
            T.reads(pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx], vae_decoder_up_blocks_0_upsamplers_0_conv_weight[v_ff, v_rc, v_ry, v_rx])
            T.writes(var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx])
            with T.init():
                var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = T.float32(0)
            var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] + pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx] * vae_decoder_up_blocks_0_upsamplers_0_conv_weight[v_ff, v_rc, v_ry, v_rx]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(512), T.int64(256), T.int64(256)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], lv106[v_ax0, v_ax1, T.int64(0), T.int64(0)])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] + lv106[v_ax0, v_ax1, T.int64(0), T.int64(0)]

@T.prim_func
def fused_conv2d30_add34_add35_divide7(lv114: T.Buffer((T.int64(1), T.int64(512), T.int64(256), T.int64(256)), "float32"), vae_decoder_up_blocks_1_resnets_0_conv2_weight: T.Buffer((T.int64(512), T.int64(512), T.int64(3), T.int64(3)), "float32"), lv116: T.Buffer((T.int64(1), T.int64(512), T.int64(1), T.int64(1)), "float32"), lv107: T.Buffer((T.int64(1), T.int64(512), T.int64(256), T.int64(256)), "float32"), var_T_divide_intermediate: T.Buffer((T.int64(1), T.int64(512), T.int64(256), T.int64(256)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    pad_temp = T.alloc_buffer((T.int64(1), T.int64(512), T.int64(258), T.int64(258)))
    var_conv2d_nchw_intermediate = T.alloc_buffer((T.int64(1), T.int64(512), T.int64(256), T.int64(256)))
    var_T_add_intermediate = T.alloc_buffer((T.int64(1), T.int64(512), T.int64(256), T.int64(256)))
    var_T_add_intermediate_1 = T.alloc_buffer((T.int64(1), T.int64(512), T.int64(256), T.int64(256)))
    for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(512), T.int64(258), T.int64(258)):
        with T.block("pad_temp"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(lv114[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)])
            T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
            pad_temp[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(T.int64(1) <= v_i2 and v_i2 < T.int64(257) and T.int64(1) <= v_i3 and v_i3 < T.int64(257), lv114[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)], T.float32(0))
    for nn, ff, yy, xx, rc, ry, rx in T.grid(T.int64(1), T.int64(512), T.int64(256), T.int64(256), T.int64(512), T.int64(3), T.int64(3)):
        with T.block("conv2d_nchw"):
            v_nn, v_ff, v_yy, v_xx, v_rc, v_ry, v_rx = T.axis.remap("SSSSRRR", [nn, ff, yy, xx, rc, ry, rx])
            T.reads(pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx], vae_decoder_up_blocks_1_resnets_0_conv2_weight[v_ff, v_rc, v_ry, v_rx])
            T.writes(var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx])
            with T.init():
                var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = T.float32(0)
            var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] + pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx] * vae_decoder_up_blocks_1_resnets_0_conv2_weight[v_ff, v_rc, v_ry, v_rx]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(512), T.int64(256), T.int64(256)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], lv116[v_ax0, v_ax1, T.int64(0), T.int64(0)])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] + lv116[v_ax0, v_ax1, T.int64(0), T.int64(0)]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(512), T.int64(256), T.int64(256)):
        with T.block("T_add_1"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(lv107[v_ax0, v_ax1, v_ax2, v_ax3], var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            T.writes(var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3] = lv107[v_ax0, v_ax1, v_ax2, v_ax3] + var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(512), T.int64(256), T.int64(256)):
        with T.block("T_divide"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3])
            T.writes(var_T_divide_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_divide_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3]

@T.prim_func
def fused_conv2d31_add36(lv144: T.Buffer((T.int64(1), T.int64(512), T.int64(512), T.int64(512)), "float32"), vae_decoder_up_blocks_1_upsamplers_0_conv_weight: T.Buffer((T.int64(512), T.int64(512), T.int64(3), T.int64(3)), "float32"), lv146: T.Buffer((T.int64(1), T.int64(512), T.int64(1), T.int64(1)), "float32"), var_T_add_intermediate: T.Buffer((T.int64(1), T.int64(512), T.int64(512), T.int64(512)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    pad_temp = T.alloc_buffer((T.int64(1), T.int64(512), T.int64(514), T.int64(514)))
    var_conv2d_nchw_intermediate = T.alloc_buffer((T.int64(1), T.int64(512), T.int64(512), T.int64(512)))
    for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(512), T.int64(514), T.int64(514)):
        with T.block("pad_temp"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(lv144[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)])
            T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
            pad_temp[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(T.int64(1) <= v_i2 and v_i2 < T.int64(513) and T.int64(1) <= v_i3 and v_i3 < T.int64(513), lv144[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)], T.float32(0))
    for nn, ff, yy, xx, rc, ry, rx in T.grid(T.int64(1), T.int64(512), T.int64(512), T.int64(512), T.int64(512), T.int64(3), T.int64(3)):
        with T.block("conv2d_nchw"):
            v_nn, v_ff, v_yy, v_xx, v_rc, v_ry, v_rx = T.axis.remap("SSSSRRR", [nn, ff, yy, xx, rc, ry, rx])
            T.reads(pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx], vae_decoder_up_blocks_1_upsamplers_0_conv_weight[v_ff, v_rc, v_ry, v_rx])
            T.writes(var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx])
            with T.init():
                var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = T.float32(0)
            var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] + pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx] * vae_decoder_up_blocks_1_upsamplers_0_conv_weight[v_ff, v_rc, v_ry, v_rx]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(512), T.int64(512), T.int64(512)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], lv146[v_ax0, v_ax1, T.int64(0), T.int64(0)])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] + lv146[v_ax0, v_ax1, T.int64(0), T.int64(0)]

@T.prim_func
def fused_conv2d32_add37(lv149: T.Buffer((T.int64(1), T.int64(512), T.int64(512), T.int64(512)), "float32"), vae_decoder_up_blocks_2_resnets_0_conv1_weight: T.Buffer((T.int64(256), T.int64(512), T.int64(3), T.int64(3)), "float32"), lv151: T.Buffer((T.int64(1), T.int64(256), T.int64(1), T.int64(1)), "float32"), var_T_add_intermediate: T.Buffer((T.int64(1), T.int64(256), T.int64(512), T.int64(512)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    pad_temp = T.alloc_buffer((T.int64(1), T.int64(512), T.int64(514), T.int64(514)))
    var_conv2d_nchw_intermediate = T.alloc_buffer((T.int64(1), T.int64(256), T.int64(512), T.int64(512)))
    for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(512), T.int64(514), T.int64(514)):
        with T.block("pad_temp"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(lv149[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)])
            T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
            pad_temp[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(T.int64(1) <= v_i2 and v_i2 < T.int64(513) and T.int64(1) <= v_i3 and v_i3 < T.int64(513), lv149[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)], T.float32(0))
    for nn, ff, yy, xx, rc, ry, rx in T.grid(T.int64(1), T.int64(256), T.int64(512), T.int64(512), T.int64(512), T.int64(3), T.int64(3)):
        with T.block("conv2d_nchw"):
            v_nn, v_ff, v_yy, v_xx, v_rc, v_ry, v_rx = T.axis.remap("SSSSRRR", [nn, ff, yy, xx, rc, ry, rx])
            T.reads(pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx], vae_decoder_up_blocks_2_resnets_0_conv1_weight[v_ff, v_rc, v_ry, v_rx])
            T.writes(var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx])
            with T.init():
                var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = T.float32(0)
            var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] + pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx] * vae_decoder_up_blocks_2_resnets_0_conv1_weight[v_ff, v_rc, v_ry, v_rx]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(256), T.int64(512), T.int64(512)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], lv151[v_ax0, v_ax1, T.int64(0), T.int64(0)])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] + lv151[v_ax0, v_ax1, T.int64(0), T.int64(0)]

@T.prim_func
def fused_conv2d33_add37(lv164: T.Buffer((T.int64(1), T.int64(256), T.int64(512), T.int64(512)), "float32"), vae_decoder_up_blocks_2_resnets_1_conv1_weight: T.Buffer((T.int64(256), T.int64(256), T.int64(3), T.int64(3)), "float32"), lv166: T.Buffer((T.int64(1), T.int64(256), T.int64(1), T.int64(1)), "float32"), var_T_add_intermediate: T.Buffer((T.int64(1), T.int64(256), T.int64(512), T.int64(512)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    pad_temp = T.alloc_buffer((T.int64(1), T.int64(256), T.int64(514), T.int64(514)))
    var_conv2d_nchw_intermediate = T.alloc_buffer((T.int64(1), T.int64(256), T.int64(512), T.int64(512)))
    for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(256), T.int64(514), T.int64(514)):
        with T.block("pad_temp"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(lv164[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)])
            T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
            pad_temp[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(T.int64(1) <= v_i2 and v_i2 < T.int64(513) and T.int64(1) <= v_i3 and v_i3 < T.int64(513), lv164[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)], T.float32(0))
    for nn, ff, yy, xx, rc, ry, rx in T.grid(T.int64(1), T.int64(256), T.int64(512), T.int64(512), T.int64(256), T.int64(3), T.int64(3)):
        with T.block("conv2d_nchw"):
            v_nn, v_ff, v_yy, v_xx, v_rc, v_ry, v_rx = T.axis.remap("SSSSRRR", [nn, ff, yy, xx, rc, ry, rx])
            T.reads(pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx], vae_decoder_up_blocks_2_resnets_1_conv1_weight[v_ff, v_rc, v_ry, v_rx])
            T.writes(var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx])
            with T.init():
                var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = T.float32(0)
            var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] + pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx] * vae_decoder_up_blocks_2_resnets_1_conv1_weight[v_ff, v_rc, v_ry, v_rx]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(256), T.int64(512), T.int64(512)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], lv166[v_ax0, v_ax1, T.int64(0), T.int64(0)])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] + lv166[v_ax0, v_ax1, T.int64(0), T.int64(0)]

@T.prim_func
def fused_conv2d33_add37_add38_divide8(lv154: T.Buffer((T.int64(1), T.int64(256), T.int64(512), T.int64(512)), "float32"), vae_decoder_up_blocks_2_resnets_0_conv2_weight: T.Buffer((T.int64(256), T.int64(256), T.int64(3), T.int64(3)), "float32"), lv156: T.Buffer((T.int64(1), T.int64(256), T.int64(1), T.int64(1)), "float32"), lv160: T.Buffer((T.int64(1), T.int64(256), T.int64(512), T.int64(512)), "float32"), var_T_divide_intermediate: T.Buffer((T.int64(1), T.int64(256), T.int64(512), T.int64(512)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    pad_temp = T.alloc_buffer((T.int64(1), T.int64(256), T.int64(514), T.int64(514)))
    var_conv2d_nchw_intermediate = T.alloc_buffer((T.int64(1), T.int64(256), T.int64(512), T.int64(512)))
    var_T_add_intermediate = T.alloc_buffer((T.int64(1), T.int64(256), T.int64(512), T.int64(512)))
    var_T_add_intermediate_1 = T.alloc_buffer((T.int64(1), T.int64(256), T.int64(512), T.int64(512)))
    for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(256), T.int64(514), T.int64(514)):
        with T.block("pad_temp"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(lv154[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)])
            T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
            pad_temp[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(T.int64(1) <= v_i2 and v_i2 < T.int64(513) and T.int64(1) <= v_i3 and v_i3 < T.int64(513), lv154[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)], T.float32(0))
    for nn, ff, yy, xx, rc, ry, rx in T.grid(T.int64(1), T.int64(256), T.int64(512), T.int64(512), T.int64(256), T.int64(3), T.int64(3)):
        with T.block("conv2d_nchw"):
            v_nn, v_ff, v_yy, v_xx, v_rc, v_ry, v_rx = T.axis.remap("SSSSRRR", [nn, ff, yy, xx, rc, ry, rx])
            T.reads(pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx], vae_decoder_up_blocks_2_resnets_0_conv2_weight[v_ff, v_rc, v_ry, v_rx])
            T.writes(var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx])
            with T.init():
                var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = T.float32(0)
            var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] + pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx] * vae_decoder_up_blocks_2_resnets_0_conv2_weight[v_ff, v_rc, v_ry, v_rx]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(256), T.int64(512), T.int64(512)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], lv156[v_ax0, v_ax1, T.int64(0), T.int64(0)])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] + lv156[v_ax0, v_ax1, T.int64(0), T.int64(0)]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(256), T.int64(512), T.int64(512)):
        with T.block("T_add_1"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(lv160[v_ax0, v_ax1, v_ax2, v_ax3], var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            T.writes(var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3] = lv160[v_ax0, v_ax1, v_ax2, v_ax3] + var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(256), T.int64(512), T.int64(512)):
        with T.block("T_divide"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3])
            T.writes(var_T_divide_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_divide_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3]

@T.prim_func
def fused_conv2d34_add37(lv147: T.Buffer((T.int64(1), T.int64(512), T.int64(512), T.int64(512)), "float32"), vae_decoder_up_blocks_2_resnets_0_conv_shortcut_weight: T.Buffer((T.int64(256), T.int64(512), T.int64(1), T.int64(1)), "float32"), lv159: T.Buffer((T.int64(1), T.int64(256), T.int64(1), T.int64(1)), "float32"), var_T_add_intermediate: T.Buffer((T.int64(1), T.int64(256), T.int64(512), T.int64(512)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    pad_temp = T.alloc_buffer((T.int64(1), T.int64(512), T.int64(512), T.int64(512)))
    var_conv2d_nchw_intermediate = T.alloc_buffer((T.int64(1), T.int64(256), T.int64(512), T.int64(512)))
    for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(512), T.int64(512), T.int64(512)):
        with T.block("pad_temp"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(lv147[v_i0, v_i1, v_i2, v_i3])
            T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
            pad_temp[v_i0, v_i1, v_i2, v_i3] = lv147[v_i0, v_i1, v_i2, v_i3]
    for nn, ff, yy, xx, rc, ry, rx in T.grid(T.int64(1), T.int64(256), T.int64(512), T.int64(512), T.int64(512), T.int64(1), T.int64(1)):
        with T.block("conv2d_nchw"):
            v_nn, v_ff, v_yy, v_xx, v_rc, v_ry, v_rx = T.axis.remap("SSSSRRR", [nn, ff, yy, xx, rc, ry, rx])
            T.reads(pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx], vae_decoder_up_blocks_2_resnets_0_conv_shortcut_weight[v_ff, v_rc, v_ry, v_rx])
            T.writes(var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx])
            with T.init():
                var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = T.float32(0)
            var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] + pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx] * vae_decoder_up_blocks_2_resnets_0_conv_shortcut_weight[v_ff, v_rc, v_ry, v_rx]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(256), T.int64(512), T.int64(512)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], lv159[v_ax0, v_ax1, T.int64(0), T.int64(0)])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] + lv159[v_ax0, v_ax1, T.int64(0), T.int64(0)]

@T.prim_func
def fused_conv2d35_add39(lv187: T.Buffer((T.int64(1), T.int64(256), T.int64(1024), T.int64(1024)), "float32"), vae_decoder_up_blocks_2_upsamplers_0_conv_weight: T.Buffer((T.int64(256), T.int64(256), T.int64(3), T.int64(3)), "float32"), lv189: T.Buffer((T.int64(1), T.int64(256), T.int64(1), T.int64(1)), "float32"), var_T_add_intermediate: T.Buffer((T.int64(1), T.int64(256), T.int64(1024), T.int64(1024)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    pad_temp = T.alloc_buffer((T.int64(1), T.int64(256), T.int64(1026), T.int64(1026)))
    var_conv2d_nchw_intermediate = T.alloc_buffer((T.int64(1), T.int64(256), T.int64(1024), T.int64(1024)))
    for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(256), T.int64(1026), T.int64(1026)):
        with T.block("pad_temp"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(lv187[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)])
            T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
            pad_temp[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(T.int64(1) <= v_i2 and v_i2 < T.int64(1025) and T.int64(1) <= v_i3 and v_i3 < T.int64(1025), lv187[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)], T.float32(0))
    for nn, ff, yy, xx, rc, ry, rx in T.grid(T.int64(1), T.int64(256), T.int64(1024), T.int64(1024), T.int64(256), T.int64(3), T.int64(3)):
        with T.block("conv2d_nchw"):
            v_nn, v_ff, v_yy, v_xx, v_rc, v_ry, v_rx = T.axis.remap("SSSSRRR", [nn, ff, yy, xx, rc, ry, rx])
            T.reads(pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx], vae_decoder_up_blocks_2_upsamplers_0_conv_weight[v_ff, v_rc, v_ry, v_rx])
            T.writes(var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx])
            with T.init():
                var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = T.float32(0)
            var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] + pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx] * vae_decoder_up_blocks_2_upsamplers_0_conv_weight[v_ff, v_rc, v_ry, v_rx]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(256), T.int64(1024), T.int64(1024)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], lv189[v_ax0, v_ax1, T.int64(0), T.int64(0)])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] + lv189[v_ax0, v_ax1, T.int64(0), T.int64(0)]

@T.prim_func
def fused_conv2d36_add40(lv192: T.Buffer((T.int64(1), T.int64(256), T.int64(1024), T.int64(1024)), "float32"), vae_decoder_up_blocks_3_resnets_0_conv1_weight: T.Buffer((T.int64(128), T.int64(256), T.int64(3), T.int64(3)), "float32"), lv194: T.Buffer((T.int64(1), T.int64(128), T.int64(1), T.int64(1)), "float32"), var_T_add_intermediate: T.Buffer((T.int64(1), T.int64(128), T.int64(1024), T.int64(1024)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    pad_temp = T.alloc_buffer((T.int64(1), T.int64(256), T.int64(1026), T.int64(1026)))
    var_conv2d_nchw_intermediate = T.alloc_buffer((T.int64(1), T.int64(128), T.int64(1024), T.int64(1024)))
    for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(256), T.int64(1026), T.int64(1026)):
        with T.block("pad_temp"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(lv192[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)])
            T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
            pad_temp[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(T.int64(1) <= v_i2 and v_i2 < T.int64(1025) and T.int64(1) <= v_i3 and v_i3 < T.int64(1025), lv192[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)], T.float32(0))
    for nn, ff, yy, xx, rc, ry, rx in T.grid(T.int64(1), T.int64(128), T.int64(1024), T.int64(1024), T.int64(256), T.int64(3), T.int64(3)):
        with T.block("conv2d_nchw"):
            v_nn, v_ff, v_yy, v_xx, v_rc, v_ry, v_rx = T.axis.remap("SSSSRRR", [nn, ff, yy, xx, rc, ry, rx])
            T.reads(pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx], vae_decoder_up_blocks_3_resnets_0_conv1_weight[v_ff, v_rc, v_ry, v_rx])
            T.writes(var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx])
            with T.init():
                var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = T.float32(0)
            var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] + pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx] * vae_decoder_up_blocks_3_resnets_0_conv1_weight[v_ff, v_rc, v_ry, v_rx]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(128), T.int64(1024), T.int64(1024)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], lv194[v_ax0, v_ax1, T.int64(0), T.int64(0)])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] + lv194[v_ax0, v_ax1, T.int64(0), T.int64(0)]

@T.prim_func
def fused_conv2d37_add40(lv207: T.Buffer((T.int64(1), T.int64(128), T.int64(1024), T.int64(1024)), "float32"), vae_decoder_up_blocks_3_resnets_1_conv1_weight: T.Buffer((T.int64(128), T.int64(128), T.int64(3), T.int64(3)), "float32"), lv209: T.Buffer((T.int64(1), T.int64(128), T.int64(1), T.int64(1)), "float32"), var_T_add_intermediate: T.Buffer((T.int64(1), T.int64(128), T.int64(1024), T.int64(1024)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    pad_temp = T.alloc_buffer((T.int64(1), T.int64(128), T.int64(1026), T.int64(1026)))
    var_conv2d_nchw_intermediate = T.alloc_buffer((T.int64(1), T.int64(128), T.int64(1024), T.int64(1024)))
    for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(128), T.int64(1026), T.int64(1026)):
        with T.block("pad_temp"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(lv207[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)])
            T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
            pad_temp[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(T.int64(1) <= v_i2 and v_i2 < T.int64(1025) and T.int64(1) <= v_i3 and v_i3 < T.int64(1025), lv207[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)], T.float32(0))
    for nn, ff, yy, xx, rc, ry, rx in T.grid(T.int64(1), T.int64(128), T.int64(1024), T.int64(1024), T.int64(128), T.int64(3), T.int64(3)):
        with T.block("conv2d_nchw"):
            v_nn, v_ff, v_yy, v_xx, v_rc, v_ry, v_rx = T.axis.remap("SSSSRRR", [nn, ff, yy, xx, rc, ry, rx])
            T.reads(pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx], vae_decoder_up_blocks_3_resnets_1_conv1_weight[v_ff, v_rc, v_ry, v_rx])
            T.writes(var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx])
            with T.init():
                var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = T.float32(0)
            var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] + pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx] * vae_decoder_up_blocks_3_resnets_1_conv1_weight[v_ff, v_rc, v_ry, v_rx]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(128), T.int64(1024), T.int64(1024)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], lv209[v_ax0, v_ax1, T.int64(0), T.int64(0)])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] + lv209[v_ax0, v_ax1, T.int64(0), T.int64(0)]

@T.prim_func
def fused_conv2d37_add40_add41_divide9(lv197: T.Buffer((T.int64(1), T.int64(128), T.int64(1024), T.int64(1024)), "float32"), vae_decoder_up_blocks_3_resnets_0_conv2_weight: T.Buffer((T.int64(128), T.int64(128), T.int64(3), T.int64(3)), "float32"), lv199: T.Buffer((T.int64(1), T.int64(128), T.int64(1), T.int64(1)), "float32"), lv203: T.Buffer((T.int64(1), T.int64(128), T.int64(1024), T.int64(1024)), "float32"), var_T_divide_intermediate: T.Buffer((T.int64(1), T.int64(128), T.int64(1024), T.int64(1024)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    pad_temp = T.alloc_buffer((T.int64(1), T.int64(128), T.int64(1026), T.int64(1026)))
    var_conv2d_nchw_intermediate = T.alloc_buffer((T.int64(1), T.int64(128), T.int64(1024), T.int64(1024)))
    var_T_add_intermediate = T.alloc_buffer((T.int64(1), T.int64(128), T.int64(1024), T.int64(1024)))
    var_T_add_intermediate_1 = T.alloc_buffer((T.int64(1), T.int64(128), T.int64(1024), T.int64(1024)))
    for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(128), T.int64(1026), T.int64(1026)):
        with T.block("pad_temp"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(lv197[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)])
            T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
            pad_temp[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(T.int64(1) <= v_i2 and v_i2 < T.int64(1025) and T.int64(1) <= v_i3 and v_i3 < T.int64(1025), lv197[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)], T.float32(0))
    for nn, ff, yy, xx, rc, ry, rx in T.grid(T.int64(1), T.int64(128), T.int64(1024), T.int64(1024), T.int64(128), T.int64(3), T.int64(3)):
        with T.block("conv2d_nchw"):
            v_nn, v_ff, v_yy, v_xx, v_rc, v_ry, v_rx = T.axis.remap("SSSSRRR", [nn, ff, yy, xx, rc, ry, rx])
            T.reads(pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx], vae_decoder_up_blocks_3_resnets_0_conv2_weight[v_ff, v_rc, v_ry, v_rx])
            T.writes(var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx])
            with T.init():
                var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = T.float32(0)
            var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] + pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx] * vae_decoder_up_blocks_3_resnets_0_conv2_weight[v_ff, v_rc, v_ry, v_rx]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(128), T.int64(1024), T.int64(1024)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], lv199[v_ax0, v_ax1, T.int64(0), T.int64(0)])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] + lv199[v_ax0, v_ax1, T.int64(0), T.int64(0)]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(128), T.int64(1024), T.int64(1024)):
        with T.block("T_add_1"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(lv203[v_ax0, v_ax1, v_ax2, v_ax3], var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            T.writes(var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3] = lv203[v_ax0, v_ax1, v_ax2, v_ax3] + var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(128), T.int64(1024), T.int64(1024)):
        with T.block("T_divide"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3])
            T.writes(var_T_divide_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_divide_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3]

@T.prim_func
def fused_conv2d38_add40(lv190: T.Buffer((T.int64(1), T.int64(256), T.int64(1024), T.int64(1024)), "float32"), vae_decoder_up_blocks_3_resnets_0_conv_shortcut_weight: T.Buffer((T.int64(128), T.int64(256), T.int64(1), T.int64(1)), "float32"), lv202: T.Buffer((T.int64(1), T.int64(128), T.int64(1), T.int64(1)), "float32"), var_T_add_intermediate: T.Buffer((T.int64(1), T.int64(128), T.int64(1024), T.int64(1024)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    pad_temp = T.alloc_buffer((T.int64(1), T.int64(256), T.int64(1024), T.int64(1024)))
    var_conv2d_nchw_intermediate = T.alloc_buffer((T.int64(1), T.int64(128), T.int64(1024), T.int64(1024)))
    for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(256), T.int64(1024), T.int64(1024)):
        with T.block("pad_temp"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(lv190[v_i0, v_i1, v_i2, v_i3])
            T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
            pad_temp[v_i0, v_i1, v_i2, v_i3] = lv190[v_i0, v_i1, v_i2, v_i3]
    for nn, ff, yy, xx, rc, ry, rx in T.grid(T.int64(1), T.int64(128), T.int64(1024), T.int64(1024), T.int64(256), T.int64(1), T.int64(1)):
        with T.block("conv2d_nchw"):
            v_nn, v_ff, v_yy, v_xx, v_rc, v_ry, v_rx = T.axis.remap("SSSSRRR", [nn, ff, yy, xx, rc, ry, rx])
            T.reads(pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx], vae_decoder_up_blocks_3_resnets_0_conv_shortcut_weight[v_ff, v_rc, v_ry, v_rx])
            T.writes(var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx])
            with T.init():
                var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = T.float32(0)
            var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] + pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx] * vae_decoder_up_blocks_3_resnets_0_conv_shortcut_weight[v_ff, v_rc, v_ry, v_rx]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(128), T.int64(1024), T.int64(1024)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], lv202[v_ax0, v_ax1, T.int64(0), T.int64(0)])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] + lv202[v_ax0, v_ax1, T.int64(0), T.int64(0)]

@T.prim_func
def fused_conv2d39_add42_divide10_add43_tir_clip(lv231: T.Buffer((T.int64(1), T.int64(128), T.int64(1024), T.int64(1024)), "float32"), vae_decoder_conv_out_weight: T.Buffer((T.int64(3), T.int64(128), T.int64(3), T.int64(3)), "float32"), lv233: T.Buffer((T.int64(1), T.int64(3), T.int64(1), T.int64(1)), "float32"), var_compute_intermediate: T.Buffer((T.int64(1), T.int64(3), T.int64(1024), T.int64(1024)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    pad_temp = T.alloc_buffer((T.int64(1), T.int64(128), T.int64(1026), T.int64(1026)))
    var_conv2d_nchw_intermediate = T.alloc_buffer((T.int64(1), T.int64(3), T.int64(1024), T.int64(1024)))
    var_T_add_intermediate = T.alloc_buffer((T.int64(1), T.int64(3), T.int64(1024), T.int64(1024)))
    var_T_divide_intermediate = T.alloc_buffer((T.int64(1), T.int64(3), T.int64(1024), T.int64(1024)))
    var_T_add_intermediate_1 = T.alloc_buffer((T.int64(1), T.int64(3), T.int64(1024), T.int64(1024)))
    for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(128), T.int64(1026), T.int64(1026)):
        with T.block("pad_temp"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(lv231[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)])
            T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
            pad_temp[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(T.int64(1) <= v_i2 and v_i2 < T.int64(1025) and T.int64(1) <= v_i3 and v_i3 < T.int64(1025), lv231[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)], T.float32(0))
    for nn, ff, yy, xx, rc, ry, rx in T.grid(T.int64(1), T.int64(3), T.int64(1024), T.int64(1024), T.int64(128), T.int64(3), T.int64(3)):
        with T.block("conv2d_nchw"):
            v_nn, v_ff, v_yy, v_xx, v_rc, v_ry, v_rx = T.axis.remap("SSSSRRR", [nn, ff, yy, xx, rc, ry, rx])
            T.reads(pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx], vae_decoder_conv_out_weight[v_ff, v_rc, v_ry, v_rx])
            T.writes(var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx])
            with T.init():
                var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = T.float32(0)
            var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] + pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx] * vae_decoder_conv_out_weight[v_ff, v_rc, v_ry, v_rx]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(3), T.int64(1024), T.int64(1024)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], lv233[v_ax0, v_ax1, T.int64(0), T.int64(0)])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] + lv233[v_ax0, v_ax1, T.int64(0), T.int64(0)]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(3), T.int64(1024), T.int64(1024)):
        with T.block("T_divide"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            T.writes(var_T_divide_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_divide_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] * T.float32(0.5)
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(3), T.int64(1024), T.int64(1024)):
        with T.block("T_add_1"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_divide_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            T.writes(var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3] = var_T_divide_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] + T.float32(0.5)
    for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(3), T.int64(1024), T.int64(1024)):
        with T.block("compute"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(var_T_add_intermediate_1[v_i0, v_i1, v_i2, v_i3])
            T.writes(var_compute_intermediate[v_i0, v_i1, v_i2, v_i3])
            var_compute_intermediate[v_i0, v_i1, v_i2, v_i3] = T.max(T.min(var_T_add_intermediate_1[v_i0, v_i1, v_i2, v_i3], T.float32(1)), T.float32(0))

@T.prim_func
def fused_conv2d3_add12_add14(lv91: T.Buffer((T.int64(2), T.int64(320), T.int64(64), T.int64(64)), "float32"), unet_down_blocks_1_resnets_0_conv1_weight: T.Buffer((T.int64(640), T.int64(320), T.int64(3), T.int64(3)), "float32"), lv93: T.Buffer((T.int64(1), T.int64(640), T.int64(1), T.int64(1)), "float32"), lv100: T.Buffer((T.int64(2), T.int64(640), T.int64(1), T.int64(1)), "float32"), var_T_add_intermediate: T.Buffer((T.int64(2), T.int64(640), T.int64(64), T.int64(64)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    pad_temp = T.alloc_buffer((T.int64(2), T.int64(320), T.int64(66), T.int64(66)))
    var_conv2d_nchw_intermediate = T.alloc_buffer((T.int64(2), T.int64(640), T.int64(64), T.int64(64)))
    var_T_add_intermediate_1 = T.alloc_buffer((T.int64(2), T.int64(640), T.int64(64), T.int64(64)))
    for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(320), T.int64(66), T.int64(66)):
        with T.block("pad_temp"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(lv91[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)])
            T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
            pad_temp[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(T.int64(1) <= v_i2 and v_i2 < T.int64(65) and T.int64(1) <= v_i3 and v_i3 < T.int64(65), lv91[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)], T.float32(0))
    for nn, ff, yy, xx, rc, ry, rx in T.grid(T.int64(2), T.int64(640), T.int64(64), T.int64(64), T.int64(320), T.int64(3), T.int64(3)):
        with T.block("conv2d_nchw"):
            v_nn, v_ff, v_yy, v_xx, v_rc, v_ry, v_rx = T.axis.remap("SSSSRRR", [nn, ff, yy, xx, rc, ry, rx])
            T.reads(pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx], unet_down_blocks_1_resnets_0_conv1_weight[v_ff, v_rc, v_ry, v_rx])
            T.writes(var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx])
            with T.init():
                var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = T.float32(0)
            var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] + pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx] * unet_down_blocks_1_resnets_0_conv1_weight[v_ff, v_rc, v_ry, v_rx]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(640), T.int64(64), T.int64(64)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], lv93[T.int64(0), v_ax1, T.int64(0), T.int64(0)])
            T.writes(var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3] = var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] + lv93[T.int64(0), v_ax1, T.int64(0), T.int64(0)]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(640), T.int64(64), T.int64(64)):
        with T.block("T_add_1"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3], lv100[v_ax0, v_ax1, T.int64(0), T.int64(0)])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3] + lv100[v_ax0, v_ax1, T.int64(0), T.int64(0)]

@T.prim_func
def fused_conv2d4_add12_add14(lv259: T.Buffer((T.int64(2), T.int64(640), T.int64(64), T.int64(64)), "float32"), unet_down_blocks_1_resnets_1_conv1_weight: T.Buffer((T.int64(640), T.int64(640), T.int64(3), T.int64(3)), "float32"), lv261: T.Buffer((T.int64(1), T.int64(640), T.int64(1), T.int64(1)), "float32"), lv268: T.Buffer((T.int64(2), T.int64(640), T.int64(1), T.int64(1)), "float32"), var_T_add_intermediate: T.Buffer((T.int64(2), T.int64(640), T.int64(64), T.int64(64)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    pad_temp = T.alloc_buffer((T.int64(2), T.int64(640), T.int64(66), T.int64(66)))
    var_conv2d_nchw_intermediate = T.alloc_buffer((T.int64(2), T.int64(640), T.int64(64), T.int64(64)))
    var_T_add_intermediate_1 = T.alloc_buffer((T.int64(2), T.int64(640), T.int64(64), T.int64(64)))
    for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(640), T.int64(66), T.int64(66)):
        with T.block("pad_temp"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(lv259[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)])
            T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
            pad_temp[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(T.int64(1) <= v_i2 and v_i2 < T.int64(65) and T.int64(1) <= v_i3 and v_i3 < T.int64(65), lv259[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)], T.float32(0))
    for nn, ff, yy, xx, rc, ry, rx in T.grid(T.int64(2), T.int64(640), T.int64(64), T.int64(64), T.int64(640), T.int64(3), T.int64(3)):
        with T.block("conv2d_nchw"):
            v_nn, v_ff, v_yy, v_xx, v_rc, v_ry, v_rx = T.axis.remap("SSSSRRR", [nn, ff, yy, xx, rc, ry, rx])
            T.reads(pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx], unet_down_blocks_1_resnets_1_conv1_weight[v_ff, v_rc, v_ry, v_rx])
            T.writes(var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx])
            with T.init():
                var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = T.float32(0)
            var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] + pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx] * unet_down_blocks_1_resnets_1_conv1_weight[v_ff, v_rc, v_ry, v_rx]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(640), T.int64(64), T.int64(64)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], lv261[T.int64(0), v_ax1, T.int64(0), T.int64(0)])
            T.writes(var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3] = var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] + lv261[T.int64(0), v_ax1, T.int64(0), T.int64(0)]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(640), T.int64(64), T.int64(64)):
        with T.block("T_add_1"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3], lv268[v_ax0, v_ax1, T.int64(0), T.int64(0)])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3] + lv268[v_ax0, v_ax1, T.int64(0), T.int64(0)]

@T.prim_func
def fused_conv2d4_add12_add15_divide1(lv103: T.Buffer((T.int64(2), T.int64(640), T.int64(64), T.int64(64)), "float32"), unet_down_blocks_1_resnets_0_conv2_weight: T.Buffer((T.int64(640), T.int64(640), T.int64(3), T.int64(3)), "float32"), lv105: T.Buffer((T.int64(1), T.int64(640), T.int64(1), T.int64(1)), "float32"), lv109: T.Buffer((T.int64(2), T.int64(640), T.int64(64), T.int64(64)), "float32"), var_T_divide_intermediate: T.Buffer((T.int64(2), T.int64(640), T.int64(64), T.int64(64)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    pad_temp = T.alloc_buffer((T.int64(2), T.int64(640), T.int64(66), T.int64(66)))
    var_conv2d_nchw_intermediate = T.alloc_buffer((T.int64(2), T.int64(640), T.int64(64), T.int64(64)))
    var_T_add_intermediate = T.alloc_buffer((T.int64(2), T.int64(640), T.int64(64), T.int64(64)))
    var_T_add_intermediate_1 = T.alloc_buffer((T.int64(2), T.int64(640), T.int64(64), T.int64(64)))
    for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(640), T.int64(66), T.int64(66)):
        with T.block("pad_temp"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(lv103[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)])
            T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
            pad_temp[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(T.int64(1) <= v_i2 and v_i2 < T.int64(65) and T.int64(1) <= v_i3 and v_i3 < T.int64(65), lv103[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)], T.float32(0))
    for nn, ff, yy, xx, rc, ry, rx in T.grid(T.int64(2), T.int64(640), T.int64(64), T.int64(64), T.int64(640), T.int64(3), T.int64(3)):
        with T.block("conv2d_nchw"):
            v_nn, v_ff, v_yy, v_xx, v_rc, v_ry, v_rx = T.axis.remap("SSSSRRR", [nn, ff, yy, xx, rc, ry, rx])
            T.reads(pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx], unet_down_blocks_1_resnets_0_conv2_weight[v_ff, v_rc, v_ry, v_rx])
            T.writes(var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx])
            with T.init():
                var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = T.float32(0)
            var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] + pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx] * unet_down_blocks_1_resnets_0_conv2_weight[v_ff, v_rc, v_ry, v_rx]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(640), T.int64(64), T.int64(64)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], lv105[T.int64(0), v_ax1, T.int64(0), T.int64(0)])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] + lv105[T.int64(0), v_ax1, T.int64(0), T.int64(0)]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(640), T.int64(64), T.int64(64)):
        with T.block("T_add_1"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(lv109[v_ax0, v_ax1, v_ax2, v_ax3], var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            T.writes(var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3] = lv109[v_ax0, v_ax1, v_ax2, v_ax3] + var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(640), T.int64(64), T.int64(64)):
        with T.block("T_divide"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3])
            T.writes(var_T_divide_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_divide_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3]

@T.prim_func
def fused_conv2d5_add12(lv89: T.Buffer((T.int64(2), T.int64(320), T.int64(64), T.int64(64)), "float32"), unet_down_blocks_1_resnets_0_conv_shortcut_weight: T.Buffer((T.int64(640), T.int64(320), T.int64(1), T.int64(1)), "float32"), lv108: T.Buffer((T.int64(1), T.int64(640), T.int64(1), T.int64(1)), "float32"), var_T_add_intermediate: T.Buffer((T.int64(2), T.int64(640), T.int64(64), T.int64(64)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    pad_temp = T.alloc_buffer((T.int64(2), T.int64(320), T.int64(64), T.int64(64)))
    var_conv2d_nchw_intermediate = T.alloc_buffer((T.int64(2), T.int64(640), T.int64(64), T.int64(64)))
    for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(320), T.int64(64), T.int64(64)):
        with T.block("pad_temp"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(lv89[v_i0, v_i1, v_i2, v_i3])
            T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
            pad_temp[v_i0, v_i1, v_i2, v_i3] = lv89[v_i0, v_i1, v_i2, v_i3]
    for nn, ff, yy, xx, rc, ry, rx in T.grid(T.int64(2), T.int64(640), T.int64(64), T.int64(64), T.int64(320), T.int64(1), T.int64(1)):
        with T.block("conv2d_nchw"):
            v_nn, v_ff, v_yy, v_xx, v_rc, v_ry, v_rx = T.axis.remap("SSSSRRR", [nn, ff, yy, xx, rc, ry, rx])
            T.reads(pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx], unet_down_blocks_1_resnets_0_conv_shortcut_weight[v_ff, v_rc, v_ry, v_rx])
            T.writes(var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx])
            with T.init():
                var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = T.float32(0)
            var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] + pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx] * unet_down_blocks_1_resnets_0_conv_shortcut_weight[v_ff, v_rc, v_ry, v_rx]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(640), T.int64(64), T.int64(64)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], lv108[T.int64(0), v_ax1, T.int64(0), T.int64(0)])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] + lv108[T.int64(0), v_ax1, T.int64(0), T.int64(0)]

@T.prim_func
def fused_conv2d6_add19(lv422: T.Buffer((T.int64(2), T.int64(640), T.int64(64), T.int64(64)), "float32"), unet_down_blocks_1_downsamplers_0_conv_weight: T.Buffer((T.int64(640), T.int64(640), T.int64(3), T.int64(3)), "float32"), lv424: T.Buffer((T.int64(1), T.int64(640), T.int64(1), T.int64(1)), "float32"), var_T_add_intermediate: T.Buffer((T.int64(2), T.int64(640), T.int64(32), T.int64(32)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    pad_temp = T.alloc_buffer((T.int64(2), T.int64(640), T.int64(66), T.int64(66)))
    var_conv2d_nchw_intermediate = T.alloc_buffer((T.int64(2), T.int64(640), T.int64(32), T.int64(32)))
    for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(640), T.int64(66), T.int64(66)):
        with T.block("pad_temp"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(lv422[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)])
            T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
            pad_temp[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(T.int64(1) <= v_i2 and v_i2 < T.int64(65) and T.int64(1) <= v_i3 and v_i3 < T.int64(65), lv422[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)], T.float32(0))
    for nn, ff, yy, xx, rc, ry, rx in T.grid(T.int64(2), T.int64(640), T.int64(32), T.int64(32), T.int64(640), T.int64(3), T.int64(3)):
        with T.block("conv2d_nchw"):
            v_nn, v_ff, v_yy, v_xx, v_rc, v_ry, v_rx = T.axis.remap("SSSSRRR", [nn, ff, yy, xx, rc, ry, rx])
            T.reads(pad_temp[v_nn, v_rc, v_yy * T.int64(2) + v_ry, v_xx * T.int64(2) + v_rx], unet_down_blocks_1_downsamplers_0_conv_weight[v_ff, v_rc, v_ry, v_rx])
            T.writes(var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx])
            with T.init():
                var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = T.float32(0)
            var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] + pad_temp[v_nn, v_rc, v_yy * T.int64(2) + v_ry, v_xx * T.int64(2) + v_rx] * unet_down_blocks_1_downsamplers_0_conv_weight[v_ff, v_rc, v_ry, v_rx]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(640), T.int64(32), T.int64(32)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], lv424[T.int64(0), v_ax1, T.int64(0), T.int64(0)])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] + lv424[T.int64(0), v_ax1, T.int64(0), T.int64(0)]

@T.prim_func
def fused_conv2d7_add20_add21(lv427: T.Buffer((T.int64(2), T.int64(640), T.int64(32), T.int64(32)), "float32"), unet_down_blocks_2_resnets_0_conv1_weight: T.Buffer((T.int64(1280), T.int64(640), T.int64(3), T.int64(3)), "float32"), lv429: T.Buffer((T.int64(1), T.int64(1280), T.int64(1), T.int64(1)), "float32"), lv436: T.Buffer((T.int64(2), T.int64(1280), T.int64(1), T.int64(1)), "float32"), var_T_add_intermediate: T.Buffer((T.int64(2), T.int64(1280), T.int64(32), T.int64(32)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    pad_temp = T.alloc_buffer((T.int64(2), T.int64(640), T.int64(34), T.int64(34)))
    var_conv2d_nchw_intermediate = T.alloc_buffer((T.int64(2), T.int64(1280), T.int64(32), T.int64(32)))
    var_T_add_intermediate_1 = T.alloc_buffer((T.int64(2), T.int64(1280), T.int64(32), T.int64(32)))
    for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(640), T.int64(34), T.int64(34)):
        with T.block("pad_temp"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(lv427[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)])
            T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
            pad_temp[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(T.int64(1) <= v_i2 and v_i2 < T.int64(33) and T.int64(1) <= v_i3 and v_i3 < T.int64(33), lv427[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)], T.float32(0))
    for nn, ff, yy, xx, rc, ry, rx in T.grid(T.int64(2), T.int64(1280), T.int64(32), T.int64(32), T.int64(640), T.int64(3), T.int64(3)):
        with T.block("conv2d_nchw"):
            v_nn, v_ff, v_yy, v_xx, v_rc, v_ry, v_rx = T.axis.remap("SSSSRRR", [nn, ff, yy, xx, rc, ry, rx])
            T.reads(pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx], unet_down_blocks_2_resnets_0_conv1_weight[v_ff, v_rc, v_ry, v_rx])
            T.writes(var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx])
            with T.init():
                var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = T.float32(0)
            var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] + pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx] * unet_down_blocks_2_resnets_0_conv1_weight[v_ff, v_rc, v_ry, v_rx]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(1280), T.int64(32), T.int64(32)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], lv429[T.int64(0), v_ax1, T.int64(0), T.int64(0)])
            T.writes(var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3] = var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] + lv429[T.int64(0), v_ax1, T.int64(0), T.int64(0)]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(1280), T.int64(32), T.int64(32)):
        with T.block("T_add_1"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3], lv436[v_ax0, v_ax1, T.int64(0), T.int64(0)])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3] + lv436[v_ax0, v_ax1, T.int64(0), T.int64(0)]

@T.prim_func
def fused_conv2d8_add20_add21(lv1131: T.Buffer((T.int64(2), T.int64(1280), T.int64(32), T.int64(32)), "float32"), unet_down_blocks_2_resnets_1_conv1_weight: T.Buffer((T.int64(1280), T.int64(1280), T.int64(3), T.int64(3)), "float32"), lv1133: T.Buffer((T.int64(1), T.int64(1280), T.int64(1), T.int64(1)), "float32"), lv1140: T.Buffer((T.int64(2), T.int64(1280), T.int64(1), T.int64(1)), "float32"), var_T_add_intermediate: T.Buffer((T.int64(2), T.int64(1280), T.int64(32), T.int64(32)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    pad_temp = T.alloc_buffer((T.int64(2), T.int64(1280), T.int64(34), T.int64(34)))
    var_conv2d_nchw_intermediate = T.alloc_buffer((T.int64(2), T.int64(1280), T.int64(32), T.int64(32)))
    var_T_add_intermediate_1 = T.alloc_buffer((T.int64(2), T.int64(1280), T.int64(32), T.int64(32)))
    for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(1280), T.int64(34), T.int64(34)):
        with T.block("pad_temp"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(lv1131[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)])
            T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
            pad_temp[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(T.int64(1) <= v_i2 and v_i2 < T.int64(33) and T.int64(1) <= v_i3 and v_i3 < T.int64(33), lv1131[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)], T.float32(0))
    for nn, ff, yy, xx, rc, ry, rx in T.grid(T.int64(2), T.int64(1280), T.int64(32), T.int64(32), T.int64(1280), T.int64(3), T.int64(3)):
        with T.block("conv2d_nchw"):
            v_nn, v_ff, v_yy, v_xx, v_rc, v_ry, v_rx = T.axis.remap("SSSSRRR", [nn, ff, yy, xx, rc, ry, rx])
            T.reads(pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx], unet_down_blocks_2_resnets_1_conv1_weight[v_ff, v_rc, v_ry, v_rx])
            T.writes(var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx])
            with T.init():
                var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = T.float32(0)
            var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] + pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx] * unet_down_blocks_2_resnets_1_conv1_weight[v_ff, v_rc, v_ry, v_rx]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(1280), T.int64(32), T.int64(32)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], lv1133[T.int64(0), v_ax1, T.int64(0), T.int64(0)])
            T.writes(var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3] = var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] + lv1133[T.int64(0), v_ax1, T.int64(0), T.int64(0)]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(1280), T.int64(32), T.int64(32)):
        with T.block("T_add_1"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3], lv1140[v_ax0, v_ax1, T.int64(0), T.int64(0)])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3] + lv1140[v_ax0, v_ax1, T.int64(0), T.int64(0)]

@T.prim_func
def fused_conv2d8_add20_add22_divide4(lv439: T.Buffer((T.int64(2), T.int64(1280), T.int64(32), T.int64(32)), "float32"), unet_down_blocks_2_resnets_0_conv2_weight: T.Buffer((T.int64(1280), T.int64(1280), T.int64(3), T.int64(3)), "float32"), lv441: T.Buffer((T.int64(1), T.int64(1280), T.int64(1), T.int64(1)), "float32"), lv445: T.Buffer((T.int64(2), T.int64(1280), T.int64(32), T.int64(32)), "float32"), var_T_divide_intermediate: T.Buffer((T.int64(2), T.int64(1280), T.int64(32), T.int64(32)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    pad_temp = T.alloc_buffer((T.int64(2), T.int64(1280), T.int64(34), T.int64(34)))
    var_conv2d_nchw_intermediate = T.alloc_buffer((T.int64(2), T.int64(1280), T.int64(32), T.int64(32)))
    var_T_add_intermediate = T.alloc_buffer((T.int64(2), T.int64(1280), T.int64(32), T.int64(32)))
    var_T_add_intermediate_1 = T.alloc_buffer((T.int64(2), T.int64(1280), T.int64(32), T.int64(32)))
    for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(1280), T.int64(34), T.int64(34)):
        with T.block("pad_temp"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(lv439[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)])
            T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
            pad_temp[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(T.int64(1) <= v_i2 and v_i2 < T.int64(33) and T.int64(1) <= v_i3 and v_i3 < T.int64(33), lv439[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)], T.float32(0))
    for nn, ff, yy, xx, rc, ry, rx in T.grid(T.int64(2), T.int64(1280), T.int64(32), T.int64(32), T.int64(1280), T.int64(3), T.int64(3)):
        with T.block("conv2d_nchw"):
            v_nn, v_ff, v_yy, v_xx, v_rc, v_ry, v_rx = T.axis.remap("SSSSRRR", [nn, ff, yy, xx, rc, ry, rx])
            T.reads(pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx], unet_down_blocks_2_resnets_0_conv2_weight[v_ff, v_rc, v_ry, v_rx])
            T.writes(var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx])
            with T.init():
                var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = T.float32(0)
            var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] + pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx] * unet_down_blocks_2_resnets_0_conv2_weight[v_ff, v_rc, v_ry, v_rx]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(1280), T.int64(32), T.int64(32)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], lv441[T.int64(0), v_ax1, T.int64(0), T.int64(0)])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] + lv441[T.int64(0), v_ax1, T.int64(0), T.int64(0)]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(1280), T.int64(32), T.int64(32)):
        with T.block("T_add_1"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(lv445[v_ax0, v_ax1, v_ax2, v_ax3], var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            T.writes(var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3] = lv445[v_ax0, v_ax1, v_ax2, v_ax3] + var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(1280), T.int64(32), T.int64(32)):
        with T.block("T_divide"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3])
            T.writes(var_T_divide_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_divide_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3]

@T.prim_func
def fused_conv2d9_add20(lv425: T.Buffer((T.int64(2), T.int64(640), T.int64(32), T.int64(32)), "float32"), unet_down_blocks_2_resnets_0_conv_shortcut_weight: T.Buffer((T.int64(1280), T.int64(640), T.int64(1), T.int64(1)), "float32"), lv444: T.Buffer((T.int64(1), T.int64(1280), T.int64(1), T.int64(1)), "float32"), var_T_add_intermediate: T.Buffer((T.int64(2), T.int64(1280), T.int64(32), T.int64(32)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    pad_temp = T.alloc_buffer((T.int64(2), T.int64(640), T.int64(32), T.int64(32)))
    var_conv2d_nchw_intermediate = T.alloc_buffer((T.int64(2), T.int64(1280), T.int64(32), T.int64(32)))
    for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(640), T.int64(32), T.int64(32)):
        with T.block("pad_temp"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(lv425[v_i0, v_i1, v_i2, v_i3])
            T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
            pad_temp[v_i0, v_i1, v_i2, v_i3] = lv425[v_i0, v_i1, v_i2, v_i3]
    for nn, ff, yy, xx, rc, ry, rx in T.grid(T.int64(2), T.int64(1280), T.int64(32), T.int64(32), T.int64(640), T.int64(1), T.int64(1)):
        with T.block("conv2d_nchw"):
            v_nn, v_ff, v_yy, v_xx, v_rc, v_ry, v_rx = T.axis.remap("SSSSRRR", [nn, ff, yy, xx, rc, ry, rx])
            T.reads(pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx], unet_down_blocks_2_resnets_0_conv_shortcut_weight[v_ff, v_rc, v_ry, v_rx])
            T.writes(var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx])
            with T.init():
                var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = T.float32(0)
            var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] + pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx] * unet_down_blocks_2_resnets_0_conv_shortcut_weight[v_ff, v_rc, v_ry, v_rx]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(1280), T.int64(32), T.int64(32)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], lv444[T.int64(0), v_ax1, T.int64(0), T.int64(0)])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] + lv444[T.int64(0), v_ax1, T.int64(0), T.int64(0)]

@T.prim_func
def fused_conv2d_add7(inp_0: T.Buffer((T.int64(2), T.int64(4), T.int64(128), T.int64(128)), "float32"), unet_conv_in_weight: T.Buffer((T.int64(320), T.int64(4), T.int64(3), T.int64(3)), "float32"), lv47: T.Buffer((T.int64(1), T.int64(320), T.int64(1), T.int64(1)), "float32"), var_T_add_intermediate: T.Buffer((T.int64(2), T.int64(320), T.int64(128), T.int64(128)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    pad_temp = T.alloc_buffer((T.int64(2), T.int64(4), T.int64(130), T.int64(130)))
    var_conv2d_nchw_intermediate = T.alloc_buffer((T.int64(2), T.int64(320), T.int64(128), T.int64(128)))
    for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(4), T.int64(130), T.int64(130)):
        with T.block("pad_temp"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(inp_0[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)])
            T.writes(pad_temp[v_i0, v_i1, v_i2, v_i3])
            pad_temp[v_i0, v_i1, v_i2, v_i3] = T.if_then_else(T.int64(1) <= v_i2 and v_i2 < T.int64(129) and T.int64(1) <= v_i3 and v_i3 < T.int64(129), inp_0[v_i0, v_i1, v_i2 - T.int64(1), v_i3 - T.int64(1)], T.float32(0))
    for nn, ff, yy, xx, rc, ry, rx in T.grid(T.int64(2), T.int64(320), T.int64(128), T.int64(128), T.int64(4), T.int64(3), T.int64(3)):
        with T.block("conv2d_nchw"):
            v_nn, v_ff, v_yy, v_xx, v_rc, v_ry, v_rx = T.axis.remap("SSSSRRR", [nn, ff, yy, xx, rc, ry, rx])
            T.reads(pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx], unet_conv_in_weight[v_ff, v_rc, v_ry, v_rx])
            T.writes(var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx])
            with T.init():
                var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = T.float32(0)
            var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] = var_conv2d_nchw_intermediate[v_nn, v_ff, v_yy, v_xx] + pad_temp[v_nn, v_rc, v_yy + v_ry, v_xx + v_rx] * unet_conv_in_weight[v_ff, v_rc, v_ry, v_rx]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(320), T.int64(128), T.int64(128)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], lv47[T.int64(0), v_ax1, T.int64(0), T.int64(0)])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_conv2d_nchw_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] + lv47[T.int64(0), v_ax1, T.int64(0), T.int64(0)]

@T.prim_func
def fused_group_norm10_silu9(lv4839: T.Buffer((T.int64(2), T.int64(1280), T.int64(64), T.int64(64)), "float32"), unet_up_blocks_1_resnets_1_norm1_weight: T.Buffer((T.int64(1280),), "float32"), unet_up_blocks_1_resnets_1_norm1_bias: T.Buffer((T.int64(1280),), "float32"), var_T_multiply_intermediate: T.Buffer((T.int64(2), T.int64(1280), T.int64(64), T.int64(64)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    T_reshape = T.alloc_buffer((T.int64(2), T.int64(32), T.int64(40), T.int64(64), T.int64(64)))
    A_red_temp_v0 = T.alloc_buffer((T.int64(2), T.int64(32)))
    A_red_temp_v1 = T.alloc_buffer((T.int64(2), T.int64(32)))
    T_reshape_1 = T.alloc_buffer((T.int64(32), T.int64(40)))
    T_reshape_2 = T.alloc_buffer((T.int64(32), T.int64(40)))
    T_group_norm = T.alloc_buffer((T.int64(2), T.int64(32), T.int64(40), T.int64(64), T.int64(64)))
    var_T_reshape_intermediate = T.alloc_buffer((T.int64(2), T.int64(1280), T.int64(64), T.int64(64)))
    compute = T.alloc_buffer((T.int64(2), T.int64(1280), T.int64(64), T.int64(64)))
    for ax0, ax1, ax2, ax3, ax4 in T.grid(T.int64(2), T.int64(32), T.int64(40), T.int64(64), T.int64(64)):
        with T.block("T_reshape"):
            v_ax0, v_ax1, v_ax2, v_ax3, v_ax4 = T.axis.remap("SSSSS", [ax0, ax1, ax2, ax3, ax4])
            T.reads(lv4839[((v_ax1 * T.int64(40) + (v_ax4 // T.int64(64) + v_ax3) // T.int64(64) + v_ax2) // T.int64(1280) + v_ax0) % T.int64(2), (v_ax1 * T.int64(40) + (v_ax4 // T.int64(64) + v_ax3) // T.int64(64) + v_ax2) % T.int64(1280), (v_ax4 // T.int64(64) + v_ax3) % T.int64(64), v_ax4 % T.int64(64)])
            T.writes(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4])
            T_reshape[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] = lv4839[((v_ax1 * T.int64(40) + (v_ax4 // T.int64(64) + v_ax3) // T.int64(64) + v_ax2) // T.int64(1280) + v_ax0) % T.int64(2), (v_ax1 * T.int64(40) + (v_ax4 // T.int64(64) + v_ax3) // T.int64(64) + v_ax2) % T.int64(1280), (v_ax4 // T.int64(64) + v_ax3) % T.int64(64), v_ax4 % T.int64(64)]
    for ax0, ax1, k2, k3, k4 in T.grid(T.int64(2), T.int64(32), T.int64(40), T.int64(64), T.int64(64)):
        with T.block("A_red_temp"):
            v_ax0, v_ax1, v_k2, v_k3, v_k4 = T.axis.remap("SSRRR", [ax0, ax1, k2, k3, k4])
            T.reads(T_reshape[v_ax0, v_ax1, v_k2, v_k3, v_k4])
            T.writes(A_red_temp_v0[v_ax0, v_ax1], A_red_temp_v1[v_ax0, v_ax1])
            with T.init():
                A_red_temp_v0[v_ax0, v_ax1] = T.float32(0)
                A_red_temp_v1[v_ax0, v_ax1] = T.float32(0)
            v_A_red_temp_v0: T.float32 = A_red_temp_v0[v_ax0, v_ax1] + T_reshape[v_ax0, v_ax1, v_k2, v_k3, v_k4]
            v_A_red_temp_v1: T.float32 = A_red_temp_v1[v_ax0, v_ax1] + T_reshape[v_ax0, v_ax1, v_k2, v_k3, v_k4] * T_reshape[v_ax0, v_ax1, v_k2, v_k3, v_k4]
            A_red_temp_v0[v_ax0, v_ax1] = v_A_red_temp_v0
            A_red_temp_v1[v_ax0, v_ax1] = v_A_red_temp_v1
    for ax0, ax1 in T.grid(T.int64(32), T.int64(40)):
        with T.block("T_reshape_1"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(unet_up_blocks_1_resnets_1_norm1_weight[(v_ax0 * T.int64(40) + v_ax1) % T.int64(1280)])
            T.writes(T_reshape_1[v_ax0, v_ax1])
            T_reshape_1[v_ax0, v_ax1] = unet_up_blocks_1_resnets_1_norm1_weight[(v_ax0 * T.int64(40) + v_ax1) % T.int64(1280)]
    for ax0, ax1 in T.grid(T.int64(32), T.int64(40)):
        with T.block("T_reshape_2"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(unet_up_blocks_1_resnets_1_norm1_bias[(v_ax0 * T.int64(40) + v_ax1) % T.int64(1280)])
            T.writes(T_reshape_2[v_ax0, v_ax1])
            T_reshape_2[v_ax0, v_ax1] = unet_up_blocks_1_resnets_1_norm1_bias[(v_ax0 * T.int64(40) + v_ax1) % T.int64(1280)]
    for ax0, ax1, ax2, ax3, ax4 in T.grid(T.int64(2), T.int64(32), T.int64(40), T.int64(64), T.int64(64)):
        with T.block("T_group_norm"):
            v_ax0, v_ax1, v_ax2, v_ax3, v_ax4 = T.axis.remap("SSSSS", [ax0, ax1, ax2, ax3, ax4])
            T.reads(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4], A_red_temp_v0[v_ax0, v_ax1], A_red_temp_v1[v_ax0, v_ax1], T_reshape_1[v_ax1, v_ax2], T_reshape_2[v_ax1, v_ax2])
            T.writes(T_group_norm[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4])
            T_group_norm[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] = (T_reshape[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] - A_red_temp_v0[v_ax0, v_ax1] * T.float32(6.1035156250000003e-06)) * T.rsqrt(A_red_temp_v1[v_ax0, v_ax1] * T.float32(6.1035156250000003e-06) - A_red_temp_v0[v_ax0, v_ax1] * T.float32(6.1035156250000003e-06) * (A_red_temp_v0[v_ax0, v_ax1] * T.float32(6.1035156250000003e-06)) + T.float32(1.0000000000000001e-05)) * T_reshape_1[v_ax1, v_ax2] + T_reshape_2[v_ax1, v_ax2]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(1280), T.int64(64), T.int64(64)):
        with T.block("T_reshape_3"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(T_group_norm[(((v_ax3 // T.int64(64) + v_ax2) // T.int64(64) + v_ax1) // T.int64(1280) + v_ax0) % T.int64(2), ((v_ax3 // T.int64(64) + v_ax2) // T.int64(64) + v_ax1) % T.int64(1280) // T.int64(40), ((v_ax3 // T.int64(64) + v_ax2) // T.int64(64) + v_ax1) % T.int64(40), (v_ax3 // T.int64(64) + v_ax2) % T.int64(64), v_ax3 % T.int64(64)])
            T.writes(var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = T_group_norm[(((v_ax3 // T.int64(64) + v_ax2) // T.int64(64) + v_ax1) // T.int64(1280) + v_ax0) % T.int64(2), ((v_ax3 // T.int64(64) + v_ax2) // T.int64(64) + v_ax1) % T.int64(1280) // T.int64(40), ((v_ax3 // T.int64(64) + v_ax2) // T.int64(64) + v_ax1) % T.int64(40), (v_ax3 // T.int64(64) + v_ax2) % T.int64(64), v_ax3 % T.int64(64)]
    for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(1280), T.int64(64), T.int64(64)):
        with T.block("compute"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(var_T_reshape_intermediate[v_i0, v_i1, v_i2, v_i3])
            T.writes(compute[v_i0, v_i1, v_i2, v_i3])
            compute[v_i0, v_i1, v_i2, v_i3] = T.sigmoid(var_T_reshape_intermediate[v_i0, v_i1, v_i2, v_i3])
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(1280), T.int64(64), T.int64(64)):
        with T.block("T_multiply"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], compute[v_ax0, v_ax1, v_ax2, v_ax3])
            T.writes(var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] * compute[v_ax0, v_ax1, v_ax2, v_ax3]

@T.prim_func
def fused_group_norm11_silu10(lv5008: T.Buffer((T.int64(2), T.int64(960), T.int64(64), T.int64(64)), "float32"), unet_up_blocks_1_resnets_2_norm1_weight: T.Buffer((T.int64(960),), "float32"), unet_up_blocks_1_resnets_2_norm1_bias: T.Buffer((T.int64(960),), "float32"), var_T_multiply_intermediate: T.Buffer((T.int64(2), T.int64(960), T.int64(64), T.int64(64)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    T_reshape = T.alloc_buffer((T.int64(2), T.int64(32), T.int64(30), T.int64(64), T.int64(64)))
    A_red_temp_v0 = T.alloc_buffer((T.int64(2), T.int64(32)))
    A_red_temp_v1 = T.alloc_buffer((T.int64(2), T.int64(32)))
    T_reshape_1 = T.alloc_buffer((T.int64(32), T.int64(30)))
    T_reshape_2 = T.alloc_buffer((T.int64(32), T.int64(30)))
    T_group_norm = T.alloc_buffer((T.int64(2), T.int64(32), T.int64(30), T.int64(64), T.int64(64)))
    var_T_reshape_intermediate = T.alloc_buffer((T.int64(2), T.int64(960), T.int64(64), T.int64(64)))
    compute = T.alloc_buffer((T.int64(2), T.int64(960), T.int64(64), T.int64(64)))
    for ax0, ax1, ax2, ax3, ax4 in T.grid(T.int64(2), T.int64(32), T.int64(30), T.int64(64), T.int64(64)):
        with T.block("T_reshape"):
            v_ax0, v_ax1, v_ax2, v_ax3, v_ax4 = T.axis.remap("SSSSS", [ax0, ax1, ax2, ax3, ax4])
            T.reads(lv5008[((v_ax1 * T.int64(30) + (v_ax4 // T.int64(64) + v_ax3) // T.int64(64) + v_ax2) // T.int64(960) + v_ax0) % T.int64(2), (v_ax1 * T.int64(30) + (v_ax4 // T.int64(64) + v_ax3) // T.int64(64) + v_ax2) % T.int64(960), (v_ax4 // T.int64(64) + v_ax3) % T.int64(64), v_ax4 % T.int64(64)])
            T.writes(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4])
            T_reshape[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] = lv5008[((v_ax1 * T.int64(30) + (v_ax4 // T.int64(64) + v_ax3) // T.int64(64) + v_ax2) // T.int64(960) + v_ax0) % T.int64(2), (v_ax1 * T.int64(30) + (v_ax4 // T.int64(64) + v_ax3) // T.int64(64) + v_ax2) % T.int64(960), (v_ax4 // T.int64(64) + v_ax3) % T.int64(64), v_ax4 % T.int64(64)]
    for ax0, ax1, k2, k3, k4 in T.grid(T.int64(2), T.int64(32), T.int64(30), T.int64(64), T.int64(64)):
        with T.block("A_red_temp"):
            v_ax0, v_ax1, v_k2, v_k3, v_k4 = T.axis.remap("SSRRR", [ax0, ax1, k2, k3, k4])
            T.reads(T_reshape[v_ax0, v_ax1, v_k2, v_k3, v_k4])
            T.writes(A_red_temp_v0[v_ax0, v_ax1], A_red_temp_v1[v_ax0, v_ax1])
            with T.init():
                A_red_temp_v0[v_ax0, v_ax1] = T.float32(0)
                A_red_temp_v1[v_ax0, v_ax1] = T.float32(0)
            v_A_red_temp_v0: T.float32 = A_red_temp_v0[v_ax0, v_ax1] + T_reshape[v_ax0, v_ax1, v_k2, v_k3, v_k4]
            v_A_red_temp_v1: T.float32 = A_red_temp_v1[v_ax0, v_ax1] + T_reshape[v_ax0, v_ax1, v_k2, v_k3, v_k4] * T_reshape[v_ax0, v_ax1, v_k2, v_k3, v_k4]
            A_red_temp_v0[v_ax0, v_ax1] = v_A_red_temp_v0
            A_red_temp_v1[v_ax0, v_ax1] = v_A_red_temp_v1
    for ax0, ax1 in T.grid(T.int64(32), T.int64(30)):
        with T.block("T_reshape_1"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(unet_up_blocks_1_resnets_2_norm1_weight[(v_ax0 * T.int64(30) + v_ax1) % T.int64(960)])
            T.writes(T_reshape_1[v_ax0, v_ax1])
            T_reshape_1[v_ax0, v_ax1] = unet_up_blocks_1_resnets_2_norm1_weight[(v_ax0 * T.int64(30) + v_ax1) % T.int64(960)]
    for ax0, ax1 in T.grid(T.int64(32), T.int64(30)):
        with T.block("T_reshape_2"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(unet_up_blocks_1_resnets_2_norm1_bias[(v_ax0 * T.int64(30) + v_ax1) % T.int64(960)])
            T.writes(T_reshape_2[v_ax0, v_ax1])
            T_reshape_2[v_ax0, v_ax1] = unet_up_blocks_1_resnets_2_norm1_bias[(v_ax0 * T.int64(30) + v_ax1) % T.int64(960)]
    for ax0, ax1, ax2, ax3, ax4 in T.grid(T.int64(2), T.int64(32), T.int64(30), T.int64(64), T.int64(64)):
        with T.block("T_group_norm"):
            v_ax0, v_ax1, v_ax2, v_ax3, v_ax4 = T.axis.remap("SSSSS", [ax0, ax1, ax2, ax3, ax4])
            T.reads(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4], A_red_temp_v0[v_ax0, v_ax1], A_red_temp_v1[v_ax0, v_ax1], T_reshape_1[v_ax1, v_ax2], T_reshape_2[v_ax1, v_ax2])
            T.writes(T_group_norm[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4])
            T_group_norm[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] = (T_reshape[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] - A_red_temp_v0[v_ax0, v_ax1] * T.float32(8.1380208333333332e-06)) * T.rsqrt(A_red_temp_v1[v_ax0, v_ax1] * T.float32(8.1380208333333332e-06) - A_red_temp_v0[v_ax0, v_ax1] * T.float32(8.1380208333333332e-06) * (A_red_temp_v0[v_ax0, v_ax1] * T.float32(8.1380208333333332e-06)) + T.float32(1.0000000000000001e-05)) * T_reshape_1[v_ax1, v_ax2] + T_reshape_2[v_ax1, v_ax2]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(960), T.int64(64), T.int64(64)):
        with T.block("T_reshape_3"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(T_group_norm[(((v_ax3 // T.int64(64) + v_ax2) // T.int64(64) + v_ax1) // T.int64(960) + v_ax0) % T.int64(2), ((v_ax3 // T.int64(64) + v_ax2) // T.int64(64) + v_ax1) % T.int64(960) // T.int64(30), ((v_ax3 // T.int64(64) + v_ax2) // T.int64(64) + v_ax1) % T.int64(30), (v_ax3 // T.int64(64) + v_ax2) % T.int64(64), v_ax3 % T.int64(64)])
            T.writes(var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = T_group_norm[(((v_ax3 // T.int64(64) + v_ax2) // T.int64(64) + v_ax1) // T.int64(960) + v_ax0) % T.int64(2), ((v_ax3 // T.int64(64) + v_ax2) // T.int64(64) + v_ax1) % T.int64(960) // T.int64(30), ((v_ax3 // T.int64(64) + v_ax2) // T.int64(64) + v_ax1) % T.int64(30), (v_ax3 // T.int64(64) + v_ax2) % T.int64(64), v_ax3 % T.int64(64)]
    for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(960), T.int64(64), T.int64(64)):
        with T.block("compute"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(var_T_reshape_intermediate[v_i0, v_i1, v_i2, v_i3])
            T.writes(compute[v_i0, v_i1, v_i2, v_i3])
            compute[v_i0, v_i1, v_i2, v_i3] = T.sigmoid(var_T_reshape_intermediate[v_i0, v_i1, v_i2, v_i3])
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(960), T.int64(64), T.int64(64)):
        with T.block("T_multiply"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], compute[v_ax0, v_ax1, v_ax2, v_ax3])
            T.writes(var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] * compute[v_ax0, v_ax1, v_ax2, v_ax3]

@T.prim_func
def fused_group_norm12_silu11(lv5181: T.Buffer((T.int64(2), T.int64(960), T.int64(128), T.int64(128)), "float32"), unet_up_blocks_2_resnets_0_norm1_weight: T.Buffer((T.int64(960),), "float32"), unet_up_blocks_2_resnets_0_norm1_bias: T.Buffer((T.int64(960),), "float32"), var_T_multiply_intermediate: T.Buffer((T.int64(2), T.int64(960), T.int64(128), T.int64(128)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    T_reshape = T.alloc_buffer((T.int64(2), T.int64(32), T.int64(30), T.int64(128), T.int64(128)))
    A_red_temp_v0 = T.alloc_buffer((T.int64(2), T.int64(32)))
    A_red_temp_v1 = T.alloc_buffer((T.int64(2), T.int64(32)))
    T_reshape_1 = T.alloc_buffer((T.int64(32), T.int64(30)))
    T_reshape_2 = T.alloc_buffer((T.int64(32), T.int64(30)))
    T_group_norm = T.alloc_buffer((T.int64(2), T.int64(32), T.int64(30), T.int64(128), T.int64(128)))
    var_T_reshape_intermediate = T.alloc_buffer((T.int64(2), T.int64(960), T.int64(128), T.int64(128)))
    compute = T.alloc_buffer((T.int64(2), T.int64(960), T.int64(128), T.int64(128)))
    for ax0, ax1, ax2, ax3, ax4 in T.grid(T.int64(2), T.int64(32), T.int64(30), T.int64(128), T.int64(128)):
        with T.block("T_reshape"):
            v_ax0, v_ax1, v_ax2, v_ax3, v_ax4 = T.axis.remap("SSSSS", [ax0, ax1, ax2, ax3, ax4])
            T.reads(lv5181[((v_ax1 * T.int64(30) + (v_ax4 // T.int64(128) + v_ax3) // T.int64(128) + v_ax2) // T.int64(960) + v_ax0) % T.int64(2), (v_ax1 * T.int64(30) + (v_ax4 // T.int64(128) + v_ax3) // T.int64(128) + v_ax2) % T.int64(960), (v_ax4 // T.int64(128) + v_ax3) % T.int64(128), v_ax4 % T.int64(128)])
            T.writes(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4])
            T_reshape[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] = lv5181[((v_ax1 * T.int64(30) + (v_ax4 // T.int64(128) + v_ax3) // T.int64(128) + v_ax2) // T.int64(960) + v_ax0) % T.int64(2), (v_ax1 * T.int64(30) + (v_ax4 // T.int64(128) + v_ax3) // T.int64(128) + v_ax2) % T.int64(960), (v_ax4 // T.int64(128) + v_ax3) % T.int64(128), v_ax4 % T.int64(128)]
    for ax0, ax1, k2, k3, k4 in T.grid(T.int64(2), T.int64(32), T.int64(30), T.int64(128), T.int64(128)):
        with T.block("A_red_temp"):
            v_ax0, v_ax1, v_k2, v_k3, v_k4 = T.axis.remap("SSRRR", [ax0, ax1, k2, k3, k4])
            T.reads(T_reshape[v_ax0, v_ax1, v_k2, v_k3, v_k4])
            T.writes(A_red_temp_v0[v_ax0, v_ax1], A_red_temp_v1[v_ax0, v_ax1])
            with T.init():
                A_red_temp_v0[v_ax0, v_ax1] = T.float32(0)
                A_red_temp_v1[v_ax0, v_ax1] = T.float32(0)
            v_A_red_temp_v0: T.float32 = A_red_temp_v0[v_ax0, v_ax1] + T_reshape[v_ax0, v_ax1, v_k2, v_k3, v_k4]
            v_A_red_temp_v1: T.float32 = A_red_temp_v1[v_ax0, v_ax1] + T_reshape[v_ax0, v_ax1, v_k2, v_k3, v_k4] * T_reshape[v_ax0, v_ax1, v_k2, v_k3, v_k4]
            A_red_temp_v0[v_ax0, v_ax1] = v_A_red_temp_v0
            A_red_temp_v1[v_ax0, v_ax1] = v_A_red_temp_v1
    for ax0, ax1 in T.grid(T.int64(32), T.int64(30)):
        with T.block("T_reshape_1"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(unet_up_blocks_2_resnets_0_norm1_weight[(v_ax0 * T.int64(30) + v_ax1) % T.int64(960)])
            T.writes(T_reshape_1[v_ax0, v_ax1])
            T_reshape_1[v_ax0, v_ax1] = unet_up_blocks_2_resnets_0_norm1_weight[(v_ax0 * T.int64(30) + v_ax1) % T.int64(960)]
    for ax0, ax1 in T.grid(T.int64(32), T.int64(30)):
        with T.block("T_reshape_2"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(unet_up_blocks_2_resnets_0_norm1_bias[(v_ax0 * T.int64(30) + v_ax1) % T.int64(960)])
            T.writes(T_reshape_2[v_ax0, v_ax1])
            T_reshape_2[v_ax0, v_ax1] = unet_up_blocks_2_resnets_0_norm1_bias[(v_ax0 * T.int64(30) + v_ax1) % T.int64(960)]
    for ax0, ax1, ax2, ax3, ax4 in T.grid(T.int64(2), T.int64(32), T.int64(30), T.int64(128), T.int64(128)):
        with T.block("T_group_norm"):
            v_ax0, v_ax1, v_ax2, v_ax3, v_ax4 = T.axis.remap("SSSSS", [ax0, ax1, ax2, ax3, ax4])
            T.reads(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4], A_red_temp_v0[v_ax0, v_ax1], A_red_temp_v1[v_ax0, v_ax1], T_reshape_1[v_ax1, v_ax2], T_reshape_2[v_ax1, v_ax2])
            T.writes(T_group_norm[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4])
            T_group_norm[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] = (T_reshape[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] - A_red_temp_v0[v_ax0, v_ax1] * T.float32(2.0345052083333333e-06)) * T.rsqrt(A_red_temp_v1[v_ax0, v_ax1] * T.float32(2.0345052083333333e-06) - A_red_temp_v0[v_ax0, v_ax1] * T.float32(2.0345052083333333e-06) * (A_red_temp_v0[v_ax0, v_ax1] * T.float32(2.0345052083333333e-06)) + T.float32(1.0000000000000001e-05)) * T_reshape_1[v_ax1, v_ax2] + T_reshape_2[v_ax1, v_ax2]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(960), T.int64(128), T.int64(128)):
        with T.block("T_reshape_3"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(T_group_norm[(((v_ax3 // T.int64(128) + v_ax2) // T.int64(128) + v_ax1) // T.int64(960) + v_ax0) % T.int64(2), ((v_ax3 // T.int64(128) + v_ax2) // T.int64(128) + v_ax1) % T.int64(960) // T.int64(30), ((v_ax3 // T.int64(128) + v_ax2) // T.int64(128) + v_ax1) % T.int64(30), (v_ax3 // T.int64(128) + v_ax2) % T.int64(128), v_ax3 % T.int64(128)])
            T.writes(var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = T_group_norm[(((v_ax3 // T.int64(128) + v_ax2) // T.int64(128) + v_ax1) // T.int64(960) + v_ax0) % T.int64(2), ((v_ax3 // T.int64(128) + v_ax2) // T.int64(128) + v_ax1) % T.int64(960) // T.int64(30), ((v_ax3 // T.int64(128) + v_ax2) // T.int64(128) + v_ax1) % T.int64(30), (v_ax3 // T.int64(128) + v_ax2) % T.int64(128), v_ax3 % T.int64(128)]
    for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(960), T.int64(128), T.int64(128)):
        with T.block("compute"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(var_T_reshape_intermediate[v_i0, v_i1, v_i2, v_i3])
            T.writes(compute[v_i0, v_i1, v_i2, v_i3])
            compute[v_i0, v_i1, v_i2, v_i3] = T.sigmoid(var_T_reshape_intermediate[v_i0, v_i1, v_i2, v_i3])
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(960), T.int64(128), T.int64(128)):
        with T.block("T_multiply"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], compute[v_ax0, v_ax1, v_ax2, v_ax3])
            T.writes(var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] * compute[v_ax0, v_ax1, v_ax2, v_ax3]

@T.prim_func
def fused_group_norm13_silu12(lv5204: T.Buffer((T.int64(2), T.int64(640), T.int64(128), T.int64(128)), "float32"), unet_up_blocks_2_resnets_1_norm1_weight: T.Buffer((T.int64(640),), "float32"), unet_up_blocks_2_resnets_1_norm1_bias: T.Buffer((T.int64(640),), "float32"), var_T_multiply_intermediate: T.Buffer((T.int64(2), T.int64(640), T.int64(128), T.int64(128)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    T_reshape = T.alloc_buffer((T.int64(2), T.int64(32), T.int64(20), T.int64(128), T.int64(128)))
    A_red_temp_v0 = T.alloc_buffer((T.int64(2), T.int64(32)))
    A_red_temp_v1 = T.alloc_buffer((T.int64(2), T.int64(32)))
    T_reshape_1 = T.alloc_buffer((T.int64(32), T.int64(20)))
    T_reshape_2 = T.alloc_buffer((T.int64(32), T.int64(20)))
    T_group_norm = T.alloc_buffer((T.int64(2), T.int64(32), T.int64(20), T.int64(128), T.int64(128)))
    var_T_reshape_intermediate = T.alloc_buffer((T.int64(2), T.int64(640), T.int64(128), T.int64(128)))
    compute = T.alloc_buffer((T.int64(2), T.int64(640), T.int64(128), T.int64(128)))
    for ax0, ax1, ax2, ax3, ax4 in T.grid(T.int64(2), T.int64(32), T.int64(20), T.int64(128), T.int64(128)):
        with T.block("T_reshape"):
            v_ax0, v_ax1, v_ax2, v_ax3, v_ax4 = T.axis.remap("SSSSS", [ax0, ax1, ax2, ax3, ax4])
            T.reads(lv5204[((v_ax1 * T.int64(20) + (v_ax4 // T.int64(128) + v_ax3) // T.int64(128) + v_ax2) // T.int64(640) + v_ax0) % T.int64(2), (v_ax1 * T.int64(20) + (v_ax4 // T.int64(128) + v_ax3) // T.int64(128) + v_ax2) % T.int64(640), (v_ax4 // T.int64(128) + v_ax3) % T.int64(128), v_ax4 % T.int64(128)])
            T.writes(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4])
            T_reshape[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] = lv5204[((v_ax1 * T.int64(20) + (v_ax4 // T.int64(128) + v_ax3) // T.int64(128) + v_ax2) // T.int64(640) + v_ax0) % T.int64(2), (v_ax1 * T.int64(20) + (v_ax4 // T.int64(128) + v_ax3) // T.int64(128) + v_ax2) % T.int64(640), (v_ax4 // T.int64(128) + v_ax3) % T.int64(128), v_ax4 % T.int64(128)]
    for ax0, ax1, k2, k3, k4 in T.grid(T.int64(2), T.int64(32), T.int64(20), T.int64(128), T.int64(128)):
        with T.block("A_red_temp"):
            v_ax0, v_ax1, v_k2, v_k3, v_k4 = T.axis.remap("SSRRR", [ax0, ax1, k2, k3, k4])
            T.reads(T_reshape[v_ax0, v_ax1, v_k2, v_k3, v_k4])
            T.writes(A_red_temp_v0[v_ax0, v_ax1], A_red_temp_v1[v_ax0, v_ax1])
            with T.init():
                A_red_temp_v0[v_ax0, v_ax1] = T.float32(0)
                A_red_temp_v1[v_ax0, v_ax1] = T.float32(0)
            v_A_red_temp_v0: T.float32 = A_red_temp_v0[v_ax0, v_ax1] + T_reshape[v_ax0, v_ax1, v_k2, v_k3, v_k4]
            v_A_red_temp_v1: T.float32 = A_red_temp_v1[v_ax0, v_ax1] + T_reshape[v_ax0, v_ax1, v_k2, v_k3, v_k4] * T_reshape[v_ax0, v_ax1, v_k2, v_k3, v_k4]
            A_red_temp_v0[v_ax0, v_ax1] = v_A_red_temp_v0
            A_red_temp_v1[v_ax0, v_ax1] = v_A_red_temp_v1
    for ax0, ax1 in T.grid(T.int64(32), T.int64(20)):
        with T.block("T_reshape_1"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(unet_up_blocks_2_resnets_1_norm1_weight[(v_ax0 * T.int64(20) + v_ax1) % T.int64(640)])
            T.writes(T_reshape_1[v_ax0, v_ax1])
            T_reshape_1[v_ax0, v_ax1] = unet_up_blocks_2_resnets_1_norm1_weight[(v_ax0 * T.int64(20) + v_ax1) % T.int64(640)]
    for ax0, ax1 in T.grid(T.int64(32), T.int64(20)):
        with T.block("T_reshape_2"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(unet_up_blocks_2_resnets_1_norm1_bias[(v_ax0 * T.int64(20) + v_ax1) % T.int64(640)])
            T.writes(T_reshape_2[v_ax0, v_ax1])
            T_reshape_2[v_ax0, v_ax1] = unet_up_blocks_2_resnets_1_norm1_bias[(v_ax0 * T.int64(20) + v_ax1) % T.int64(640)]
    for ax0, ax1, ax2, ax3, ax4 in T.grid(T.int64(2), T.int64(32), T.int64(20), T.int64(128), T.int64(128)):
        with T.block("T_group_norm"):
            v_ax0, v_ax1, v_ax2, v_ax3, v_ax4 = T.axis.remap("SSSSS", [ax0, ax1, ax2, ax3, ax4])
            T.reads(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4], A_red_temp_v0[v_ax0, v_ax1], A_red_temp_v1[v_ax0, v_ax1], T_reshape_1[v_ax1, v_ax2], T_reshape_2[v_ax1, v_ax2])
            T.writes(T_group_norm[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4])
            T_group_norm[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] = (T_reshape[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] - A_red_temp_v0[v_ax0, v_ax1] * T.float32(3.0517578125000002e-06)) * T.rsqrt(A_red_temp_v1[v_ax0, v_ax1] * T.float32(3.0517578125000002e-06) - A_red_temp_v0[v_ax0, v_ax1] * T.float32(3.0517578125000002e-06) * (A_red_temp_v0[v_ax0, v_ax1] * T.float32(3.0517578125000002e-06)) + T.float32(1.0000000000000001e-05)) * T_reshape_1[v_ax1, v_ax2] + T_reshape_2[v_ax1, v_ax2]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(640), T.int64(128), T.int64(128)):
        with T.block("T_reshape_3"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(T_group_norm[(((v_ax3 // T.int64(128) + v_ax2) // T.int64(128) + v_ax1) // T.int64(640) + v_ax0) % T.int64(2), ((v_ax3 // T.int64(128) + v_ax2) // T.int64(128) + v_ax1) % T.int64(640) // T.int64(20), ((v_ax3 // T.int64(128) + v_ax2) // T.int64(128) + v_ax1) % T.int64(20), (v_ax3 // T.int64(128) + v_ax2) % T.int64(128), v_ax3 % T.int64(128)])
            T.writes(var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = T_group_norm[(((v_ax3 // T.int64(128) + v_ax2) // T.int64(128) + v_ax1) // T.int64(640) + v_ax0) % T.int64(2), ((v_ax3 // T.int64(128) + v_ax2) // T.int64(128) + v_ax1) % T.int64(640) // T.int64(20), ((v_ax3 // T.int64(128) + v_ax2) // T.int64(128) + v_ax1) % T.int64(20), (v_ax3 // T.int64(128) + v_ax2) % T.int64(128), v_ax3 % T.int64(128)]
    for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(640), T.int64(128), T.int64(128)):
        with T.block("compute"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(var_T_reshape_intermediate[v_i0, v_i1, v_i2, v_i3])
            T.writes(compute[v_i0, v_i1, v_i2, v_i3])
            compute[v_i0, v_i1, v_i2, v_i3] = T.sigmoid(var_T_reshape_intermediate[v_i0, v_i1, v_i2, v_i3])
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(640), T.int64(128), T.int64(128)):
        with T.block("T_multiply"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], compute[v_ax0, v_ax1, v_ax2, v_ax3])
            T.writes(var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] * compute[v_ax0, v_ax1, v_ax2, v_ax3]

@T.prim_func
def fused_group_norm14_silu13(lv6: T.Buffer((T.int64(1), T.int64(512), T.int64(128), T.int64(128)), "float32"), vae_decoder_mid_block_resnets_0_norm1_weight: T.Buffer((T.int64(512),), "float32"), vae_decoder_mid_block_resnets_0_norm1_bias: T.Buffer((T.int64(512),), "float32"), var_T_multiply_intermediate: T.Buffer((T.int64(1), T.int64(512), T.int64(128), T.int64(128)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    T_reshape = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(16), T.int64(128), T.int64(128)))
    A_red_temp_v0 = T.alloc_buffer((T.int64(1), T.int64(32)))
    A_red_temp_v1 = T.alloc_buffer((T.int64(1), T.int64(32)))
    T_reshape_1 = T.alloc_buffer((T.int64(32), T.int64(16)))
    T_reshape_2 = T.alloc_buffer((T.int64(32), T.int64(16)))
    T_group_norm = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(16), T.int64(128), T.int64(128)))
    var_T_reshape_intermediate = T.alloc_buffer((T.int64(1), T.int64(512), T.int64(128), T.int64(128)))
    compute = T.alloc_buffer((T.int64(1), T.int64(512), T.int64(128), T.int64(128)))
    for ax0, ax1, ax2, ax3, ax4 in T.grid(T.int64(1), T.int64(32), T.int64(16), T.int64(128), T.int64(128)):
        with T.block("T_reshape"):
            v_ax0, v_ax1, v_ax2, v_ax3, v_ax4 = T.axis.remap("SSSSS", [ax0, ax1, ax2, ax3, ax4])
            T.reads(lv6[T.int64(0), (v_ax1 * T.int64(16) + (v_ax4 // T.int64(128) + v_ax3) // T.int64(128) + v_ax2) % T.int64(512), (v_ax4 // T.int64(128) + v_ax3) % T.int64(128), v_ax4 % T.int64(128)])
            T.writes(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4])
            T_reshape[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] = lv6[T.int64(0), (v_ax1 * T.int64(16) + (v_ax4 // T.int64(128) + v_ax3) // T.int64(128) + v_ax2) % T.int64(512), (v_ax4 // T.int64(128) + v_ax3) % T.int64(128), v_ax4 % T.int64(128)]
    for ax0, ax1, k2, k3, k4 in T.grid(T.int64(1), T.int64(32), T.int64(16), T.int64(128), T.int64(128)):
        with T.block("A_red_temp"):
            v_ax0, v_ax1, v_k2, v_k3, v_k4 = T.axis.remap("SSRRR", [ax0, ax1, k2, k3, k4])
            T.reads(T_reshape[v_ax0, v_ax1, v_k2, v_k3, v_k4])
            T.writes(A_red_temp_v0[v_ax0, v_ax1], A_red_temp_v1[v_ax0, v_ax1])
            with T.init():
                A_red_temp_v0[v_ax0, v_ax1] = T.float32(0)
                A_red_temp_v1[v_ax0, v_ax1] = T.float32(0)
            v_A_red_temp_v0: T.float32 = A_red_temp_v0[v_ax0, v_ax1] + T_reshape[v_ax0, v_ax1, v_k2, v_k3, v_k4]
            v_A_red_temp_v1: T.float32 = A_red_temp_v1[v_ax0, v_ax1] + T_reshape[v_ax0, v_ax1, v_k2, v_k3, v_k4] * T_reshape[v_ax0, v_ax1, v_k2, v_k3, v_k4]
            A_red_temp_v0[v_ax0, v_ax1] = v_A_red_temp_v0
            A_red_temp_v1[v_ax0, v_ax1] = v_A_red_temp_v1
    for ax0, ax1 in T.grid(T.int64(32), T.int64(16)):
        with T.block("T_reshape_1"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(vae_decoder_mid_block_resnets_0_norm1_weight[(v_ax0 * T.int64(16) + v_ax1) % T.int64(512)])
            T.writes(T_reshape_1[v_ax0, v_ax1])
            T_reshape_1[v_ax0, v_ax1] = vae_decoder_mid_block_resnets_0_norm1_weight[(v_ax0 * T.int64(16) + v_ax1) % T.int64(512)]
    for ax0, ax1 in T.grid(T.int64(32), T.int64(16)):
        with T.block("T_reshape_2"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(vae_decoder_mid_block_resnets_0_norm1_bias[(v_ax0 * T.int64(16) + v_ax1) % T.int64(512)])
            T.writes(T_reshape_2[v_ax0, v_ax1])
            T_reshape_2[v_ax0, v_ax1] = vae_decoder_mid_block_resnets_0_norm1_bias[(v_ax0 * T.int64(16) + v_ax1) % T.int64(512)]
    for ax0, ax1, ax2, ax3, ax4 in T.grid(T.int64(1), T.int64(32), T.int64(16), T.int64(128), T.int64(128)):
        with T.block("T_group_norm"):
            v_ax0, v_ax1, v_ax2, v_ax3, v_ax4 = T.axis.remap("SSSSS", [ax0, ax1, ax2, ax3, ax4])
            T.reads(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4], A_red_temp_v0[v_ax0, v_ax1], A_red_temp_v1[v_ax0, v_ax1], T_reshape_1[v_ax1, v_ax2], T_reshape_2[v_ax1, v_ax2])
            T.writes(T_group_norm[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4])
            T_group_norm[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] = (T_reshape[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] - A_red_temp_v0[v_ax0, v_ax1] * T.float32(3.814697265625e-06)) * T.rsqrt(A_red_temp_v1[v_ax0, v_ax1] * T.float32(3.814697265625e-06) - A_red_temp_v0[v_ax0, v_ax1] * T.float32(3.814697265625e-06) * (A_red_temp_v0[v_ax0, v_ax1] * T.float32(3.814697265625e-06)) + T.float32(9.9999999999999995e-07)) * T_reshape_1[v_ax1, v_ax2] + T_reshape_2[v_ax1, v_ax2]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(512), T.int64(128), T.int64(128)):
        with T.block("T_reshape_3"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(T_group_norm[T.int64(0), ((v_ax3 // T.int64(128) + v_ax2) // T.int64(128) + v_ax1) % T.int64(512) // T.int64(16), ((v_ax3 // T.int64(128) + v_ax2) // T.int64(128) + v_ax1) % T.int64(16), (v_ax3 // T.int64(128) + v_ax2) % T.int64(128), v_ax3 % T.int64(128)])
            T.writes(var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = T_group_norm[T.int64(0), ((v_ax3 // T.int64(128) + v_ax2) // T.int64(128) + v_ax1) % T.int64(512) // T.int64(16), ((v_ax3 // T.int64(128) + v_ax2) // T.int64(128) + v_ax1) % T.int64(16), (v_ax3 // T.int64(128) + v_ax2) % T.int64(128), v_ax3 % T.int64(128)]
    for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(512), T.int64(128), T.int64(128)):
        with T.block("compute"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(var_T_reshape_intermediate[v_i0, v_i1, v_i2, v_i3])
            T.writes(compute[v_i0, v_i1, v_i2, v_i3])
            compute[v_i0, v_i1, v_i2, v_i3] = T.sigmoid(var_T_reshape_intermediate[v_i0, v_i1, v_i2, v_i3])
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(512), T.int64(128), T.int64(128)):
        with T.block("T_multiply"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], compute[v_ax0, v_ax1, v_ax2, v_ax3])
            T.writes(var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] * compute[v_ax0, v_ax1, v_ax2, v_ax3]

@T.prim_func
def fused_group_norm16_silu14(lv107: T.Buffer((T.int64(1), T.int64(512), T.int64(256), T.int64(256)), "float32"), vae_decoder_up_blocks_1_resnets_0_norm1_weight: T.Buffer((T.int64(512),), "float32"), vae_decoder_up_blocks_1_resnets_0_norm1_bias: T.Buffer((T.int64(512),), "float32"), var_T_multiply_intermediate: T.Buffer((T.int64(1), T.int64(512), T.int64(256), T.int64(256)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    T_reshape = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(16), T.int64(256), T.int64(256)))
    A_red_temp_v0 = T.alloc_buffer((T.int64(1), T.int64(32)))
    A_red_temp_v1 = T.alloc_buffer((T.int64(1), T.int64(32)))
    T_reshape_1 = T.alloc_buffer((T.int64(32), T.int64(16)))
    T_reshape_2 = T.alloc_buffer((T.int64(32), T.int64(16)))
    T_group_norm = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(16), T.int64(256), T.int64(256)))
    var_T_reshape_intermediate = T.alloc_buffer((T.int64(1), T.int64(512), T.int64(256), T.int64(256)))
    compute = T.alloc_buffer((T.int64(1), T.int64(512), T.int64(256), T.int64(256)))
    for ax0, ax1, ax2, ax3, ax4 in T.grid(T.int64(1), T.int64(32), T.int64(16), T.int64(256), T.int64(256)):
        with T.block("T_reshape"):
            v_ax0, v_ax1, v_ax2, v_ax3, v_ax4 = T.axis.remap("SSSSS", [ax0, ax1, ax2, ax3, ax4])
            T.reads(lv107[T.int64(0), (v_ax1 * T.int64(16) + (v_ax4 // T.int64(256) + v_ax3) // T.int64(256) + v_ax2) % T.int64(512), (v_ax4 // T.int64(256) + v_ax3) % T.int64(256), v_ax4 % T.int64(256)])
            T.writes(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4])
            T_reshape[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] = lv107[T.int64(0), (v_ax1 * T.int64(16) + (v_ax4 // T.int64(256) + v_ax3) // T.int64(256) + v_ax2) % T.int64(512), (v_ax4 // T.int64(256) + v_ax3) % T.int64(256), v_ax4 % T.int64(256)]
    for ax0, ax1, k2, k3, k4 in T.grid(T.int64(1), T.int64(32), T.int64(16), T.int64(256), T.int64(256)):
        with T.block("A_red_temp"):
            v_ax0, v_ax1, v_k2, v_k3, v_k4 = T.axis.remap("SSRRR", [ax0, ax1, k2, k3, k4])
            T.reads(T_reshape[v_ax0, v_ax1, v_k2, v_k3, v_k4])
            T.writes(A_red_temp_v0[v_ax0, v_ax1], A_red_temp_v1[v_ax0, v_ax1])
            with T.init():
                A_red_temp_v0[v_ax0, v_ax1] = T.float32(0)
                A_red_temp_v1[v_ax0, v_ax1] = T.float32(0)
            v_A_red_temp_v0: T.float32 = A_red_temp_v0[v_ax0, v_ax1] + T_reshape[v_ax0, v_ax1, v_k2, v_k3, v_k4]
            v_A_red_temp_v1: T.float32 = A_red_temp_v1[v_ax0, v_ax1] + T_reshape[v_ax0, v_ax1, v_k2, v_k3, v_k4] * T_reshape[v_ax0, v_ax1, v_k2, v_k3, v_k4]
            A_red_temp_v0[v_ax0, v_ax1] = v_A_red_temp_v0
            A_red_temp_v1[v_ax0, v_ax1] = v_A_red_temp_v1
    for ax0, ax1 in T.grid(T.int64(32), T.int64(16)):
        with T.block("T_reshape_1"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(vae_decoder_up_blocks_1_resnets_0_norm1_weight[(v_ax0 * T.int64(16) + v_ax1) % T.int64(512)])
            T.writes(T_reshape_1[v_ax0, v_ax1])
            T_reshape_1[v_ax0, v_ax1] = vae_decoder_up_blocks_1_resnets_0_norm1_weight[(v_ax0 * T.int64(16) + v_ax1) % T.int64(512)]
    for ax0, ax1 in T.grid(T.int64(32), T.int64(16)):
        with T.block("T_reshape_2"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(vae_decoder_up_blocks_1_resnets_0_norm1_bias[(v_ax0 * T.int64(16) + v_ax1) % T.int64(512)])
            T.writes(T_reshape_2[v_ax0, v_ax1])
            T_reshape_2[v_ax0, v_ax1] = vae_decoder_up_blocks_1_resnets_0_norm1_bias[(v_ax0 * T.int64(16) + v_ax1) % T.int64(512)]
    for ax0, ax1, ax2, ax3, ax4 in T.grid(T.int64(1), T.int64(32), T.int64(16), T.int64(256), T.int64(256)):
        with T.block("T_group_norm"):
            v_ax0, v_ax1, v_ax2, v_ax3, v_ax4 = T.axis.remap("SSSSS", [ax0, ax1, ax2, ax3, ax4])
            T.reads(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4], A_red_temp_v0[v_ax0, v_ax1], A_red_temp_v1[v_ax0, v_ax1], T_reshape_1[v_ax1, v_ax2], T_reshape_2[v_ax1, v_ax2])
            T.writes(T_group_norm[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4])
            T_group_norm[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] = (T_reshape[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] - A_red_temp_v0[v_ax0, v_ax1] * T.float32(9.5367431640625e-07)) * T.rsqrt(A_red_temp_v1[v_ax0, v_ax1] * T.float32(9.5367431640625e-07) - A_red_temp_v0[v_ax0, v_ax1] * T.float32(9.5367431640625e-07) * (A_red_temp_v0[v_ax0, v_ax1] * T.float32(9.5367431640625e-07)) + T.float32(9.9999999999999995e-07)) * T_reshape_1[v_ax1, v_ax2] + T_reshape_2[v_ax1, v_ax2]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(512), T.int64(256), T.int64(256)):
        with T.block("T_reshape_3"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(T_group_norm[T.int64(0), ((v_ax3 // T.int64(256) + v_ax2) // T.int64(256) + v_ax1) % T.int64(512) // T.int64(16), ((v_ax3 // T.int64(256) + v_ax2) // T.int64(256) + v_ax1) % T.int64(16), (v_ax3 // T.int64(256) + v_ax2) % T.int64(256), v_ax3 % T.int64(256)])
            T.writes(var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = T_group_norm[T.int64(0), ((v_ax3 // T.int64(256) + v_ax2) // T.int64(256) + v_ax1) % T.int64(512) // T.int64(16), ((v_ax3 // T.int64(256) + v_ax2) // T.int64(256) + v_ax1) % T.int64(16), (v_ax3 // T.int64(256) + v_ax2) % T.int64(256), v_ax3 % T.int64(256)]
    for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(512), T.int64(256), T.int64(256)):
        with T.block("compute"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(var_T_reshape_intermediate[v_i0, v_i1, v_i2, v_i3])
            T.writes(compute[v_i0, v_i1, v_i2, v_i3])
            compute[v_i0, v_i1, v_i2, v_i3] = T.sigmoid(var_T_reshape_intermediate[v_i0, v_i1, v_i2, v_i3])
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(512), T.int64(256), T.int64(256)):
        with T.block("T_multiply"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], compute[v_ax0, v_ax1, v_ax2, v_ax3])
            T.writes(var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] * compute[v_ax0, v_ax1, v_ax2, v_ax3]

@T.prim_func
def fused_group_norm17_silu15(lv147: T.Buffer((T.int64(1), T.int64(512), T.int64(512), T.int64(512)), "float32"), vae_decoder_up_blocks_2_resnets_0_norm1_weight: T.Buffer((T.int64(512),), "float32"), vae_decoder_up_blocks_2_resnets_0_norm1_bias: T.Buffer((T.int64(512),), "float32"), var_T_multiply_intermediate: T.Buffer((T.int64(1), T.int64(512), T.int64(512), T.int64(512)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    T_reshape = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(16), T.int64(512), T.int64(512)))
    A_red_temp_v0 = T.alloc_buffer((T.int64(1), T.int64(32)))
    A_red_temp_v1 = T.alloc_buffer((T.int64(1), T.int64(32)))
    T_reshape_1 = T.alloc_buffer((T.int64(32), T.int64(16)))
    T_reshape_2 = T.alloc_buffer((T.int64(32), T.int64(16)))
    T_group_norm = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(16), T.int64(512), T.int64(512)))
    var_T_reshape_intermediate = T.alloc_buffer((T.int64(1), T.int64(512), T.int64(512), T.int64(512)))
    compute = T.alloc_buffer((T.int64(1), T.int64(512), T.int64(512), T.int64(512)))
    for ax0, ax1, ax2, ax3, ax4 in T.grid(T.int64(1), T.int64(32), T.int64(16), T.int64(512), T.int64(512)):
        with T.block("T_reshape"):
            v_ax0, v_ax1, v_ax2, v_ax3, v_ax4 = T.axis.remap("SSSSS", [ax0, ax1, ax2, ax3, ax4])
            T.reads(lv147[T.int64(0), (v_ax1 * T.int64(16) + (v_ax4 // T.int64(512) + v_ax3) // T.int64(512) + v_ax2) % T.int64(512), (v_ax4 // T.int64(512) + v_ax3) % T.int64(512), v_ax4 % T.int64(512)])
            T.writes(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4])
            T_reshape[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] = lv147[T.int64(0), (v_ax1 * T.int64(16) + (v_ax4 // T.int64(512) + v_ax3) // T.int64(512) + v_ax2) % T.int64(512), (v_ax4 // T.int64(512) + v_ax3) % T.int64(512), v_ax4 % T.int64(512)]
    for ax0, ax1, k2, k3, k4 in T.grid(T.int64(1), T.int64(32), T.int64(16), T.int64(512), T.int64(512)):
        with T.block("A_red_temp"):
            v_ax0, v_ax1, v_k2, v_k3, v_k4 = T.axis.remap("SSRRR", [ax0, ax1, k2, k3, k4])
            T.reads(T_reshape[v_ax0, v_ax1, v_k2, v_k3, v_k4])
            T.writes(A_red_temp_v0[v_ax0, v_ax1], A_red_temp_v1[v_ax0, v_ax1])
            with T.init():
                A_red_temp_v0[v_ax0, v_ax1] = T.float32(0)
                A_red_temp_v1[v_ax0, v_ax1] = T.float32(0)
            v_A_red_temp_v0: T.float32 = A_red_temp_v0[v_ax0, v_ax1] + T_reshape[v_ax0, v_ax1, v_k2, v_k3, v_k4]
            v_A_red_temp_v1: T.float32 = A_red_temp_v1[v_ax0, v_ax1] + T_reshape[v_ax0, v_ax1, v_k2, v_k3, v_k4] * T_reshape[v_ax0, v_ax1, v_k2, v_k3, v_k4]
            A_red_temp_v0[v_ax0, v_ax1] = v_A_red_temp_v0
            A_red_temp_v1[v_ax0, v_ax1] = v_A_red_temp_v1
    for ax0, ax1 in T.grid(T.int64(32), T.int64(16)):
        with T.block("T_reshape_1"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(vae_decoder_up_blocks_2_resnets_0_norm1_weight[(v_ax0 * T.int64(16) + v_ax1) % T.int64(512)])
            T.writes(T_reshape_1[v_ax0, v_ax1])
            T_reshape_1[v_ax0, v_ax1] = vae_decoder_up_blocks_2_resnets_0_norm1_weight[(v_ax0 * T.int64(16) + v_ax1) % T.int64(512)]
    for ax0, ax1 in T.grid(T.int64(32), T.int64(16)):
        with T.block("T_reshape_2"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(vae_decoder_up_blocks_2_resnets_0_norm1_bias[(v_ax0 * T.int64(16) + v_ax1) % T.int64(512)])
            T.writes(T_reshape_2[v_ax0, v_ax1])
            T_reshape_2[v_ax0, v_ax1] = vae_decoder_up_blocks_2_resnets_0_norm1_bias[(v_ax0 * T.int64(16) + v_ax1) % T.int64(512)]
    for ax0, ax1, ax2, ax3, ax4 in T.grid(T.int64(1), T.int64(32), T.int64(16), T.int64(512), T.int64(512)):
        with T.block("T_group_norm"):
            v_ax0, v_ax1, v_ax2, v_ax3, v_ax4 = T.axis.remap("SSSSS", [ax0, ax1, ax2, ax3, ax4])
            T.reads(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4], A_red_temp_v0[v_ax0, v_ax1], A_red_temp_v1[v_ax0, v_ax1], T_reshape_1[v_ax1, v_ax2], T_reshape_2[v_ax1, v_ax2])
            T.writes(T_group_norm[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4])
            T_group_norm[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] = (T_reshape[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] - A_red_temp_v0[v_ax0, v_ax1] * T.float32(2.384185791015625e-07)) * T.rsqrt(A_red_temp_v1[v_ax0, v_ax1] * T.float32(2.384185791015625e-07) - A_red_temp_v0[v_ax0, v_ax1] * T.float32(2.384185791015625e-07) * (A_red_temp_v0[v_ax0, v_ax1] * T.float32(2.384185791015625e-07)) + T.float32(9.9999999999999995e-07)) * T_reshape_1[v_ax1, v_ax2] + T_reshape_2[v_ax1, v_ax2]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(512), T.int64(512), T.int64(512)):
        with T.block("T_reshape_3"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(T_group_norm[T.int64(0), ((v_ax3 // T.int64(512) + v_ax2) // T.int64(512) + v_ax1) % T.int64(512) // T.int64(16), ((v_ax3 // T.int64(512) + v_ax2) // T.int64(512) + v_ax1) % T.int64(16), (v_ax3 // T.int64(512) + v_ax2) % T.int64(512), v_ax3 % T.int64(512)])
            T.writes(var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = T_group_norm[T.int64(0), ((v_ax3 // T.int64(512) + v_ax2) // T.int64(512) + v_ax1) % T.int64(512) // T.int64(16), ((v_ax3 // T.int64(512) + v_ax2) // T.int64(512) + v_ax1) % T.int64(16), (v_ax3 // T.int64(512) + v_ax2) % T.int64(512), v_ax3 % T.int64(512)]
    for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(512), T.int64(512), T.int64(512)):
        with T.block("compute"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(var_T_reshape_intermediate[v_i0, v_i1, v_i2, v_i3])
            T.writes(compute[v_i0, v_i1, v_i2, v_i3])
            compute[v_i0, v_i1, v_i2, v_i3] = T.sigmoid(var_T_reshape_intermediate[v_i0, v_i1, v_i2, v_i3])
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(512), T.int64(512), T.int64(512)):
        with T.block("T_multiply"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], compute[v_ax0, v_ax1, v_ax2, v_ax3])
            T.writes(var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] * compute[v_ax0, v_ax1, v_ax2, v_ax3]

@T.prim_func
def fused_group_norm18_silu16(lv152: T.Buffer((T.int64(1), T.int64(256), T.int64(512), T.int64(512)), "float32"), vae_decoder_up_blocks_2_resnets_0_norm2_weight: T.Buffer((T.int64(256),), "float32"), vae_decoder_up_blocks_2_resnets_0_norm2_bias: T.Buffer((T.int64(256),), "float32"), var_T_multiply_intermediate: T.Buffer((T.int64(1), T.int64(256), T.int64(512), T.int64(512)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    T_reshape = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(8), T.int64(512), T.int64(512)))
    A_red_temp_v0 = T.alloc_buffer((T.int64(1), T.int64(32)))
    A_red_temp_v1 = T.alloc_buffer((T.int64(1), T.int64(32)))
    T_reshape_1 = T.alloc_buffer((T.int64(32), T.int64(8)))
    T_reshape_2 = T.alloc_buffer((T.int64(32), T.int64(8)))
    T_group_norm = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(8), T.int64(512), T.int64(512)))
    var_T_reshape_intermediate = T.alloc_buffer((T.int64(1), T.int64(256), T.int64(512), T.int64(512)))
    compute = T.alloc_buffer((T.int64(1), T.int64(256), T.int64(512), T.int64(512)))
    for ax0, ax1, ax2, ax3, ax4 in T.grid(T.int64(1), T.int64(32), T.int64(8), T.int64(512), T.int64(512)):
        with T.block("T_reshape"):
            v_ax0, v_ax1, v_ax2, v_ax3, v_ax4 = T.axis.remap("SSSSS", [ax0, ax1, ax2, ax3, ax4])
            T.reads(lv152[T.int64(0), (v_ax1 * T.int64(8) + (v_ax4 // T.int64(512) + v_ax3) // T.int64(512) + v_ax2) % T.int64(256), (v_ax4 // T.int64(512) + v_ax3) % T.int64(512), v_ax4 % T.int64(512)])
            T.writes(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4])
            T_reshape[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] = lv152[T.int64(0), (v_ax1 * T.int64(8) + (v_ax4 // T.int64(512) + v_ax3) // T.int64(512) + v_ax2) % T.int64(256), (v_ax4 // T.int64(512) + v_ax3) % T.int64(512), v_ax4 % T.int64(512)]
    for ax0, ax1, k2, k3, k4 in T.grid(T.int64(1), T.int64(32), T.int64(8), T.int64(512), T.int64(512)):
        with T.block("A_red_temp"):
            v_ax0, v_ax1, v_k2, v_k3, v_k4 = T.axis.remap("SSRRR", [ax0, ax1, k2, k3, k4])
            T.reads(T_reshape[v_ax0, v_ax1, v_k2, v_k3, v_k4])
            T.writes(A_red_temp_v0[v_ax0, v_ax1], A_red_temp_v1[v_ax0, v_ax1])
            with T.init():
                A_red_temp_v0[v_ax0, v_ax1] = T.float32(0)
                A_red_temp_v1[v_ax0, v_ax1] = T.float32(0)
            v_A_red_temp_v0: T.float32 = A_red_temp_v0[v_ax0, v_ax1] + T_reshape[v_ax0, v_ax1, v_k2, v_k3, v_k4]
            v_A_red_temp_v1: T.float32 = A_red_temp_v1[v_ax0, v_ax1] + T_reshape[v_ax0, v_ax1, v_k2, v_k3, v_k4] * T_reshape[v_ax0, v_ax1, v_k2, v_k3, v_k4]
            A_red_temp_v0[v_ax0, v_ax1] = v_A_red_temp_v0
            A_red_temp_v1[v_ax0, v_ax1] = v_A_red_temp_v1
    for ax0, ax1 in T.grid(T.int64(32), T.int64(8)):
        with T.block("T_reshape_1"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(vae_decoder_up_blocks_2_resnets_0_norm2_weight[(v_ax0 * T.int64(8) + v_ax1) % T.int64(256)])
            T.writes(T_reshape_1[v_ax0, v_ax1])
            T_reshape_1[v_ax0, v_ax1] = vae_decoder_up_blocks_2_resnets_0_norm2_weight[(v_ax0 * T.int64(8) + v_ax1) % T.int64(256)]
    for ax0, ax1 in T.grid(T.int64(32), T.int64(8)):
        with T.block("T_reshape_2"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(vae_decoder_up_blocks_2_resnets_0_norm2_bias[(v_ax0 * T.int64(8) + v_ax1) % T.int64(256)])
            T.writes(T_reshape_2[v_ax0, v_ax1])
            T_reshape_2[v_ax0, v_ax1] = vae_decoder_up_blocks_2_resnets_0_norm2_bias[(v_ax0 * T.int64(8) + v_ax1) % T.int64(256)]
    for ax0, ax1, ax2, ax3, ax4 in T.grid(T.int64(1), T.int64(32), T.int64(8), T.int64(512), T.int64(512)):
        with T.block("T_group_norm"):
            v_ax0, v_ax1, v_ax2, v_ax3, v_ax4 = T.axis.remap("SSSSS", [ax0, ax1, ax2, ax3, ax4])
            T.reads(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4], A_red_temp_v0[v_ax0, v_ax1], A_red_temp_v1[v_ax0, v_ax1], T_reshape_1[v_ax1, v_ax2], T_reshape_2[v_ax1, v_ax2])
            T.writes(T_group_norm[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4])
            T_group_norm[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] = (T_reshape[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] - A_red_temp_v0[v_ax0, v_ax1] * T.float32(4.76837158203125e-07)) * T.rsqrt(A_red_temp_v1[v_ax0, v_ax1] * T.float32(4.76837158203125e-07) - A_red_temp_v0[v_ax0, v_ax1] * T.float32(4.76837158203125e-07) * (A_red_temp_v0[v_ax0, v_ax1] * T.float32(4.76837158203125e-07)) + T.float32(9.9999999999999995e-07)) * T_reshape_1[v_ax1, v_ax2] + T_reshape_2[v_ax1, v_ax2]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(256), T.int64(512), T.int64(512)):
        with T.block("T_reshape_3"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(T_group_norm[T.int64(0), ((v_ax3 // T.int64(512) + v_ax2) // T.int64(512) + v_ax1) % T.int64(256) // T.int64(8), ((v_ax3 // T.int64(512) + v_ax2) // T.int64(512) + v_ax1) % T.int64(8), (v_ax3 // T.int64(512) + v_ax2) % T.int64(512), v_ax3 % T.int64(512)])
            T.writes(var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = T_group_norm[T.int64(0), ((v_ax3 // T.int64(512) + v_ax2) // T.int64(512) + v_ax1) % T.int64(256) // T.int64(8), ((v_ax3 // T.int64(512) + v_ax2) // T.int64(512) + v_ax1) % T.int64(8), (v_ax3 // T.int64(512) + v_ax2) % T.int64(512), v_ax3 % T.int64(512)]
    for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(256), T.int64(512), T.int64(512)):
        with T.block("compute"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(var_T_reshape_intermediate[v_i0, v_i1, v_i2, v_i3])
            T.writes(compute[v_i0, v_i1, v_i2, v_i3])
            compute[v_i0, v_i1, v_i2, v_i3] = T.sigmoid(var_T_reshape_intermediate[v_i0, v_i1, v_i2, v_i3])
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(256), T.int64(512), T.int64(512)):
        with T.block("T_multiply"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], compute[v_ax0, v_ax1, v_ax2, v_ax3])
            T.writes(var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] * compute[v_ax0, v_ax1, v_ax2, v_ax3]

@T.prim_func
def fused_group_norm19_silu17(lv190: T.Buffer((T.int64(1), T.int64(256), T.int64(1024), T.int64(1024)), "float32"), vae_decoder_up_blocks_3_resnets_0_norm1_weight: T.Buffer((T.int64(256),), "float32"), vae_decoder_up_blocks_3_resnets_0_norm1_bias: T.Buffer((T.int64(256),), "float32"), var_T_multiply_intermediate: T.Buffer((T.int64(1), T.int64(256), T.int64(1024), T.int64(1024)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    T_reshape = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(8), T.int64(1024), T.int64(1024)))
    A_red_temp_v0 = T.alloc_buffer((T.int64(1), T.int64(32)))
    A_red_temp_v1 = T.alloc_buffer((T.int64(1), T.int64(32)))
    T_reshape_1 = T.alloc_buffer((T.int64(32), T.int64(8)))
    T_reshape_2 = T.alloc_buffer((T.int64(32), T.int64(8)))
    T_group_norm = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(8), T.int64(1024), T.int64(1024)))
    var_T_reshape_intermediate = T.alloc_buffer((T.int64(1), T.int64(256), T.int64(1024), T.int64(1024)))
    compute = T.alloc_buffer((T.int64(1), T.int64(256), T.int64(1024), T.int64(1024)))
    for ax0, ax1, ax2, ax3, ax4 in T.grid(T.int64(1), T.int64(32), T.int64(8), T.int64(1024), T.int64(1024)):
        with T.block("T_reshape"):
            v_ax0, v_ax1, v_ax2, v_ax3, v_ax4 = T.axis.remap("SSSSS", [ax0, ax1, ax2, ax3, ax4])
            T.reads(lv190[T.int64(0), (v_ax1 * T.int64(8) + (v_ax4 // T.int64(1024) + v_ax3) // T.int64(1024) + v_ax2) % T.int64(256), (v_ax4 // T.int64(1024) + v_ax3) % T.int64(1024), v_ax4 % T.int64(1024)])
            T.writes(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4])
            T_reshape[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] = lv190[T.int64(0), (v_ax1 * T.int64(8) + (v_ax4 // T.int64(1024) + v_ax3) // T.int64(1024) + v_ax2) % T.int64(256), (v_ax4 // T.int64(1024) + v_ax3) % T.int64(1024), v_ax4 % T.int64(1024)]
    for ax0, ax1, k2, k3, k4 in T.grid(T.int64(1), T.int64(32), T.int64(8), T.int64(1024), T.int64(1024)):
        with T.block("A_red_temp"):
            v_ax0, v_ax1, v_k2, v_k3, v_k4 = T.axis.remap("SSRRR", [ax0, ax1, k2, k3, k4])
            T.reads(T_reshape[v_ax0, v_ax1, v_k2, v_k3, v_k4])
            T.writes(A_red_temp_v0[v_ax0, v_ax1], A_red_temp_v1[v_ax0, v_ax1])
            with T.init():
                A_red_temp_v0[v_ax0, v_ax1] = T.float32(0)
                A_red_temp_v1[v_ax0, v_ax1] = T.float32(0)
            v_A_red_temp_v0: T.float32 = A_red_temp_v0[v_ax0, v_ax1] + T_reshape[v_ax0, v_ax1, v_k2, v_k3, v_k4]
            v_A_red_temp_v1: T.float32 = A_red_temp_v1[v_ax0, v_ax1] + T_reshape[v_ax0, v_ax1, v_k2, v_k3, v_k4] * T_reshape[v_ax0, v_ax1, v_k2, v_k3, v_k4]
            A_red_temp_v0[v_ax0, v_ax1] = v_A_red_temp_v0
            A_red_temp_v1[v_ax0, v_ax1] = v_A_red_temp_v1
    for ax0, ax1 in T.grid(T.int64(32), T.int64(8)):
        with T.block("T_reshape_1"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(vae_decoder_up_blocks_3_resnets_0_norm1_weight[(v_ax0 * T.int64(8) + v_ax1) % T.int64(256)])
            T.writes(T_reshape_1[v_ax0, v_ax1])
            T_reshape_1[v_ax0, v_ax1] = vae_decoder_up_blocks_3_resnets_0_norm1_weight[(v_ax0 * T.int64(8) + v_ax1) % T.int64(256)]
    for ax0, ax1 in T.grid(T.int64(32), T.int64(8)):
        with T.block("T_reshape_2"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(vae_decoder_up_blocks_3_resnets_0_norm1_bias[(v_ax0 * T.int64(8) + v_ax1) % T.int64(256)])
            T.writes(T_reshape_2[v_ax0, v_ax1])
            T_reshape_2[v_ax0, v_ax1] = vae_decoder_up_blocks_3_resnets_0_norm1_bias[(v_ax0 * T.int64(8) + v_ax1) % T.int64(256)]
    for ax0, ax1, ax2, ax3, ax4 in T.grid(T.int64(1), T.int64(32), T.int64(8), T.int64(1024), T.int64(1024)):
        with T.block("T_group_norm"):
            v_ax0, v_ax1, v_ax2, v_ax3, v_ax4 = T.axis.remap("SSSSS", [ax0, ax1, ax2, ax3, ax4])
            T.reads(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4], A_red_temp_v0[v_ax0, v_ax1], A_red_temp_v1[v_ax0, v_ax1], T_reshape_1[v_ax1, v_ax2], T_reshape_2[v_ax1, v_ax2])
            T.writes(T_group_norm[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4])
            T_group_norm[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] = (T_reshape[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] - A_red_temp_v0[v_ax0, v_ax1] * T.float32(1.1920928955078125e-07)) * T.rsqrt(A_red_temp_v1[v_ax0, v_ax1] * T.float32(1.1920928955078125e-07) - A_red_temp_v0[v_ax0, v_ax1] * T.float32(1.1920928955078125e-07) * (A_red_temp_v0[v_ax0, v_ax1] * T.float32(1.1920928955078125e-07)) + T.float32(9.9999999999999995e-07)) * T_reshape_1[v_ax1, v_ax2] + T_reshape_2[v_ax1, v_ax2]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(256), T.int64(1024), T.int64(1024)):
        with T.block("T_reshape_3"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(T_group_norm[T.int64(0), ((v_ax3 // T.int64(1024) + v_ax2) // T.int64(1024) + v_ax1) % T.int64(256) // T.int64(8), ((v_ax3 // T.int64(1024) + v_ax2) // T.int64(1024) + v_ax1) % T.int64(8), (v_ax3 // T.int64(1024) + v_ax2) % T.int64(1024), v_ax3 % T.int64(1024)])
            T.writes(var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = T_group_norm[T.int64(0), ((v_ax3 // T.int64(1024) + v_ax2) // T.int64(1024) + v_ax1) % T.int64(256) // T.int64(8), ((v_ax3 // T.int64(1024) + v_ax2) // T.int64(1024) + v_ax1) % T.int64(8), (v_ax3 // T.int64(1024) + v_ax2) % T.int64(1024), v_ax3 % T.int64(1024)]
    for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(256), T.int64(1024), T.int64(1024)):
        with T.block("compute"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(var_T_reshape_intermediate[v_i0, v_i1, v_i2, v_i3])
            T.writes(compute[v_i0, v_i1, v_i2, v_i3])
            compute[v_i0, v_i1, v_i2, v_i3] = T.sigmoid(var_T_reshape_intermediate[v_i0, v_i1, v_i2, v_i3])
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(256), T.int64(1024), T.int64(1024)):
        with T.block("T_multiply"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], compute[v_ax0, v_ax1, v_ax2, v_ax3])
            T.writes(var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] * compute[v_ax0, v_ax1, v_ax2, v_ax3]

@T.prim_func
def fused_group_norm1_silu2(lv89: T.Buffer((T.int64(2), T.int64(320), T.int64(64), T.int64(64)), "float32"), unet_down_blocks_1_resnets_0_norm1_weight: T.Buffer((T.int64(320),), "float32"), unet_down_blocks_1_resnets_0_norm1_bias: T.Buffer((T.int64(320),), "float32"), var_T_multiply_intermediate: T.Buffer((T.int64(2), T.int64(320), T.int64(64), T.int64(64)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    T_reshape = T.alloc_buffer((T.int64(2), T.int64(32), T.int64(10), T.int64(64), T.int64(64)))
    A_red_temp_v0 = T.alloc_buffer((T.int64(2), T.int64(32)))
    A_red_temp_v1 = T.alloc_buffer((T.int64(2), T.int64(32)))
    T_reshape_1 = T.alloc_buffer((T.int64(32), T.int64(10)))
    T_reshape_2 = T.alloc_buffer((T.int64(32), T.int64(10)))
    T_group_norm = T.alloc_buffer((T.int64(2), T.int64(32), T.int64(10), T.int64(64), T.int64(64)))
    var_T_reshape_intermediate = T.alloc_buffer((T.int64(2), T.int64(320), T.int64(64), T.int64(64)))
    compute = T.alloc_buffer((T.int64(2), T.int64(320), T.int64(64), T.int64(64)))
    for ax0, ax1, ax2, ax3, ax4 in T.grid(T.int64(2), T.int64(32), T.int64(10), T.int64(64), T.int64(64)):
        with T.block("T_reshape"):
            v_ax0, v_ax1, v_ax2, v_ax3, v_ax4 = T.axis.remap("SSSSS", [ax0, ax1, ax2, ax3, ax4])
            T.reads(lv89[((v_ax1 * T.int64(10) + (v_ax4 // T.int64(64) + v_ax3) // T.int64(64) + v_ax2) // T.int64(320) + v_ax0) % T.int64(2), (v_ax1 * T.int64(10) + (v_ax4 // T.int64(64) + v_ax3) // T.int64(64) + v_ax2) % T.int64(320), (v_ax4 // T.int64(64) + v_ax3) % T.int64(64), v_ax4 % T.int64(64)])
            T.writes(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4])
            T_reshape[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] = lv89[((v_ax1 * T.int64(10) + (v_ax4 // T.int64(64) + v_ax3) // T.int64(64) + v_ax2) // T.int64(320) + v_ax0) % T.int64(2), (v_ax1 * T.int64(10) + (v_ax4 // T.int64(64) + v_ax3) // T.int64(64) + v_ax2) % T.int64(320), (v_ax4 // T.int64(64) + v_ax3) % T.int64(64), v_ax4 % T.int64(64)]
    for ax0, ax1, k2, k3, k4 in T.grid(T.int64(2), T.int64(32), T.int64(10), T.int64(64), T.int64(64)):
        with T.block("A_red_temp"):
            v_ax0, v_ax1, v_k2, v_k3, v_k4 = T.axis.remap("SSRRR", [ax0, ax1, k2, k3, k4])
            T.reads(T_reshape[v_ax0, v_ax1, v_k2, v_k3, v_k4])
            T.writes(A_red_temp_v0[v_ax0, v_ax1], A_red_temp_v1[v_ax0, v_ax1])
            with T.init():
                A_red_temp_v0[v_ax0, v_ax1] = T.float32(0)
                A_red_temp_v1[v_ax0, v_ax1] = T.float32(0)
            v_A_red_temp_v0: T.float32 = A_red_temp_v0[v_ax0, v_ax1] + T_reshape[v_ax0, v_ax1, v_k2, v_k3, v_k4]
            v_A_red_temp_v1: T.float32 = A_red_temp_v1[v_ax0, v_ax1] + T_reshape[v_ax0, v_ax1, v_k2, v_k3, v_k4] * T_reshape[v_ax0, v_ax1, v_k2, v_k3, v_k4]
            A_red_temp_v0[v_ax0, v_ax1] = v_A_red_temp_v0
            A_red_temp_v1[v_ax0, v_ax1] = v_A_red_temp_v1
    for ax0, ax1 in T.grid(T.int64(32), T.int64(10)):
        with T.block("T_reshape_1"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(unet_down_blocks_1_resnets_0_norm1_weight[(v_ax0 * T.int64(10) + v_ax1) % T.int64(320)])
            T.writes(T_reshape_1[v_ax0, v_ax1])
            T_reshape_1[v_ax0, v_ax1] = unet_down_blocks_1_resnets_0_norm1_weight[(v_ax0 * T.int64(10) + v_ax1) % T.int64(320)]
    for ax0, ax1 in T.grid(T.int64(32), T.int64(10)):
        with T.block("T_reshape_2"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(unet_down_blocks_1_resnets_0_norm1_bias[(v_ax0 * T.int64(10) + v_ax1) % T.int64(320)])
            T.writes(T_reshape_2[v_ax0, v_ax1])
            T_reshape_2[v_ax0, v_ax1] = unet_down_blocks_1_resnets_0_norm1_bias[(v_ax0 * T.int64(10) + v_ax1) % T.int64(320)]
    for ax0, ax1, ax2, ax3, ax4 in T.grid(T.int64(2), T.int64(32), T.int64(10), T.int64(64), T.int64(64)):
        with T.block("T_group_norm"):
            v_ax0, v_ax1, v_ax2, v_ax3, v_ax4 = T.axis.remap("SSSSS", [ax0, ax1, ax2, ax3, ax4])
            T.reads(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4], A_red_temp_v0[v_ax0, v_ax1], A_red_temp_v1[v_ax0, v_ax1], T_reshape_1[v_ax1, v_ax2], T_reshape_2[v_ax1, v_ax2])
            T.writes(T_group_norm[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4])
            T_group_norm[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] = (T_reshape[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] - A_red_temp_v0[v_ax0, v_ax1] * T.float32(2.4414062500000001e-05)) * T.rsqrt(A_red_temp_v1[v_ax0, v_ax1] * T.float32(2.4414062500000001e-05) - A_red_temp_v0[v_ax0, v_ax1] * T.float32(2.4414062500000001e-05) * (A_red_temp_v0[v_ax0, v_ax1] * T.float32(2.4414062500000001e-05)) + T.float32(1.0000000000000001e-05)) * T_reshape_1[v_ax1, v_ax2] + T_reshape_2[v_ax1, v_ax2]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(320), T.int64(64), T.int64(64)):
        with T.block("T_reshape_3"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(T_group_norm[(((v_ax3 // T.int64(64) + v_ax2) // T.int64(64) + v_ax1) // T.int64(320) + v_ax0) % T.int64(2), ((v_ax3 // T.int64(64) + v_ax2) // T.int64(64) + v_ax1) % T.int64(320) // T.int64(10), ((v_ax3 // T.int64(64) + v_ax2) // T.int64(64) + v_ax1) % T.int64(10), (v_ax3 // T.int64(64) + v_ax2) % T.int64(64), v_ax3 % T.int64(64)])
            T.writes(var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = T_group_norm[(((v_ax3 // T.int64(64) + v_ax2) // T.int64(64) + v_ax1) // T.int64(320) + v_ax0) % T.int64(2), ((v_ax3 // T.int64(64) + v_ax2) // T.int64(64) + v_ax1) % T.int64(320) // T.int64(10), ((v_ax3 // T.int64(64) + v_ax2) // T.int64(64) + v_ax1) % T.int64(10), (v_ax3 // T.int64(64) + v_ax2) % T.int64(64), v_ax3 % T.int64(64)]
    for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(320), T.int64(64), T.int64(64)):
        with T.block("compute"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(var_T_reshape_intermediate[v_i0, v_i1, v_i2, v_i3])
            T.writes(compute[v_i0, v_i1, v_i2, v_i3])
            compute[v_i0, v_i1, v_i2, v_i3] = T.sigmoid(var_T_reshape_intermediate[v_i0, v_i1, v_i2, v_i3])
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(320), T.int64(64), T.int64(64)):
        with T.block("T_multiply"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], compute[v_ax0, v_ax1, v_ax2, v_ax3])
            T.writes(var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] * compute[v_ax0, v_ax1, v_ax2, v_ax3]

@T.prim_func
def fused_group_norm20_silu18(lv195: T.Buffer((T.int64(1), T.int64(128), T.int64(1024), T.int64(1024)), "float32"), vae_decoder_up_blocks_3_resnets_0_norm2_weight: T.Buffer((T.int64(128),), "float32"), vae_decoder_up_blocks_3_resnets_0_norm2_bias: T.Buffer((T.int64(128),), "float32"), var_T_multiply_intermediate: T.Buffer((T.int64(1), T.int64(128), T.int64(1024), T.int64(1024)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    T_reshape = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(4), T.int64(1024), T.int64(1024)))
    A_red_temp_v0 = T.alloc_buffer((T.int64(1), T.int64(32)))
    A_red_temp_v1 = T.alloc_buffer((T.int64(1), T.int64(32)))
    T_reshape_1 = T.alloc_buffer((T.int64(32), T.int64(4)))
    T_reshape_2 = T.alloc_buffer((T.int64(32), T.int64(4)))
    T_group_norm = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(4), T.int64(1024), T.int64(1024)))
    var_T_reshape_intermediate = T.alloc_buffer((T.int64(1), T.int64(128), T.int64(1024), T.int64(1024)))
    compute = T.alloc_buffer((T.int64(1), T.int64(128), T.int64(1024), T.int64(1024)))
    for ax0, ax1, ax2, ax3, ax4 in T.grid(T.int64(1), T.int64(32), T.int64(4), T.int64(1024), T.int64(1024)):
        with T.block("T_reshape"):
            v_ax0, v_ax1, v_ax2, v_ax3, v_ax4 = T.axis.remap("SSSSS", [ax0, ax1, ax2, ax3, ax4])
            T.reads(lv195[T.int64(0), (v_ax1 * T.int64(4) + (v_ax4 // T.int64(1024) + v_ax3) // T.int64(1024) + v_ax2) % T.int64(128), (v_ax4 // T.int64(1024) + v_ax3) % T.int64(1024), v_ax4 % T.int64(1024)])
            T.writes(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4])
            T_reshape[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] = lv195[T.int64(0), (v_ax1 * T.int64(4) + (v_ax4 // T.int64(1024) + v_ax3) // T.int64(1024) + v_ax2) % T.int64(128), (v_ax4 // T.int64(1024) + v_ax3) % T.int64(1024), v_ax4 % T.int64(1024)]
    for ax0, ax1, k2, k3, k4 in T.grid(T.int64(1), T.int64(32), T.int64(4), T.int64(1024), T.int64(1024)):
        with T.block("A_red_temp"):
            v_ax0, v_ax1, v_k2, v_k3, v_k4 = T.axis.remap("SSRRR", [ax0, ax1, k2, k3, k4])
            T.reads(T_reshape[v_ax0, v_ax1, v_k2, v_k3, v_k4])
            T.writes(A_red_temp_v0[v_ax0, v_ax1], A_red_temp_v1[v_ax0, v_ax1])
            with T.init():
                A_red_temp_v0[v_ax0, v_ax1] = T.float32(0)
                A_red_temp_v1[v_ax0, v_ax1] = T.float32(0)
            v_A_red_temp_v0: T.float32 = A_red_temp_v0[v_ax0, v_ax1] + T_reshape[v_ax0, v_ax1, v_k2, v_k3, v_k4]
            v_A_red_temp_v1: T.float32 = A_red_temp_v1[v_ax0, v_ax1] + T_reshape[v_ax0, v_ax1, v_k2, v_k3, v_k4] * T_reshape[v_ax0, v_ax1, v_k2, v_k3, v_k4]
            A_red_temp_v0[v_ax0, v_ax1] = v_A_red_temp_v0
            A_red_temp_v1[v_ax0, v_ax1] = v_A_red_temp_v1
    for ax0, ax1 in T.grid(T.int64(32), T.int64(4)):
        with T.block("T_reshape_1"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(vae_decoder_up_blocks_3_resnets_0_norm2_weight[(v_ax0 * T.int64(4) + v_ax1) % T.int64(128)])
            T.writes(T_reshape_1[v_ax0, v_ax1])
            T_reshape_1[v_ax0, v_ax1] = vae_decoder_up_blocks_3_resnets_0_norm2_weight[(v_ax0 * T.int64(4) + v_ax1) % T.int64(128)]
    for ax0, ax1 in T.grid(T.int64(32), T.int64(4)):
        with T.block("T_reshape_2"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(vae_decoder_up_blocks_3_resnets_0_norm2_bias[(v_ax0 * T.int64(4) + v_ax1) % T.int64(128)])
            T.writes(T_reshape_2[v_ax0, v_ax1])
            T_reshape_2[v_ax0, v_ax1] = vae_decoder_up_blocks_3_resnets_0_norm2_bias[(v_ax0 * T.int64(4) + v_ax1) % T.int64(128)]
    for ax0, ax1, ax2, ax3, ax4 in T.grid(T.int64(1), T.int64(32), T.int64(4), T.int64(1024), T.int64(1024)):
        with T.block("T_group_norm"):
            v_ax0, v_ax1, v_ax2, v_ax3, v_ax4 = T.axis.remap("SSSSS", [ax0, ax1, ax2, ax3, ax4])
            T.reads(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4], A_red_temp_v0[v_ax0, v_ax1], A_red_temp_v1[v_ax0, v_ax1], T_reshape_1[v_ax1, v_ax2], T_reshape_2[v_ax1, v_ax2])
            T.writes(T_group_norm[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4])
            T_group_norm[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] = (T_reshape[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] - A_red_temp_v0[v_ax0, v_ax1] * T.float32(2.384185791015625e-07)) * T.rsqrt(A_red_temp_v1[v_ax0, v_ax1] * T.float32(2.384185791015625e-07) - A_red_temp_v0[v_ax0, v_ax1] * T.float32(2.384185791015625e-07) * (A_red_temp_v0[v_ax0, v_ax1] * T.float32(2.384185791015625e-07)) + T.float32(9.9999999999999995e-07)) * T_reshape_1[v_ax1, v_ax2] + T_reshape_2[v_ax1, v_ax2]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(128), T.int64(1024), T.int64(1024)):
        with T.block("T_reshape_3"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(T_group_norm[T.int64(0), ((v_ax3 // T.int64(1024) + v_ax2) // T.int64(1024) + v_ax1) % T.int64(128) // T.int64(4), ((v_ax3 // T.int64(1024) + v_ax2) // T.int64(1024) + v_ax1) % T.int64(4), (v_ax3 // T.int64(1024) + v_ax2) % T.int64(1024), v_ax3 % T.int64(1024)])
            T.writes(var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = T_group_norm[T.int64(0), ((v_ax3 // T.int64(1024) + v_ax2) // T.int64(1024) + v_ax1) % T.int64(128) // T.int64(4), ((v_ax3 // T.int64(1024) + v_ax2) // T.int64(1024) + v_ax1) % T.int64(4), (v_ax3 // T.int64(1024) + v_ax2) % T.int64(1024), v_ax3 % T.int64(1024)]
    for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(128), T.int64(1024), T.int64(1024)):
        with T.block("compute"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(var_T_reshape_intermediate[v_i0, v_i1, v_i2, v_i3])
            T.writes(compute[v_i0, v_i1, v_i2, v_i3])
            compute[v_i0, v_i1, v_i2, v_i3] = T.sigmoid(var_T_reshape_intermediate[v_i0, v_i1, v_i2, v_i3])
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(128), T.int64(1024), T.int64(1024)):
        with T.block("T_multiply"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], compute[v_ax0, v_ax1, v_ax2, v_ax3])
            T.writes(var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] * compute[v_ax0, v_ax1, v_ax2, v_ax3]

@T.prim_func
def fused_group_norm2_silu3(lv101: T.Buffer((T.int64(2), T.int64(640), T.int64(64), T.int64(64)), "float32"), unet_down_blocks_1_resnets_0_norm2_weight: T.Buffer((T.int64(640),), "float32"), unet_down_blocks_1_resnets_0_norm2_bias: T.Buffer((T.int64(640),), "float32"), var_T_multiply_intermediate: T.Buffer((T.int64(2), T.int64(640), T.int64(64), T.int64(64)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    T_reshape = T.alloc_buffer((T.int64(2), T.int64(32), T.int64(20), T.int64(64), T.int64(64)))
    A_red_temp_v0 = T.alloc_buffer((T.int64(2), T.int64(32)))
    A_red_temp_v1 = T.alloc_buffer((T.int64(2), T.int64(32)))
    T_reshape_1 = T.alloc_buffer((T.int64(32), T.int64(20)))
    T_reshape_2 = T.alloc_buffer((T.int64(32), T.int64(20)))
    T_group_norm = T.alloc_buffer((T.int64(2), T.int64(32), T.int64(20), T.int64(64), T.int64(64)))
    var_T_reshape_intermediate = T.alloc_buffer((T.int64(2), T.int64(640), T.int64(64), T.int64(64)))
    compute = T.alloc_buffer((T.int64(2), T.int64(640), T.int64(64), T.int64(64)))
    for ax0, ax1, ax2, ax3, ax4 in T.grid(T.int64(2), T.int64(32), T.int64(20), T.int64(64), T.int64(64)):
        with T.block("T_reshape"):
            v_ax0, v_ax1, v_ax2, v_ax3, v_ax4 = T.axis.remap("SSSSS", [ax0, ax1, ax2, ax3, ax4])
            T.reads(lv101[((v_ax1 * T.int64(20) + (v_ax4 // T.int64(64) + v_ax3) // T.int64(64) + v_ax2) // T.int64(640) + v_ax0) % T.int64(2), (v_ax1 * T.int64(20) + (v_ax4 // T.int64(64) + v_ax3) // T.int64(64) + v_ax2) % T.int64(640), (v_ax4 // T.int64(64) + v_ax3) % T.int64(64), v_ax4 % T.int64(64)])
            T.writes(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4])
            T_reshape[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] = lv101[((v_ax1 * T.int64(20) + (v_ax4 // T.int64(64) + v_ax3) // T.int64(64) + v_ax2) // T.int64(640) + v_ax0) % T.int64(2), (v_ax1 * T.int64(20) + (v_ax4 // T.int64(64) + v_ax3) // T.int64(64) + v_ax2) % T.int64(640), (v_ax4 // T.int64(64) + v_ax3) % T.int64(64), v_ax4 % T.int64(64)]
    for ax0, ax1, k2, k3, k4 in T.grid(T.int64(2), T.int64(32), T.int64(20), T.int64(64), T.int64(64)):
        with T.block("A_red_temp"):
            v_ax0, v_ax1, v_k2, v_k3, v_k4 = T.axis.remap("SSRRR", [ax0, ax1, k2, k3, k4])
            T.reads(T_reshape[v_ax0, v_ax1, v_k2, v_k3, v_k4])
            T.writes(A_red_temp_v0[v_ax0, v_ax1], A_red_temp_v1[v_ax0, v_ax1])
            with T.init():
                A_red_temp_v0[v_ax0, v_ax1] = T.float32(0)
                A_red_temp_v1[v_ax0, v_ax1] = T.float32(0)
            v_A_red_temp_v0: T.float32 = A_red_temp_v0[v_ax0, v_ax1] + T_reshape[v_ax0, v_ax1, v_k2, v_k3, v_k4]
            v_A_red_temp_v1: T.float32 = A_red_temp_v1[v_ax0, v_ax1] + T_reshape[v_ax0, v_ax1, v_k2, v_k3, v_k4] * T_reshape[v_ax0, v_ax1, v_k2, v_k3, v_k4]
            A_red_temp_v0[v_ax0, v_ax1] = v_A_red_temp_v0
            A_red_temp_v1[v_ax0, v_ax1] = v_A_red_temp_v1
    for ax0, ax1 in T.grid(T.int64(32), T.int64(20)):
        with T.block("T_reshape_1"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(unet_down_blocks_1_resnets_0_norm2_weight[(v_ax0 * T.int64(20) + v_ax1) % T.int64(640)])
            T.writes(T_reshape_1[v_ax0, v_ax1])
            T_reshape_1[v_ax0, v_ax1] = unet_down_blocks_1_resnets_0_norm2_weight[(v_ax0 * T.int64(20) + v_ax1) % T.int64(640)]
    for ax0, ax1 in T.grid(T.int64(32), T.int64(20)):
        with T.block("T_reshape_2"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(unet_down_blocks_1_resnets_0_norm2_bias[(v_ax0 * T.int64(20) + v_ax1) % T.int64(640)])
            T.writes(T_reshape_2[v_ax0, v_ax1])
            T_reshape_2[v_ax0, v_ax1] = unet_down_blocks_1_resnets_0_norm2_bias[(v_ax0 * T.int64(20) + v_ax1) % T.int64(640)]
    for ax0, ax1, ax2, ax3, ax4 in T.grid(T.int64(2), T.int64(32), T.int64(20), T.int64(64), T.int64(64)):
        with T.block("T_group_norm"):
            v_ax0, v_ax1, v_ax2, v_ax3, v_ax4 = T.axis.remap("SSSSS", [ax0, ax1, ax2, ax3, ax4])
            T.reads(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4], A_red_temp_v0[v_ax0, v_ax1], A_red_temp_v1[v_ax0, v_ax1], T_reshape_1[v_ax1, v_ax2], T_reshape_2[v_ax1, v_ax2])
            T.writes(T_group_norm[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4])
            T_group_norm[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] = (T_reshape[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] - A_red_temp_v0[v_ax0, v_ax1] * T.float32(1.2207031250000001e-05)) * T.rsqrt(A_red_temp_v1[v_ax0, v_ax1] * T.float32(1.2207031250000001e-05) - A_red_temp_v0[v_ax0, v_ax1] * T.float32(1.2207031250000001e-05) * (A_red_temp_v0[v_ax0, v_ax1] * T.float32(1.2207031250000001e-05)) + T.float32(1.0000000000000001e-05)) * T_reshape_1[v_ax1, v_ax2] + T_reshape_2[v_ax1, v_ax2]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(640), T.int64(64), T.int64(64)):
        with T.block("T_reshape_3"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(T_group_norm[(((v_ax3 // T.int64(64) + v_ax2) // T.int64(64) + v_ax1) // T.int64(640) + v_ax0) % T.int64(2), ((v_ax3 // T.int64(64) + v_ax2) // T.int64(64) + v_ax1) % T.int64(640) // T.int64(20), ((v_ax3 // T.int64(64) + v_ax2) // T.int64(64) + v_ax1) % T.int64(20), (v_ax3 // T.int64(64) + v_ax2) % T.int64(64), v_ax3 % T.int64(64)])
            T.writes(var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = T_group_norm[(((v_ax3 // T.int64(64) + v_ax2) // T.int64(64) + v_ax1) // T.int64(640) + v_ax0) % T.int64(2), ((v_ax3 // T.int64(64) + v_ax2) // T.int64(64) + v_ax1) % T.int64(640) // T.int64(20), ((v_ax3 // T.int64(64) + v_ax2) // T.int64(64) + v_ax1) % T.int64(20), (v_ax3 // T.int64(64) + v_ax2) % T.int64(64), v_ax3 % T.int64(64)]
    for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(640), T.int64(64), T.int64(64)):
        with T.block("compute"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(var_T_reshape_intermediate[v_i0, v_i1, v_i2, v_i3])
            T.writes(compute[v_i0, v_i1, v_i2, v_i3])
            compute[v_i0, v_i1, v_i2, v_i3] = T.sigmoid(var_T_reshape_intermediate[v_i0, v_i1, v_i2, v_i3])
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(640), T.int64(64), T.int64(64)):
        with T.block("T_multiply"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], compute[v_ax0, v_ax1, v_ax2, v_ax3])
            T.writes(var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] * compute[v_ax0, v_ax1, v_ax2, v_ax3]

@T.prim_func
def fused_group_norm4_silu4(lv425: T.Buffer((T.int64(2), T.int64(640), T.int64(32), T.int64(32)), "float32"), unet_down_blocks_2_resnets_0_norm1_weight: T.Buffer((T.int64(640),), "float32"), unet_down_blocks_2_resnets_0_norm1_bias: T.Buffer((T.int64(640),), "float32"), var_T_multiply_intermediate: T.Buffer((T.int64(2), T.int64(640), T.int64(32), T.int64(32)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    T_reshape = T.alloc_buffer((T.int64(2), T.int64(32), T.int64(20), T.int64(32), T.int64(32)))
    A_red_temp_v0 = T.alloc_buffer((T.int64(2), T.int64(32)))
    A_red_temp_v1 = T.alloc_buffer((T.int64(2), T.int64(32)))
    T_reshape_1 = T.alloc_buffer((T.int64(32), T.int64(20)))
    T_reshape_2 = T.alloc_buffer((T.int64(32), T.int64(20)))
    T_group_norm = T.alloc_buffer((T.int64(2), T.int64(32), T.int64(20), T.int64(32), T.int64(32)))
    var_T_reshape_intermediate = T.alloc_buffer((T.int64(2), T.int64(640), T.int64(32), T.int64(32)))
    compute = T.alloc_buffer((T.int64(2), T.int64(640), T.int64(32), T.int64(32)))
    for ax0, ax1, ax2, ax3, ax4 in T.grid(T.int64(2), T.int64(32), T.int64(20), T.int64(32), T.int64(32)):
        with T.block("T_reshape"):
            v_ax0, v_ax1, v_ax2, v_ax3, v_ax4 = T.axis.remap("SSSSS", [ax0, ax1, ax2, ax3, ax4])
            T.reads(lv425[((v_ax1 * T.int64(20) + (v_ax4 // T.int64(32) + v_ax3) // T.int64(32) + v_ax2) // T.int64(640) + v_ax0) % T.int64(2), (v_ax1 * T.int64(20) + (v_ax4 // T.int64(32) + v_ax3) // T.int64(32) + v_ax2) % T.int64(640), (v_ax4 // T.int64(32) + v_ax3) % T.int64(32), v_ax4 % T.int64(32)])
            T.writes(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4])
            T_reshape[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] = lv425[((v_ax1 * T.int64(20) + (v_ax4 // T.int64(32) + v_ax3) // T.int64(32) + v_ax2) // T.int64(640) + v_ax0) % T.int64(2), (v_ax1 * T.int64(20) + (v_ax4 // T.int64(32) + v_ax3) // T.int64(32) + v_ax2) % T.int64(640), (v_ax4 // T.int64(32) + v_ax3) % T.int64(32), v_ax4 % T.int64(32)]
    for ax0, ax1, k2, k3, k4 in T.grid(T.int64(2), T.int64(32), T.int64(20), T.int64(32), T.int64(32)):
        with T.block("A_red_temp"):
            v_ax0, v_ax1, v_k2, v_k3, v_k4 = T.axis.remap("SSRRR", [ax0, ax1, k2, k3, k4])
            T.reads(T_reshape[v_ax0, v_ax1, v_k2, v_k3, v_k4])
            T.writes(A_red_temp_v0[v_ax0, v_ax1], A_red_temp_v1[v_ax0, v_ax1])
            with T.init():
                A_red_temp_v0[v_ax0, v_ax1] = T.float32(0)
                A_red_temp_v1[v_ax0, v_ax1] = T.float32(0)
            v_A_red_temp_v0: T.float32 = A_red_temp_v0[v_ax0, v_ax1] + T_reshape[v_ax0, v_ax1, v_k2, v_k3, v_k4]
            v_A_red_temp_v1: T.float32 = A_red_temp_v1[v_ax0, v_ax1] + T_reshape[v_ax0, v_ax1, v_k2, v_k3, v_k4] * T_reshape[v_ax0, v_ax1, v_k2, v_k3, v_k4]
            A_red_temp_v0[v_ax0, v_ax1] = v_A_red_temp_v0
            A_red_temp_v1[v_ax0, v_ax1] = v_A_red_temp_v1
    for ax0, ax1 in T.grid(T.int64(32), T.int64(20)):
        with T.block("T_reshape_1"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(unet_down_blocks_2_resnets_0_norm1_weight[(v_ax0 * T.int64(20) + v_ax1) % T.int64(640)])
            T.writes(T_reshape_1[v_ax0, v_ax1])
            T_reshape_1[v_ax0, v_ax1] = unet_down_blocks_2_resnets_0_norm1_weight[(v_ax0 * T.int64(20) + v_ax1) % T.int64(640)]
    for ax0, ax1 in T.grid(T.int64(32), T.int64(20)):
        with T.block("T_reshape_2"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(unet_down_blocks_2_resnets_0_norm1_bias[(v_ax0 * T.int64(20) + v_ax1) % T.int64(640)])
            T.writes(T_reshape_2[v_ax0, v_ax1])
            T_reshape_2[v_ax0, v_ax1] = unet_down_blocks_2_resnets_0_norm1_bias[(v_ax0 * T.int64(20) + v_ax1) % T.int64(640)]
    for ax0, ax1, ax2, ax3, ax4 in T.grid(T.int64(2), T.int64(32), T.int64(20), T.int64(32), T.int64(32)):
        with T.block("T_group_norm"):
            v_ax0, v_ax1, v_ax2, v_ax3, v_ax4 = T.axis.remap("SSSSS", [ax0, ax1, ax2, ax3, ax4])
            T.reads(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4], A_red_temp_v0[v_ax0, v_ax1], A_red_temp_v1[v_ax0, v_ax1], T_reshape_1[v_ax1, v_ax2], T_reshape_2[v_ax1, v_ax2])
            T.writes(T_group_norm[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4])
            T_group_norm[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] = (T_reshape[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] - A_red_temp_v0[v_ax0, v_ax1] * T.float32(4.8828125000000003e-05)) * T.rsqrt(A_red_temp_v1[v_ax0, v_ax1] * T.float32(4.8828125000000003e-05) - A_red_temp_v0[v_ax0, v_ax1] * T.float32(4.8828125000000003e-05) * (A_red_temp_v0[v_ax0, v_ax1] * T.float32(4.8828125000000003e-05)) + T.float32(1.0000000000000001e-05)) * T_reshape_1[v_ax1, v_ax2] + T_reshape_2[v_ax1, v_ax2]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(640), T.int64(32), T.int64(32)):
        with T.block("T_reshape_3"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(T_group_norm[(((v_ax3 // T.int64(32) + v_ax2) // T.int64(32) + v_ax1) // T.int64(640) + v_ax0) % T.int64(2), ((v_ax3 // T.int64(32) + v_ax2) // T.int64(32) + v_ax1) % T.int64(640) // T.int64(20), ((v_ax3 // T.int64(32) + v_ax2) // T.int64(32) + v_ax1) % T.int64(20), (v_ax3 // T.int64(32) + v_ax2) % T.int64(32), v_ax3 % T.int64(32)])
            T.writes(var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = T_group_norm[(((v_ax3 // T.int64(32) + v_ax2) // T.int64(32) + v_ax1) // T.int64(640) + v_ax0) % T.int64(2), ((v_ax3 // T.int64(32) + v_ax2) // T.int64(32) + v_ax1) % T.int64(640) // T.int64(20), ((v_ax3 // T.int64(32) + v_ax2) // T.int64(32) + v_ax1) % T.int64(20), (v_ax3 // T.int64(32) + v_ax2) % T.int64(32), v_ax3 % T.int64(32)]
    for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(640), T.int64(32), T.int64(32)):
        with T.block("compute"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(var_T_reshape_intermediate[v_i0, v_i1, v_i2, v_i3])
            T.writes(compute[v_i0, v_i1, v_i2, v_i3])
            compute[v_i0, v_i1, v_i2, v_i3] = T.sigmoid(var_T_reshape_intermediate[v_i0, v_i1, v_i2, v_i3])
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(640), T.int64(32), T.int64(32)):
        with T.block("T_multiply"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], compute[v_ax0, v_ax1, v_ax2, v_ax3])
            T.writes(var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] * compute[v_ax0, v_ax1, v_ax2, v_ax3]

@T.prim_func
def fused_group_norm5_silu5(lv437: T.Buffer((T.int64(2), T.int64(1280), T.int64(32), T.int64(32)), "float32"), unet_down_blocks_2_resnets_0_norm2_weight: T.Buffer((T.int64(1280),), "float32"), unet_down_blocks_2_resnets_0_norm2_bias: T.Buffer((T.int64(1280),), "float32"), var_T_multiply_intermediate: T.Buffer((T.int64(2), T.int64(1280), T.int64(32), T.int64(32)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    T_reshape = T.alloc_buffer((T.int64(2), T.int64(32), T.int64(40), T.int64(32), T.int64(32)))
    A_red_temp_v0 = T.alloc_buffer((T.int64(2), T.int64(32)))
    A_red_temp_v1 = T.alloc_buffer((T.int64(2), T.int64(32)))
    T_reshape_1 = T.alloc_buffer((T.int64(32), T.int64(40)))
    T_reshape_2 = T.alloc_buffer((T.int64(32), T.int64(40)))
    T_group_norm = T.alloc_buffer((T.int64(2), T.int64(32), T.int64(40), T.int64(32), T.int64(32)))
    var_T_reshape_intermediate = T.alloc_buffer((T.int64(2), T.int64(1280), T.int64(32), T.int64(32)))
    compute = T.alloc_buffer((T.int64(2), T.int64(1280), T.int64(32), T.int64(32)))
    for ax0, ax1, ax2, ax3, ax4 in T.grid(T.int64(2), T.int64(32), T.int64(40), T.int64(32), T.int64(32)):
        with T.block("T_reshape"):
            v_ax0, v_ax1, v_ax2, v_ax3, v_ax4 = T.axis.remap("SSSSS", [ax0, ax1, ax2, ax3, ax4])
            T.reads(lv437[((v_ax1 * T.int64(40) + (v_ax4 // T.int64(32) + v_ax3) // T.int64(32) + v_ax2) // T.int64(1280) + v_ax0) % T.int64(2), (v_ax1 * T.int64(40) + (v_ax4 // T.int64(32) + v_ax3) // T.int64(32) + v_ax2) % T.int64(1280), (v_ax4 // T.int64(32) + v_ax3) % T.int64(32), v_ax4 % T.int64(32)])
            T.writes(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4])
            T_reshape[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] = lv437[((v_ax1 * T.int64(40) + (v_ax4 // T.int64(32) + v_ax3) // T.int64(32) + v_ax2) // T.int64(1280) + v_ax0) % T.int64(2), (v_ax1 * T.int64(40) + (v_ax4 // T.int64(32) + v_ax3) // T.int64(32) + v_ax2) % T.int64(1280), (v_ax4 // T.int64(32) + v_ax3) % T.int64(32), v_ax4 % T.int64(32)]
    for ax0, ax1, k2, k3, k4 in T.grid(T.int64(2), T.int64(32), T.int64(40), T.int64(32), T.int64(32)):
        with T.block("A_red_temp"):
            v_ax0, v_ax1, v_k2, v_k3, v_k4 = T.axis.remap("SSRRR", [ax0, ax1, k2, k3, k4])
            T.reads(T_reshape[v_ax0, v_ax1, v_k2, v_k3, v_k4])
            T.writes(A_red_temp_v0[v_ax0, v_ax1], A_red_temp_v1[v_ax0, v_ax1])
            with T.init():
                A_red_temp_v0[v_ax0, v_ax1] = T.float32(0)
                A_red_temp_v1[v_ax0, v_ax1] = T.float32(0)
            v_A_red_temp_v0: T.float32 = A_red_temp_v0[v_ax0, v_ax1] + T_reshape[v_ax0, v_ax1, v_k2, v_k3, v_k4]
            v_A_red_temp_v1: T.float32 = A_red_temp_v1[v_ax0, v_ax1] + T_reshape[v_ax0, v_ax1, v_k2, v_k3, v_k4] * T_reshape[v_ax0, v_ax1, v_k2, v_k3, v_k4]
            A_red_temp_v0[v_ax0, v_ax1] = v_A_red_temp_v0
            A_red_temp_v1[v_ax0, v_ax1] = v_A_red_temp_v1
    for ax0, ax1 in T.grid(T.int64(32), T.int64(40)):
        with T.block("T_reshape_1"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(unet_down_blocks_2_resnets_0_norm2_weight[(v_ax0 * T.int64(40) + v_ax1) % T.int64(1280)])
            T.writes(T_reshape_1[v_ax0, v_ax1])
            T_reshape_1[v_ax0, v_ax1] = unet_down_blocks_2_resnets_0_norm2_weight[(v_ax0 * T.int64(40) + v_ax1) % T.int64(1280)]
    for ax0, ax1 in T.grid(T.int64(32), T.int64(40)):
        with T.block("T_reshape_2"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(unet_down_blocks_2_resnets_0_norm2_bias[(v_ax0 * T.int64(40) + v_ax1) % T.int64(1280)])
            T.writes(T_reshape_2[v_ax0, v_ax1])
            T_reshape_2[v_ax0, v_ax1] = unet_down_blocks_2_resnets_0_norm2_bias[(v_ax0 * T.int64(40) + v_ax1) % T.int64(1280)]
    for ax0, ax1, ax2, ax3, ax4 in T.grid(T.int64(2), T.int64(32), T.int64(40), T.int64(32), T.int64(32)):
        with T.block("T_group_norm"):
            v_ax0, v_ax1, v_ax2, v_ax3, v_ax4 = T.axis.remap("SSSSS", [ax0, ax1, ax2, ax3, ax4])
            T.reads(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4], A_red_temp_v0[v_ax0, v_ax1], A_red_temp_v1[v_ax0, v_ax1], T_reshape_1[v_ax1, v_ax2], T_reshape_2[v_ax1, v_ax2])
            T.writes(T_group_norm[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4])
            T_group_norm[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] = (T_reshape[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] - A_red_temp_v0[v_ax0, v_ax1] * T.float32(2.4414062500000001e-05)) * T.rsqrt(A_red_temp_v1[v_ax0, v_ax1] * T.float32(2.4414062500000001e-05) - A_red_temp_v0[v_ax0, v_ax1] * T.float32(2.4414062500000001e-05) * (A_red_temp_v0[v_ax0, v_ax1] * T.float32(2.4414062500000001e-05)) + T.float32(1.0000000000000001e-05)) * T_reshape_1[v_ax1, v_ax2] + T_reshape_2[v_ax1, v_ax2]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(1280), T.int64(32), T.int64(32)):
        with T.block("T_reshape_3"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(T_group_norm[(((v_ax3 // T.int64(32) + v_ax2) // T.int64(32) + v_ax1) // T.int64(1280) + v_ax0) % T.int64(2), ((v_ax3 // T.int64(32) + v_ax2) // T.int64(32) + v_ax1) % T.int64(1280) // T.int64(40), ((v_ax3 // T.int64(32) + v_ax2) // T.int64(32) + v_ax1) % T.int64(40), (v_ax3 // T.int64(32) + v_ax2) % T.int64(32), v_ax3 % T.int64(32)])
            T.writes(var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = T_group_norm[(((v_ax3 // T.int64(32) + v_ax2) // T.int64(32) + v_ax1) // T.int64(1280) + v_ax0) % T.int64(2), ((v_ax3 // T.int64(32) + v_ax2) // T.int64(32) + v_ax1) % T.int64(1280) // T.int64(40), ((v_ax3 // T.int64(32) + v_ax2) // T.int64(32) + v_ax1) % T.int64(40), (v_ax3 // T.int64(32) + v_ax2) % T.int64(32), v_ax3 % T.int64(32)]
    for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(1280), T.int64(32), T.int64(32)):
        with T.block("compute"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(var_T_reshape_intermediate[v_i0, v_i1, v_i2, v_i3])
            T.writes(compute[v_i0, v_i1, v_i2, v_i3])
            compute[v_i0, v_i1, v_i2, v_i3] = T.sigmoid(var_T_reshape_intermediate[v_i0, v_i1, v_i2, v_i3])
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(1280), T.int64(32), T.int64(32)):
        with T.block("T_multiply"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], compute[v_ax0, v_ax1, v_ax2, v_ax3])
            T.writes(var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] * compute[v_ax0, v_ax1, v_ax2, v_ax3]

@T.prim_func
def fused_group_norm7_silu6(lv2551: T.Buffer((T.int64(2), T.int64(2560), T.int64(32), T.int64(32)), "float32"), unet_up_blocks_0_resnets_0_norm1_weight: T.Buffer((T.int64(2560),), "float32"), unet_up_blocks_0_resnets_0_norm1_bias: T.Buffer((T.int64(2560),), "float32"), var_T_multiply_intermediate: T.Buffer((T.int64(2), T.int64(2560), T.int64(32), T.int64(32)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    T_reshape = T.alloc_buffer((T.int64(2), T.int64(32), T.int64(80), T.int64(32), T.int64(32)))
    A_red_temp_v0 = T.alloc_buffer((T.int64(2), T.int64(32)))
    A_red_temp_v1 = T.alloc_buffer((T.int64(2), T.int64(32)))
    T_reshape_1 = T.alloc_buffer((T.int64(32), T.int64(80)))
    T_reshape_2 = T.alloc_buffer((T.int64(32), T.int64(80)))
    T_group_norm = T.alloc_buffer((T.int64(2), T.int64(32), T.int64(80), T.int64(32), T.int64(32)))
    var_T_reshape_intermediate = T.alloc_buffer((T.int64(2), T.int64(2560), T.int64(32), T.int64(32)))
    compute = T.alloc_buffer((T.int64(2), T.int64(2560), T.int64(32), T.int64(32)))
    for ax0, ax1, ax2, ax3, ax4 in T.grid(T.int64(2), T.int64(32), T.int64(80), T.int64(32), T.int64(32)):
        with T.block("T_reshape"):
            v_ax0, v_ax1, v_ax2, v_ax3, v_ax4 = T.axis.remap("SSSSS", [ax0, ax1, ax2, ax3, ax4])
            T.reads(lv2551[((v_ax1 * T.int64(80) + (v_ax4 // T.int64(32) + v_ax3) // T.int64(32) + v_ax2) // T.int64(2560) + v_ax0) % T.int64(2), (v_ax1 * T.int64(80) + (v_ax4 // T.int64(32) + v_ax3) // T.int64(32) + v_ax2) % T.int64(2560), (v_ax4 // T.int64(32) + v_ax3) % T.int64(32), v_ax4 % T.int64(32)])
            T.writes(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4])
            T_reshape[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] = lv2551[((v_ax1 * T.int64(80) + (v_ax4 // T.int64(32) + v_ax3) // T.int64(32) + v_ax2) // T.int64(2560) + v_ax0) % T.int64(2), (v_ax1 * T.int64(80) + (v_ax4 // T.int64(32) + v_ax3) // T.int64(32) + v_ax2) % T.int64(2560), (v_ax4 // T.int64(32) + v_ax3) % T.int64(32), v_ax4 % T.int64(32)]
    for ax0, ax1, k2, k3, k4 in T.grid(T.int64(2), T.int64(32), T.int64(80), T.int64(32), T.int64(32)):
        with T.block("A_red_temp"):
            v_ax0, v_ax1, v_k2, v_k3, v_k4 = T.axis.remap("SSRRR", [ax0, ax1, k2, k3, k4])
            T.reads(T_reshape[v_ax0, v_ax1, v_k2, v_k3, v_k4])
            T.writes(A_red_temp_v0[v_ax0, v_ax1], A_red_temp_v1[v_ax0, v_ax1])
            with T.init():
                A_red_temp_v0[v_ax0, v_ax1] = T.float32(0)
                A_red_temp_v1[v_ax0, v_ax1] = T.float32(0)
            v_A_red_temp_v0: T.float32 = A_red_temp_v0[v_ax0, v_ax1] + T_reshape[v_ax0, v_ax1, v_k2, v_k3, v_k4]
            v_A_red_temp_v1: T.float32 = A_red_temp_v1[v_ax0, v_ax1] + T_reshape[v_ax0, v_ax1, v_k2, v_k3, v_k4] * T_reshape[v_ax0, v_ax1, v_k2, v_k3, v_k4]
            A_red_temp_v0[v_ax0, v_ax1] = v_A_red_temp_v0
            A_red_temp_v1[v_ax0, v_ax1] = v_A_red_temp_v1
    for ax0, ax1 in T.grid(T.int64(32), T.int64(80)):
        with T.block("T_reshape_1"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(unet_up_blocks_0_resnets_0_norm1_weight[(v_ax0 * T.int64(80) + v_ax1) % T.int64(2560)])
            T.writes(T_reshape_1[v_ax0, v_ax1])
            T_reshape_1[v_ax0, v_ax1] = unet_up_blocks_0_resnets_0_norm1_weight[(v_ax0 * T.int64(80) + v_ax1) % T.int64(2560)]
    for ax0, ax1 in T.grid(T.int64(32), T.int64(80)):
        with T.block("T_reshape_2"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(unet_up_blocks_0_resnets_0_norm1_bias[(v_ax0 * T.int64(80) + v_ax1) % T.int64(2560)])
            T.writes(T_reshape_2[v_ax0, v_ax1])
            T_reshape_2[v_ax0, v_ax1] = unet_up_blocks_0_resnets_0_norm1_bias[(v_ax0 * T.int64(80) + v_ax1) % T.int64(2560)]
    for ax0, ax1, ax2, ax3, ax4 in T.grid(T.int64(2), T.int64(32), T.int64(80), T.int64(32), T.int64(32)):
        with T.block("T_group_norm"):
            v_ax0, v_ax1, v_ax2, v_ax3, v_ax4 = T.axis.remap("SSSSS", [ax0, ax1, ax2, ax3, ax4])
            T.reads(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4], A_red_temp_v0[v_ax0, v_ax1], A_red_temp_v1[v_ax0, v_ax1], T_reshape_1[v_ax1, v_ax2], T_reshape_2[v_ax1, v_ax2])
            T.writes(T_group_norm[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4])
            T_group_norm[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] = (T_reshape[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] - A_red_temp_v0[v_ax0, v_ax1] * T.float32(1.2207031250000001e-05)) * T.rsqrt(A_red_temp_v1[v_ax0, v_ax1] * T.float32(1.2207031250000001e-05) - A_red_temp_v0[v_ax0, v_ax1] * T.float32(1.2207031250000001e-05) * (A_red_temp_v0[v_ax0, v_ax1] * T.float32(1.2207031250000001e-05)) + T.float32(1.0000000000000001e-05)) * T_reshape_1[v_ax1, v_ax2] + T_reshape_2[v_ax1, v_ax2]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(2560), T.int64(32), T.int64(32)):
        with T.block("T_reshape_3"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(T_group_norm[(((v_ax3 // T.int64(32) + v_ax2) // T.int64(32) + v_ax1) // T.int64(2560) + v_ax0) % T.int64(2), ((v_ax3 // T.int64(32) + v_ax2) // T.int64(32) + v_ax1) % T.int64(2560) // T.int64(80), ((v_ax3 // T.int64(32) + v_ax2) // T.int64(32) + v_ax1) % T.int64(80), (v_ax3 // T.int64(32) + v_ax2) % T.int64(32), v_ax3 % T.int64(32)])
            T.writes(var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = T_group_norm[(((v_ax3 // T.int64(32) + v_ax2) // T.int64(32) + v_ax1) // T.int64(2560) + v_ax0) % T.int64(2), ((v_ax3 // T.int64(32) + v_ax2) // T.int64(32) + v_ax1) % T.int64(2560) // T.int64(80), ((v_ax3 // T.int64(32) + v_ax2) // T.int64(32) + v_ax1) % T.int64(80), (v_ax3 // T.int64(32) + v_ax2) % T.int64(32), v_ax3 % T.int64(32)]
    for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(2560), T.int64(32), T.int64(32)):
        with T.block("compute"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(var_T_reshape_intermediate[v_i0, v_i1, v_i2, v_i3])
            T.writes(compute[v_i0, v_i1, v_i2, v_i3])
            compute[v_i0, v_i1, v_i2, v_i3] = T.sigmoid(var_T_reshape_intermediate[v_i0, v_i1, v_i2, v_i3])
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(2560), T.int64(32), T.int64(32)):
        with T.block("T_multiply"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], compute[v_ax0, v_ax1, v_ax2, v_ax3])
            T.writes(var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] * compute[v_ax0, v_ax1, v_ax2, v_ax3]

@T.prim_func
def fused_group_norm8_silu7(lv3961: T.Buffer((T.int64(2), T.int64(1920), T.int64(32), T.int64(32)), "float32"), unet_up_blocks_0_resnets_2_norm1_weight: T.Buffer((T.int64(1920),), "float32"), unet_up_blocks_0_resnets_2_norm1_bias: T.Buffer((T.int64(1920),), "float32"), var_T_multiply_intermediate: T.Buffer((T.int64(2), T.int64(1920), T.int64(32), T.int64(32)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    T_reshape = T.alloc_buffer((T.int64(2), T.int64(32), T.int64(60), T.int64(32), T.int64(32)))
    A_red_temp_v0 = T.alloc_buffer((T.int64(2), T.int64(32)))
    A_red_temp_v1 = T.alloc_buffer((T.int64(2), T.int64(32)))
    T_reshape_1 = T.alloc_buffer((T.int64(32), T.int64(60)))
    T_reshape_2 = T.alloc_buffer((T.int64(32), T.int64(60)))
    T_group_norm = T.alloc_buffer((T.int64(2), T.int64(32), T.int64(60), T.int64(32), T.int64(32)))
    var_T_reshape_intermediate = T.alloc_buffer((T.int64(2), T.int64(1920), T.int64(32), T.int64(32)))
    compute = T.alloc_buffer((T.int64(2), T.int64(1920), T.int64(32), T.int64(32)))
    for ax0, ax1, ax2, ax3, ax4 in T.grid(T.int64(2), T.int64(32), T.int64(60), T.int64(32), T.int64(32)):
        with T.block("T_reshape"):
            v_ax0, v_ax1, v_ax2, v_ax3, v_ax4 = T.axis.remap("SSSSS", [ax0, ax1, ax2, ax3, ax4])
            T.reads(lv3961[((v_ax1 * T.int64(60) + (v_ax4 // T.int64(32) + v_ax3) // T.int64(32) + v_ax2) // T.int64(1920) + v_ax0) % T.int64(2), (v_ax1 * T.int64(60) + (v_ax4 // T.int64(32) + v_ax3) // T.int64(32) + v_ax2) % T.int64(1920), (v_ax4 // T.int64(32) + v_ax3) % T.int64(32), v_ax4 % T.int64(32)])
            T.writes(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4])
            T_reshape[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] = lv3961[((v_ax1 * T.int64(60) + (v_ax4 // T.int64(32) + v_ax3) // T.int64(32) + v_ax2) // T.int64(1920) + v_ax0) % T.int64(2), (v_ax1 * T.int64(60) + (v_ax4 // T.int64(32) + v_ax3) // T.int64(32) + v_ax2) % T.int64(1920), (v_ax4 // T.int64(32) + v_ax3) % T.int64(32), v_ax4 % T.int64(32)]
    for ax0, ax1, k2, k3, k4 in T.grid(T.int64(2), T.int64(32), T.int64(60), T.int64(32), T.int64(32)):
        with T.block("A_red_temp"):
            v_ax0, v_ax1, v_k2, v_k3, v_k4 = T.axis.remap("SSRRR", [ax0, ax1, k2, k3, k4])
            T.reads(T_reshape[v_ax0, v_ax1, v_k2, v_k3, v_k4])
            T.writes(A_red_temp_v0[v_ax0, v_ax1], A_red_temp_v1[v_ax0, v_ax1])
            with T.init():
                A_red_temp_v0[v_ax0, v_ax1] = T.float32(0)
                A_red_temp_v1[v_ax0, v_ax1] = T.float32(0)
            v_A_red_temp_v0: T.float32 = A_red_temp_v0[v_ax0, v_ax1] + T_reshape[v_ax0, v_ax1, v_k2, v_k3, v_k4]
            v_A_red_temp_v1: T.float32 = A_red_temp_v1[v_ax0, v_ax1] + T_reshape[v_ax0, v_ax1, v_k2, v_k3, v_k4] * T_reshape[v_ax0, v_ax1, v_k2, v_k3, v_k4]
            A_red_temp_v0[v_ax0, v_ax1] = v_A_red_temp_v0
            A_red_temp_v1[v_ax0, v_ax1] = v_A_red_temp_v1
    for ax0, ax1 in T.grid(T.int64(32), T.int64(60)):
        with T.block("T_reshape_1"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(unet_up_blocks_0_resnets_2_norm1_weight[(v_ax0 * T.int64(60) + v_ax1) % T.int64(1920)])
            T.writes(T_reshape_1[v_ax0, v_ax1])
            T_reshape_1[v_ax0, v_ax1] = unet_up_blocks_0_resnets_2_norm1_weight[(v_ax0 * T.int64(60) + v_ax1) % T.int64(1920)]
    for ax0, ax1 in T.grid(T.int64(32), T.int64(60)):
        with T.block("T_reshape_2"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(unet_up_blocks_0_resnets_2_norm1_bias[(v_ax0 * T.int64(60) + v_ax1) % T.int64(1920)])
            T.writes(T_reshape_2[v_ax0, v_ax1])
            T_reshape_2[v_ax0, v_ax1] = unet_up_blocks_0_resnets_2_norm1_bias[(v_ax0 * T.int64(60) + v_ax1) % T.int64(1920)]
    for ax0, ax1, ax2, ax3, ax4 in T.grid(T.int64(2), T.int64(32), T.int64(60), T.int64(32), T.int64(32)):
        with T.block("T_group_norm"):
            v_ax0, v_ax1, v_ax2, v_ax3, v_ax4 = T.axis.remap("SSSSS", [ax0, ax1, ax2, ax3, ax4])
            T.reads(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4], A_red_temp_v0[v_ax0, v_ax1], A_red_temp_v1[v_ax0, v_ax1], T_reshape_1[v_ax1, v_ax2], T_reshape_2[v_ax1, v_ax2])
            T.writes(T_group_norm[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4])
            T_group_norm[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] = (T_reshape[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] - A_red_temp_v0[v_ax0, v_ax1] * T.float32(1.6276041666666666e-05)) * T.rsqrt(A_red_temp_v1[v_ax0, v_ax1] * T.float32(1.6276041666666666e-05) - A_red_temp_v0[v_ax0, v_ax1] * T.float32(1.6276041666666666e-05) * (A_red_temp_v0[v_ax0, v_ax1] * T.float32(1.6276041666666666e-05)) + T.float32(1.0000000000000001e-05)) * T_reshape_1[v_ax1, v_ax2] + T_reshape_2[v_ax1, v_ax2]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(1920), T.int64(32), T.int64(32)):
        with T.block("T_reshape_3"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(T_group_norm[(((v_ax3 // T.int64(32) + v_ax2) // T.int64(32) + v_ax1) // T.int64(1920) + v_ax0) % T.int64(2), ((v_ax3 // T.int64(32) + v_ax2) // T.int64(32) + v_ax1) % T.int64(1920) // T.int64(60), ((v_ax3 // T.int64(32) + v_ax2) // T.int64(32) + v_ax1) % T.int64(60), (v_ax3 // T.int64(32) + v_ax2) % T.int64(32), v_ax3 % T.int64(32)])
            T.writes(var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = T_group_norm[(((v_ax3 // T.int64(32) + v_ax2) // T.int64(32) + v_ax1) // T.int64(1920) + v_ax0) % T.int64(2), ((v_ax3 // T.int64(32) + v_ax2) // T.int64(32) + v_ax1) % T.int64(1920) // T.int64(60), ((v_ax3 // T.int64(32) + v_ax2) // T.int64(32) + v_ax1) % T.int64(60), (v_ax3 // T.int64(32) + v_ax2) % T.int64(32), v_ax3 % T.int64(32)]
    for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(1920), T.int64(32), T.int64(32)):
        with T.block("compute"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(var_T_reshape_intermediate[v_i0, v_i1, v_i2, v_i3])
            T.writes(compute[v_i0, v_i1, v_i2, v_i3])
            compute[v_i0, v_i1, v_i2, v_i3] = T.sigmoid(var_T_reshape_intermediate[v_i0, v_i1, v_i2, v_i3])
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(1920), T.int64(32), T.int64(32)):
        with T.block("T_multiply"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], compute[v_ax0, v_ax1, v_ax2, v_ax3])
            T.writes(var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] * compute[v_ax0, v_ax1, v_ax2, v_ax3]

@T.prim_func
def fused_group_norm9_silu8(lv4670: T.Buffer((T.int64(2), T.int64(1920), T.int64(64), T.int64(64)), "float32"), unet_up_blocks_1_resnets_0_norm1_weight: T.Buffer((T.int64(1920),), "float32"), unet_up_blocks_1_resnets_0_norm1_bias: T.Buffer((T.int64(1920),), "float32"), var_T_multiply_intermediate: T.Buffer((T.int64(2), T.int64(1920), T.int64(64), T.int64(64)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    T_reshape = T.alloc_buffer((T.int64(2), T.int64(32), T.int64(60), T.int64(64), T.int64(64)))
    A_red_temp_v0 = T.alloc_buffer((T.int64(2), T.int64(32)))
    A_red_temp_v1 = T.alloc_buffer((T.int64(2), T.int64(32)))
    T_reshape_1 = T.alloc_buffer((T.int64(32), T.int64(60)))
    T_reshape_2 = T.alloc_buffer((T.int64(32), T.int64(60)))
    T_group_norm = T.alloc_buffer((T.int64(2), T.int64(32), T.int64(60), T.int64(64), T.int64(64)))
    var_T_reshape_intermediate = T.alloc_buffer((T.int64(2), T.int64(1920), T.int64(64), T.int64(64)))
    compute = T.alloc_buffer((T.int64(2), T.int64(1920), T.int64(64), T.int64(64)))
    for ax0, ax1, ax2, ax3, ax4 in T.grid(T.int64(2), T.int64(32), T.int64(60), T.int64(64), T.int64(64)):
        with T.block("T_reshape"):
            v_ax0, v_ax1, v_ax2, v_ax3, v_ax4 = T.axis.remap("SSSSS", [ax0, ax1, ax2, ax3, ax4])
            T.reads(lv4670[((v_ax1 * T.int64(60) + (v_ax4 // T.int64(64) + v_ax3) // T.int64(64) + v_ax2) // T.int64(1920) + v_ax0) % T.int64(2), (v_ax1 * T.int64(60) + (v_ax4 // T.int64(64) + v_ax3) // T.int64(64) + v_ax2) % T.int64(1920), (v_ax4 // T.int64(64) + v_ax3) % T.int64(64), v_ax4 % T.int64(64)])
            T.writes(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4])
            T_reshape[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] = lv4670[((v_ax1 * T.int64(60) + (v_ax4 // T.int64(64) + v_ax3) // T.int64(64) + v_ax2) // T.int64(1920) + v_ax0) % T.int64(2), (v_ax1 * T.int64(60) + (v_ax4 // T.int64(64) + v_ax3) // T.int64(64) + v_ax2) % T.int64(1920), (v_ax4 // T.int64(64) + v_ax3) % T.int64(64), v_ax4 % T.int64(64)]
    for ax0, ax1, k2, k3, k4 in T.grid(T.int64(2), T.int64(32), T.int64(60), T.int64(64), T.int64(64)):
        with T.block("A_red_temp"):
            v_ax0, v_ax1, v_k2, v_k3, v_k4 = T.axis.remap("SSRRR", [ax0, ax1, k2, k3, k4])
            T.reads(T_reshape[v_ax0, v_ax1, v_k2, v_k3, v_k4])
            T.writes(A_red_temp_v0[v_ax0, v_ax1], A_red_temp_v1[v_ax0, v_ax1])
            with T.init():
                A_red_temp_v0[v_ax0, v_ax1] = T.float32(0)
                A_red_temp_v1[v_ax0, v_ax1] = T.float32(0)
            v_A_red_temp_v0: T.float32 = A_red_temp_v0[v_ax0, v_ax1] + T_reshape[v_ax0, v_ax1, v_k2, v_k3, v_k4]
            v_A_red_temp_v1: T.float32 = A_red_temp_v1[v_ax0, v_ax1] + T_reshape[v_ax0, v_ax1, v_k2, v_k3, v_k4] * T_reshape[v_ax0, v_ax1, v_k2, v_k3, v_k4]
            A_red_temp_v0[v_ax0, v_ax1] = v_A_red_temp_v0
            A_red_temp_v1[v_ax0, v_ax1] = v_A_red_temp_v1
    for ax0, ax1 in T.grid(T.int64(32), T.int64(60)):
        with T.block("T_reshape_1"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(unet_up_blocks_1_resnets_0_norm1_weight[(v_ax0 * T.int64(60) + v_ax1) % T.int64(1920)])
            T.writes(T_reshape_1[v_ax0, v_ax1])
            T_reshape_1[v_ax0, v_ax1] = unet_up_blocks_1_resnets_0_norm1_weight[(v_ax0 * T.int64(60) + v_ax1) % T.int64(1920)]
    for ax0, ax1 in T.grid(T.int64(32), T.int64(60)):
        with T.block("T_reshape_2"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(unet_up_blocks_1_resnets_0_norm1_bias[(v_ax0 * T.int64(60) + v_ax1) % T.int64(1920)])
            T.writes(T_reshape_2[v_ax0, v_ax1])
            T_reshape_2[v_ax0, v_ax1] = unet_up_blocks_1_resnets_0_norm1_bias[(v_ax0 * T.int64(60) + v_ax1) % T.int64(1920)]
    for ax0, ax1, ax2, ax3, ax4 in T.grid(T.int64(2), T.int64(32), T.int64(60), T.int64(64), T.int64(64)):
        with T.block("T_group_norm"):
            v_ax0, v_ax1, v_ax2, v_ax3, v_ax4 = T.axis.remap("SSSSS", [ax0, ax1, ax2, ax3, ax4])
            T.reads(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4], A_red_temp_v0[v_ax0, v_ax1], A_red_temp_v1[v_ax0, v_ax1], T_reshape_1[v_ax1, v_ax2], T_reshape_2[v_ax1, v_ax2])
            T.writes(T_group_norm[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4])
            T_group_norm[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] = (T_reshape[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] - A_red_temp_v0[v_ax0, v_ax1] * T.float32(4.0690104166666666e-06)) * T.rsqrt(A_red_temp_v1[v_ax0, v_ax1] * T.float32(4.0690104166666666e-06) - A_red_temp_v0[v_ax0, v_ax1] * T.float32(4.0690104166666666e-06) * (A_red_temp_v0[v_ax0, v_ax1] * T.float32(4.0690104166666666e-06)) + T.float32(1.0000000000000001e-05)) * T_reshape_1[v_ax1, v_ax2] + T_reshape_2[v_ax1, v_ax2]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(1920), T.int64(64), T.int64(64)):
        with T.block("T_reshape_3"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(T_group_norm[(((v_ax3 // T.int64(64) + v_ax2) // T.int64(64) + v_ax1) // T.int64(1920) + v_ax0) % T.int64(2), ((v_ax3 // T.int64(64) + v_ax2) // T.int64(64) + v_ax1) % T.int64(1920) // T.int64(60), ((v_ax3 // T.int64(64) + v_ax2) // T.int64(64) + v_ax1) % T.int64(60), (v_ax3 // T.int64(64) + v_ax2) % T.int64(64), v_ax3 % T.int64(64)])
            T.writes(var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = T_group_norm[(((v_ax3 // T.int64(64) + v_ax2) // T.int64(64) + v_ax1) // T.int64(1920) + v_ax0) % T.int64(2), ((v_ax3 // T.int64(64) + v_ax2) // T.int64(64) + v_ax1) % T.int64(1920) // T.int64(60), ((v_ax3 // T.int64(64) + v_ax2) // T.int64(64) + v_ax1) % T.int64(60), (v_ax3 // T.int64(64) + v_ax2) % T.int64(64), v_ax3 % T.int64(64)]
    for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(1920), T.int64(64), T.int64(64)):
        with T.block("compute"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(var_T_reshape_intermediate[v_i0, v_i1, v_i2, v_i3])
            T.writes(compute[v_i0, v_i1, v_i2, v_i3])
            compute[v_i0, v_i1, v_i2, v_i3] = T.sigmoid(var_T_reshape_intermediate[v_i0, v_i1, v_i2, v_i3])
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(1920), T.int64(64), T.int64(64)):
        with T.block("T_multiply"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], compute[v_ax0, v_ax1, v_ax2, v_ax3])
            T.writes(var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] * compute[v_ax0, v_ax1, v_ax2, v_ax3]

@T.prim_func
def fused_group_norm_silu1(lv48: T.Buffer((T.int64(2), T.int64(320), T.int64(128), T.int64(128)), "float32"), unet_down_blocks_0_resnets_0_norm1_weight: T.Buffer((T.int64(320),), "float32"), unet_down_blocks_0_resnets_0_norm1_bias: T.Buffer((T.int64(320),), "float32"), var_T_multiply_intermediate: T.Buffer((T.int64(2), T.int64(320), T.int64(128), T.int64(128)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    T_reshape = T.alloc_buffer((T.int64(2), T.int64(32), T.int64(10), T.int64(128), T.int64(128)))
    A_red_temp_v0 = T.alloc_buffer((T.int64(2), T.int64(32)))
    A_red_temp_v1 = T.alloc_buffer((T.int64(2), T.int64(32)))
    T_reshape_1 = T.alloc_buffer((T.int64(32), T.int64(10)))
    T_reshape_2 = T.alloc_buffer((T.int64(32), T.int64(10)))
    T_group_norm = T.alloc_buffer((T.int64(2), T.int64(32), T.int64(10), T.int64(128), T.int64(128)))
    var_T_reshape_intermediate = T.alloc_buffer((T.int64(2), T.int64(320), T.int64(128), T.int64(128)))
    compute = T.alloc_buffer((T.int64(2), T.int64(320), T.int64(128), T.int64(128)))
    for ax0, ax1, ax2, ax3, ax4 in T.grid(T.int64(2), T.int64(32), T.int64(10), T.int64(128), T.int64(128)):
        with T.block("T_reshape"):
            v_ax0, v_ax1, v_ax2, v_ax3, v_ax4 = T.axis.remap("SSSSS", [ax0, ax1, ax2, ax3, ax4])
            T.reads(lv48[((v_ax1 * T.int64(10) + (v_ax4 // T.int64(128) + v_ax3) // T.int64(128) + v_ax2) // T.int64(320) + v_ax0) % T.int64(2), (v_ax1 * T.int64(10) + (v_ax4 // T.int64(128) + v_ax3) // T.int64(128) + v_ax2) % T.int64(320), (v_ax4 // T.int64(128) + v_ax3) % T.int64(128), v_ax4 % T.int64(128)])
            T.writes(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4])
            T_reshape[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] = lv48[((v_ax1 * T.int64(10) + (v_ax4 // T.int64(128) + v_ax3) // T.int64(128) + v_ax2) // T.int64(320) + v_ax0) % T.int64(2), (v_ax1 * T.int64(10) + (v_ax4 // T.int64(128) + v_ax3) // T.int64(128) + v_ax2) % T.int64(320), (v_ax4 // T.int64(128) + v_ax3) % T.int64(128), v_ax4 % T.int64(128)]
    for ax0, ax1, k2, k3, k4 in T.grid(T.int64(2), T.int64(32), T.int64(10), T.int64(128), T.int64(128)):
        with T.block("A_red_temp"):
            v_ax0, v_ax1, v_k2, v_k3, v_k4 = T.axis.remap("SSRRR", [ax0, ax1, k2, k3, k4])
            T.reads(T_reshape[v_ax0, v_ax1, v_k2, v_k3, v_k4])
            T.writes(A_red_temp_v0[v_ax0, v_ax1], A_red_temp_v1[v_ax0, v_ax1])
            with T.init():
                A_red_temp_v0[v_ax0, v_ax1] = T.float32(0)
                A_red_temp_v1[v_ax0, v_ax1] = T.float32(0)
            v_A_red_temp_v0: T.float32 = A_red_temp_v0[v_ax0, v_ax1] + T_reshape[v_ax0, v_ax1, v_k2, v_k3, v_k4]
            v_A_red_temp_v1: T.float32 = A_red_temp_v1[v_ax0, v_ax1] + T_reshape[v_ax0, v_ax1, v_k2, v_k3, v_k4] * T_reshape[v_ax0, v_ax1, v_k2, v_k3, v_k4]
            A_red_temp_v0[v_ax0, v_ax1] = v_A_red_temp_v0
            A_red_temp_v1[v_ax0, v_ax1] = v_A_red_temp_v1
    for ax0, ax1 in T.grid(T.int64(32), T.int64(10)):
        with T.block("T_reshape_1"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(unet_down_blocks_0_resnets_0_norm1_weight[(v_ax0 * T.int64(10) + v_ax1) % T.int64(320)])
            T.writes(T_reshape_1[v_ax0, v_ax1])
            T_reshape_1[v_ax0, v_ax1] = unet_down_blocks_0_resnets_0_norm1_weight[(v_ax0 * T.int64(10) + v_ax1) % T.int64(320)]
    for ax0, ax1 in T.grid(T.int64(32), T.int64(10)):
        with T.block("T_reshape_2"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(unet_down_blocks_0_resnets_0_norm1_bias[(v_ax0 * T.int64(10) + v_ax1) % T.int64(320)])
            T.writes(T_reshape_2[v_ax0, v_ax1])
            T_reshape_2[v_ax0, v_ax1] = unet_down_blocks_0_resnets_0_norm1_bias[(v_ax0 * T.int64(10) + v_ax1) % T.int64(320)]
    for ax0, ax1, ax2, ax3, ax4 in T.grid(T.int64(2), T.int64(32), T.int64(10), T.int64(128), T.int64(128)):
        with T.block("T_group_norm"):
            v_ax0, v_ax1, v_ax2, v_ax3, v_ax4 = T.axis.remap("SSSSS", [ax0, ax1, ax2, ax3, ax4])
            T.reads(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4], A_red_temp_v0[v_ax0, v_ax1], A_red_temp_v1[v_ax0, v_ax1], T_reshape_1[v_ax1, v_ax2], T_reshape_2[v_ax1, v_ax2])
            T.writes(T_group_norm[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4])
            T_group_norm[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] = (T_reshape[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] - A_red_temp_v0[v_ax0, v_ax1] * T.float32(6.1035156250000003e-06)) * T.rsqrt(A_red_temp_v1[v_ax0, v_ax1] * T.float32(6.1035156250000003e-06) - A_red_temp_v0[v_ax0, v_ax1] * T.float32(6.1035156250000003e-06) * (A_red_temp_v0[v_ax0, v_ax1] * T.float32(6.1035156250000003e-06)) + T.float32(1.0000000000000001e-05)) * T_reshape_1[v_ax1, v_ax2] + T_reshape_2[v_ax1, v_ax2]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(320), T.int64(128), T.int64(128)):
        with T.block("T_reshape_3"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(T_group_norm[(((v_ax3 // T.int64(128) + v_ax2) // T.int64(128) + v_ax1) // T.int64(320) + v_ax0) % T.int64(2), ((v_ax3 // T.int64(128) + v_ax2) // T.int64(128) + v_ax1) % T.int64(320) // T.int64(10), ((v_ax3 // T.int64(128) + v_ax2) // T.int64(128) + v_ax1) % T.int64(10), (v_ax3 // T.int64(128) + v_ax2) % T.int64(128), v_ax3 % T.int64(128)])
            T.writes(var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = T_group_norm[(((v_ax3 // T.int64(128) + v_ax2) // T.int64(128) + v_ax1) // T.int64(320) + v_ax0) % T.int64(2), ((v_ax3 // T.int64(128) + v_ax2) // T.int64(128) + v_ax1) % T.int64(320) // T.int64(10), ((v_ax3 // T.int64(128) + v_ax2) // T.int64(128) + v_ax1) % T.int64(10), (v_ax3 // T.int64(128) + v_ax2) % T.int64(128), v_ax3 % T.int64(128)]
    for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(320), T.int64(128), T.int64(128)):
        with T.block("compute"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(var_T_reshape_intermediate[v_i0, v_i1, v_i2, v_i3])
            T.writes(compute[v_i0, v_i1, v_i2, v_i3])
            compute[v_i0, v_i1, v_i2, v_i3] = T.sigmoid(var_T_reshape_intermediate[v_i0, v_i1, v_i2, v_i3])
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(320), T.int64(128), T.int64(128)):
        with T.block("T_multiply"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], compute[v_ax0, v_ax1, v_ax2, v_ax3])
            T.writes(var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] * compute[v_ax0, v_ax1, v_ax2, v_ax3]

@T.prim_func
def fused_matmul10_add13_strided_slice7(lv95: T.Buffer((T.int64(2), T.int64(1280)), "float32"), lv96: T.Buffer((T.int64(1280), T.int64(640)), "float32"), unet_down_blocks_1_resnets_0_time_emb_proj_bias: T.Buffer((T.int64(640),), "float32"), var_T_strided_slice_with_axes_intermediate: T.Buffer((T.int64(2), T.int64(640)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_matmul_intermediate = T.alloc_buffer((T.int64(2), T.int64(640)))
    var_T_add_intermediate = T.alloc_buffer((T.int64(2), T.int64(640)))
    for i0, i1, k in T.grid(T.int64(2), T.int64(640), T.int64(1280)):
        with T.block("matmul"):
            v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])
            T.reads(lv95[v_i0, v_k], lv96[v_k, v_i1])
            T.writes(var_matmul_intermediate[v_i0, v_i1])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1] = T.float32(0)
            var_matmul_intermediate[v_i0, v_i1] = var_matmul_intermediate[v_i0, v_i1] + lv95[v_i0, v_k] * lv96[v_k, v_i1]
    for ax0, ax1 in T.grid(T.int64(2), T.int64(640)):
        with T.block("T_add"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(var_matmul_intermediate[v_ax0, v_ax1], unet_down_blocks_1_resnets_0_time_emb_proj_bias[v_ax1])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1])
            var_T_add_intermediate[v_ax0, v_ax1] = var_matmul_intermediate[v_ax0, v_ax1] + unet_down_blocks_1_resnets_0_time_emb_proj_bias[v_ax1]
    for ax0, ax1 in T.grid(T.int64(2), T.int64(640)):
        with T.block("T_strided_slice_with_axes"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(var_T_add_intermediate[v_ax0, v_ax1])
            T.writes(var_T_strided_slice_with_axes_intermediate[v_ax0, v_ax1])
            var_T_strided_slice_with_axes_intermediate[v_ax0, v_ax1] = var_T_add_intermediate[v_ax0, v_ax1]

@T.prim_func
def fused_matmul11_add16(lv114: T.Buffer((T.int64(2), T.int64(4096), T.int64(640)), "float32"), lv115: T.Buffer((T.int64(640), T.int64(640)), "float32"), unet_down_blocks_1_attentions_0_proj_in_bias: T.Buffer((T.int64(640),), "float32"), var_T_add_intermediate: T.Buffer((T.int64(2), T.int64(4096), T.int64(640)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_matmul_intermediate = T.alloc_buffer((T.int64(2), T.int64(4096), T.int64(640)))
    for i0, i1, i2, k in T.grid(T.int64(2), T.int64(4096), T.int64(640), T.int64(640)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv114[v_i0, v_i1, v_k], lv115[v_k, v_i2])
            T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
            var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[v_i0, v_i1, v_i2] + lv114[v_i0, v_i1, v_k] * lv115[v_k, v_i2]
    for ax0, ax1, ax2 in T.grid(T.int64(2), T.int64(4096), T.int64(640)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_matmul_intermediate[v_ax0, v_ax1, v_ax2], unet_down_blocks_1_attentions_0_proj_in_bias[v_ax2])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2] = var_matmul_intermediate[v_ax0, v_ax1, v_ax2] + unet_down_blocks_1_attentions_0_proj_in_bias[v_ax2]

@T.prim_func
def fused_matmul11_add16_divide3_add17(lv139: T.Buffer((T.int64(2), T.int64(4096), T.int64(640)), "float32"), lv140: T.Buffer((T.int64(640), T.int64(640)), "float32"), unet_down_blocks_1_attentions_0_transformer_blocks_0_attn1_to_out_0_bias: T.Buffer((T.int64(640),), "float32"), lv117: T.Buffer((T.int64(2), T.int64(4096), T.int64(640)), "float32"), var_T_add_intermediate: T.Buffer((T.int64(2), T.int64(4096), T.int64(640)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_matmul_intermediate = T.alloc_buffer((T.int64(2), T.int64(4096), T.int64(640)))
    var_T_add_intermediate_1 = T.alloc_buffer((T.int64(2), T.int64(4096), T.int64(640)))
    var_T_divide_intermediate = T.alloc_buffer((T.int64(2), T.int64(4096), T.int64(640)))
    for i0, i1, i2, k in T.grid(T.int64(2), T.int64(4096), T.int64(640), T.int64(640)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv139[v_i0, v_i1, v_k], lv140[v_k, v_i2])
            T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
            var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[v_i0, v_i1, v_i2] + lv139[v_i0, v_i1, v_k] * lv140[v_k, v_i2]
    for ax0, ax1, ax2 in T.grid(T.int64(2), T.int64(4096), T.int64(640)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_matmul_intermediate[v_ax0, v_ax1, v_ax2], unet_down_blocks_1_attentions_0_transformer_blocks_0_attn1_to_out_0_bias[v_ax2])
            T.writes(var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2])
            var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2] = var_matmul_intermediate[v_ax0, v_ax1, v_ax2] + unet_down_blocks_1_attentions_0_transformer_blocks_0_attn1_to_out_0_bias[v_ax2]
    for ax0, ax1, ax2 in T.grid(T.int64(2), T.int64(4096), T.int64(640)):
        with T.block("T_divide"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2])
            T.writes(var_T_divide_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_divide_intermediate[v_ax0, v_ax1, v_ax2] = var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2]
    for ax0, ax1, ax2 in T.grid(T.int64(2), T.int64(4096), T.int64(640)):
        with T.block("T_add_1"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_T_divide_intermediate[v_ax0, v_ax1, v_ax2], lv117[v_ax0, v_ax1, v_ax2])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2] = var_T_divide_intermediate[v_ax0, v_ax1, v_ax2] + lv117[v_ax0, v_ax1, v_ax2]

@T.prim_func
def fused_matmul12_multiply5(lv126: T.Buffer((T.int64(2), T.int64(10), T.int64(4096), T.int64(64)), "float32"), lv133: T.Buffer((T.int64(2), T.int64(10), T.int64(64), T.int64(4096)), "float32"), param_0: T.Buffer((), "float32"), var_T_multiply_intermediate: T.Buffer((T.int64(2), T.int64(10), T.int64(4096), T.int64(4096)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_matmul_intermediate = T.alloc_buffer((T.int64(2), T.int64(10), T.int64(4096), T.int64(4096)))
    for i0, i1, i2, i3, k in T.grid(T.int64(2), T.int64(10), T.int64(4096), T.int64(4096), T.int64(64)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_i3, v_k = T.axis.remap("SSSSR", [i0, i1, i2, i3, k])
            T.reads(lv126[v_i0, v_i1, v_i2, v_k], lv133[v_i0, v_i1, v_k, v_i3])
            T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2, v_i3])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1, v_i2, v_i3] = T.float32(0)
            var_matmul_intermediate[v_i0, v_i1, v_i2, v_i3] = var_matmul_intermediate[v_i0, v_i1, v_i2, v_i3] + lv126[v_i0, v_i1, v_i2, v_k] * lv133[v_i0, v_i1, v_k, v_i3]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(10), T.int64(4096), T.int64(4096)):
        with T.block("T_multiply"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_matmul_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], param_0[()])
            T.writes(var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_matmul_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] * param_0[()]

@T.prim_func
def fused_matmul15_multiply6(lv153: T.Buffer((T.int64(2), T.int64(10), T.int64(4096), T.int64(64)), "float32"), lv160: T.Buffer((T.int64(2), T.int64(10), T.int64(64), T.int64(77)), "float32"), param_0: T.Buffer((), "float32"), var_T_multiply_intermediate: T.Buffer((T.int64(2), T.int64(10), T.int64(4096), T.int64(77)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_matmul_intermediate = T.alloc_buffer((T.int64(2), T.int64(10), T.int64(4096), T.int64(77)))
    for i0, i1, i2, i3, k in T.grid(T.int64(2), T.int64(10), T.int64(4096), T.int64(77), T.int64(64)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_i3, v_k = T.axis.remap("SSSSR", [i0, i1, i2, i3, k])
            T.reads(lv153[v_i0, v_i1, v_i2, v_k], lv160[v_i0, v_i1, v_k, v_i3])
            T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2, v_i3])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1, v_i2, v_i3] = T.float32(0)
            var_matmul_intermediate[v_i0, v_i1, v_i2, v_i3] = var_matmul_intermediate[v_i0, v_i1, v_i2, v_i3] + lv153[v_i0, v_i1, v_i2, v_k] * lv160[v_i0, v_i1, v_k, v_i3]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(10), T.int64(4096), T.int64(77)):
        with T.block("T_multiply"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_matmul_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], param_0[()])
            T.writes(var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_matmul_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] * param_0[()]

@T.prim_func
def fused_matmul17_add18(lv172: T.Buffer((T.int64(2), T.int64(4096), T.int64(640)), "float32"), lv173: T.Buffer((T.int64(640), T.int64(5120)), "float32"), unet_down_blocks_1_attentions_0_transformer_blocks_0_ff_net_0_proj_bias: T.Buffer((T.int64(5120),), "float32"), var_T_add_intermediate: T.Buffer((T.int64(2), T.int64(4096), T.int64(5120)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_matmul_intermediate = T.alloc_buffer((T.int64(2), T.int64(4096), T.int64(5120)))
    for i0, i1, i2, k in T.grid(T.int64(2), T.int64(4096), T.int64(5120), T.int64(640)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv172[v_i0, v_i1, v_k], lv173[v_k, v_i2])
            T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
            var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[v_i0, v_i1, v_i2] + lv172[v_i0, v_i1, v_k] * lv173[v_k, v_i2]
    for ax0, ax1, ax2 in T.grid(T.int64(2), T.int64(4096), T.int64(5120)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_matmul_intermediate[v_ax0, v_ax1, v_ax2], unet_down_blocks_1_attentions_0_transformer_blocks_0_ff_net_0_proj_bias[v_ax2])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2] = var_matmul_intermediate[v_ax0, v_ax1, v_ax2] + unet_down_blocks_1_attentions_0_transformer_blocks_0_ff_net_0_proj_bias[v_ax2]

@T.prim_func
def fused_matmul18_add16_add17(lv180: T.Buffer((T.int64(2), T.int64(4096), T.int64(2560)), "float32"), lv181: T.Buffer((T.int64(2560), T.int64(640)), "float32"), unet_down_blocks_1_attentions_0_transformer_blocks_0_ff_net_2_bias: T.Buffer((T.int64(640),), "float32"), lv171: T.Buffer((T.int64(2), T.int64(4096), T.int64(640)), "float32"), var_T_add_intermediate: T.Buffer((T.int64(2), T.int64(4096), T.int64(640)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_matmul_intermediate = T.alloc_buffer((T.int64(2), T.int64(4096), T.int64(640)))
    var_T_add_intermediate_1 = T.alloc_buffer((T.int64(2), T.int64(4096), T.int64(640)))
    for i0, i1, i2, k in T.grid(T.int64(2), T.int64(4096), T.int64(640), T.int64(2560)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv180[v_i0, v_i1, v_k], lv181[v_k, v_i2])
            T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
            var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[v_i0, v_i1, v_i2] + lv180[v_i0, v_i1, v_k] * lv181[v_k, v_i2]
    for ax0, ax1, ax2 in T.grid(T.int64(2), T.int64(4096), T.int64(640)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_matmul_intermediate[v_ax0, v_ax1, v_ax2], unet_down_blocks_1_attentions_0_transformer_blocks_0_ff_net_2_bias[v_ax2])
            T.writes(var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2])
            var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2] = var_matmul_intermediate[v_ax0, v_ax1, v_ax2] + unet_down_blocks_1_attentions_0_transformer_blocks_0_ff_net_2_bias[v_ax2]
    for ax0, ax1, ax2 in T.grid(T.int64(2), T.int64(4096), T.int64(640)):
        with T.block("T_add_1"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2], lv171[v_ax0, v_ax1, v_ax2])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2] = var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2] + lv171[v_ax0, v_ax1, v_ax2]

@T.prim_func
def fused_matmul19_add23(lv450: T.Buffer((T.int64(2), T.int64(1024), T.int64(1280)), "float32"), lv451: T.Buffer((T.int64(1280), T.int64(1280)), "float32"), unet_down_blocks_2_attentions_0_proj_in_bias: T.Buffer((T.int64(1280),), "float32"), var_T_add_intermediate: T.Buffer((T.int64(2), T.int64(1024), T.int64(1280)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_matmul_intermediate = T.alloc_buffer((T.int64(2), T.int64(1024), T.int64(1280)))
    for i0, i1, i2, k in T.grid(T.int64(2), T.int64(1024), T.int64(1280), T.int64(1280)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv450[v_i0, v_i1, v_k], lv451[v_k, v_i2])
            T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
            var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[v_i0, v_i1, v_i2] + lv450[v_i0, v_i1, v_k] * lv451[v_k, v_i2]
    for ax0, ax1, ax2 in T.grid(T.int64(2), T.int64(1024), T.int64(1280)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_matmul_intermediate[v_ax0, v_ax1, v_ax2], unet_down_blocks_2_attentions_0_proj_in_bias[v_ax2])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2] = var_matmul_intermediate[v_ax0, v_ax1, v_ax2] + unet_down_blocks_2_attentions_0_proj_in_bias[v_ax2]

@T.prim_func
def fused_matmul19_add23_divide5_add24(lv475: T.Buffer((T.int64(2), T.int64(1024), T.int64(1280)), "float32"), lv476: T.Buffer((T.int64(1280), T.int64(1280)), "float32"), unet_down_blocks_2_attentions_0_transformer_blocks_0_attn1_to_out_0_bias: T.Buffer((T.int64(1280),), "float32"), lv453: T.Buffer((T.int64(2), T.int64(1024), T.int64(1280)), "float32"), var_T_add_intermediate: T.Buffer((T.int64(2), T.int64(1024), T.int64(1280)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_matmul_intermediate = T.alloc_buffer((T.int64(2), T.int64(1024), T.int64(1280)))
    var_T_add_intermediate_1 = T.alloc_buffer((T.int64(2), T.int64(1024), T.int64(1280)))
    var_T_divide_intermediate = T.alloc_buffer((T.int64(2), T.int64(1024), T.int64(1280)))
    for i0, i1, i2, k in T.grid(T.int64(2), T.int64(1024), T.int64(1280), T.int64(1280)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv475[v_i0, v_i1, v_k], lv476[v_k, v_i2])
            T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
            var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[v_i0, v_i1, v_i2] + lv475[v_i0, v_i1, v_k] * lv476[v_k, v_i2]
    for ax0, ax1, ax2 in T.grid(T.int64(2), T.int64(1024), T.int64(1280)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_matmul_intermediate[v_ax0, v_ax1, v_ax2], unet_down_blocks_2_attentions_0_transformer_blocks_0_attn1_to_out_0_bias[v_ax2])
            T.writes(var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2])
            var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2] = var_matmul_intermediate[v_ax0, v_ax1, v_ax2] + unet_down_blocks_2_attentions_0_transformer_blocks_0_attn1_to_out_0_bias[v_ax2]
    for ax0, ax1, ax2 in T.grid(T.int64(2), T.int64(1024), T.int64(1280)):
        with T.block("T_divide"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2])
            T.writes(var_T_divide_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_divide_intermediate[v_ax0, v_ax1, v_ax2] = var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2]
    for ax0, ax1, ax2 in T.grid(T.int64(2), T.int64(1024), T.int64(1280)):
        with T.block("T_add_1"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_T_divide_intermediate[v_ax0, v_ax1, v_ax2], lv453[v_ax0, v_ax1, v_ax2])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2] = var_T_divide_intermediate[v_ax0, v_ax1, v_ax2] + lv453[v_ax0, v_ax1, v_ax2]

@T.prim_func
def fused_matmul20_multiply8(lv462: T.Buffer((T.int64(2), T.int64(20), T.int64(1024), T.int64(64)), "float32"), lv469: T.Buffer((T.int64(2), T.int64(20), T.int64(64), T.int64(1024)), "float32"), param_0: T.Buffer((), "float32"), var_T_multiply_intermediate: T.Buffer((T.int64(2), T.int64(20), T.int64(1024), T.int64(1024)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_matmul_intermediate = T.alloc_buffer((T.int64(2), T.int64(20), T.int64(1024), T.int64(1024)))
    for i0, i1, i2, i3, k in T.grid(T.int64(2), T.int64(20), T.int64(1024), T.int64(1024), T.int64(64)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_i3, v_k = T.axis.remap("SSSSR", [i0, i1, i2, i3, k])
            T.reads(lv462[v_i0, v_i1, v_i2, v_k], lv469[v_i0, v_i1, v_k, v_i3])
            T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2, v_i3])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1, v_i2, v_i3] = T.float32(0)
            var_matmul_intermediate[v_i0, v_i1, v_i2, v_i3] = var_matmul_intermediate[v_i0, v_i1, v_i2, v_i3] + lv462[v_i0, v_i1, v_i2, v_k] * lv469[v_i0, v_i1, v_k, v_i3]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(20), T.int64(1024), T.int64(1024)):
        with T.block("T_multiply"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_matmul_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], param_0[()])
            T.writes(var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_matmul_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] * param_0[()]

@T.prim_func
def fused_matmul23_multiply9(lv489: T.Buffer((T.int64(2), T.int64(20), T.int64(1024), T.int64(64)), "float32"), lv496: T.Buffer((T.int64(2), T.int64(20), T.int64(64), T.int64(77)), "float32"), param_0: T.Buffer((), "float32"), var_T_multiply_intermediate: T.Buffer((T.int64(2), T.int64(20), T.int64(1024), T.int64(77)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_matmul_intermediate = T.alloc_buffer((T.int64(2), T.int64(20), T.int64(1024), T.int64(77)))
    for i0, i1, i2, i3, k in T.grid(T.int64(2), T.int64(20), T.int64(1024), T.int64(77), T.int64(64)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_i3, v_k = T.axis.remap("SSSSR", [i0, i1, i2, i3, k])
            T.reads(lv489[v_i0, v_i1, v_i2, v_k], lv496[v_i0, v_i1, v_k, v_i3])
            T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2, v_i3])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1, v_i2, v_i3] = T.float32(0)
            var_matmul_intermediate[v_i0, v_i1, v_i2, v_i3] = var_matmul_intermediate[v_i0, v_i1, v_i2, v_i3] + lv489[v_i0, v_i1, v_i2, v_k] * lv496[v_i0, v_i1, v_k, v_i3]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(20), T.int64(1024), T.int64(77)):
        with T.block("T_multiply"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_matmul_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], param_0[()])
            T.writes(var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_matmul_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] * param_0[()]

@T.prim_func
def fused_matmul25_add25(lv508: T.Buffer((T.int64(2), T.int64(1024), T.int64(1280)), "float32"), lv509: T.Buffer((T.int64(1280), T.int64(10240)), "float32"), unet_down_blocks_2_attentions_0_transformer_blocks_0_ff_net_0_proj_bias: T.Buffer((T.int64(10240),), "float32"), var_T_add_intermediate: T.Buffer((T.int64(2), T.int64(1024), T.int64(10240)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_matmul_intermediate = T.alloc_buffer((T.int64(2), T.int64(1024), T.int64(10240)))
    for i0, i1, i2, k in T.grid(T.int64(2), T.int64(1024), T.int64(10240), T.int64(1280)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv508[v_i0, v_i1, v_k], lv509[v_k, v_i2])
            T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
            var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[v_i0, v_i1, v_i2] + lv508[v_i0, v_i1, v_k] * lv509[v_k, v_i2]
    for ax0, ax1, ax2 in T.grid(T.int64(2), T.int64(1024), T.int64(10240)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_matmul_intermediate[v_ax0, v_ax1, v_ax2], unet_down_blocks_2_attentions_0_transformer_blocks_0_ff_net_0_proj_bias[v_ax2])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2] = var_matmul_intermediate[v_ax0, v_ax1, v_ax2] + unet_down_blocks_2_attentions_0_transformer_blocks_0_ff_net_0_proj_bias[v_ax2]

@T.prim_func
def fused_matmul26_add23_add24(lv516: T.Buffer((T.int64(2), T.int64(1024), T.int64(5120)), "float32"), lv517: T.Buffer((T.int64(5120), T.int64(1280)), "float32"), unet_down_blocks_2_attentions_0_transformer_blocks_0_ff_net_2_bias: T.Buffer((T.int64(1280),), "float32"), lv507: T.Buffer((T.int64(2), T.int64(1024), T.int64(1280)), "float32"), var_T_add_intermediate: T.Buffer((T.int64(2), T.int64(1024), T.int64(1280)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_matmul_intermediate = T.alloc_buffer((T.int64(2), T.int64(1024), T.int64(1280)))
    var_T_add_intermediate_1 = T.alloc_buffer((T.int64(2), T.int64(1024), T.int64(1280)))
    for i0, i1, i2, k in T.grid(T.int64(2), T.int64(1024), T.int64(1280), T.int64(5120)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv516[v_i0, v_i1, v_k], lv517[v_k, v_i2])
            T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
            var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[v_i0, v_i1, v_i2] + lv516[v_i0, v_i1, v_k] * lv517[v_k, v_i2]
    for ax0, ax1, ax2 in T.grid(T.int64(2), T.int64(1024), T.int64(1280)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_matmul_intermediate[v_ax0, v_ax1, v_ax2], unet_down_blocks_2_attentions_0_transformer_blocks_0_ff_net_2_bias[v_ax2])
            T.writes(var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2])
            var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2] = var_matmul_intermediate[v_ax0, v_ax1, v_ax2] + unet_down_blocks_2_attentions_0_transformer_blocks_0_ff_net_2_bias[v_ax2]
    for ax0, ax1, ax2 in T.grid(T.int64(2), T.int64(1024), T.int64(1280)):
        with T.block("T_add_1"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2], lv507[v_ax0, v_ax1, v_ax2])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2] = var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2] + lv507[v_ax0, v_ax1, v_ax2]

@T.prim_func
def fused_matmul27_add33(lv23: T.Buffer((T.int64(1), T.int64(16384), T.int64(512)), "float32"), lv24: T.Buffer((T.int64(512), T.int64(512)), "float32"), vae_decoder_mid_block_attentions_0_to_q_bias: T.Buffer((T.int64(512),), "float32"), var_T_add_intermediate: T.Buffer((T.int64(1), T.int64(16384), T.int64(512)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_matmul_intermediate = T.alloc_buffer((T.int64(1), T.int64(16384), T.int64(512)))
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(16384), T.int64(512), T.int64(512)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv23[v_i0, v_i1, v_k], lv24[v_k, v_i2])
            T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
            var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[v_i0, v_i1, v_i2] + lv23[v_i0, v_i1, v_k] * lv24[v_k, v_i2]
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(16384), T.int64(512)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_matmul_intermediate[v_ax0, v_ax1, v_ax2], vae_decoder_mid_block_attentions_0_to_q_bias[v_ax2])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2] = var_matmul_intermediate[v_ax0, v_ax1, v_ax2] + vae_decoder_mid_block_attentions_0_to_q_bias[v_ax2]

@T.prim_func
def fused_matmul28_multiply13(lv34: T.Buffer((T.int64(1), T.int64(1), T.int64(16384), T.int64(512)), "float32"), lv41: T.Buffer((T.int64(1), T.int64(1), T.int64(512), T.int64(16384)), "float32"), param_0: T.Buffer((), "float32"), var_T_multiply_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(16384), T.int64(16384)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_matmul_intermediate = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(16384), T.int64(16384)))
    for i0, i1, i2, i3, k in T.grid(T.int64(1), T.int64(1), T.int64(16384), T.int64(16384), T.int64(512)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_i3, v_k = T.axis.remap("SSSSR", [i0, i1, i2, i3, k])
            T.reads(lv34[v_i0, v_i1, v_i2, v_k], lv41[v_i0, v_i1, v_k, v_i3])
            T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2, v_i3])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1, v_i2, v_i3] = T.float32(0)
            var_matmul_intermediate[v_i0, v_i1, v_i2, v_i3] = var_matmul_intermediate[v_i0, v_i1, v_i2, v_i3] + lv34[v_i0, v_i1, v_i2, v_k] * lv41[v_i0, v_i1, v_k, v_i3]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(1), T.int64(16384), T.int64(16384)):
        with T.block("T_multiply"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_matmul_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], param_0[()])
            T.writes(var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_matmul_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] * param_0[()]

@T.prim_func
def fused_matmul30_add45(lv21: T.Buffer((T.int64(1), T.int64(77), T.int64(768)), "float32"), lv26: T.Buffer((T.int64(768), T.int64(768)), "float32"), self_clip_text_model_encoder_layers_0_self_attn_k_proj_bias: T.Buffer((T.int64(768),), "float32"), var_T_add_intermediate: T.Buffer((T.int64(1), T.int64(77), T.int64(768)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_matmul_intermediate = T.alloc_buffer((T.int64(1), T.int64(77), T.int64(768)))
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(77), T.int64(768), T.int64(768)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv21[v_i0, v_i1, v_k], lv26[v_k, v_i2])
            T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
            var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[v_i0, v_i1, v_i2] + lv21[v_i0, v_i1, v_k] * lv26[v_k, v_i2]
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(77), T.int64(768)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_matmul_intermediate[v_ax0, v_ax1, v_ax2], self_clip_text_model_encoder_layers_0_self_attn_k_proj_bias[v_ax2])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2] = var_matmul_intermediate[v_ax0, v_ax1, v_ax2] + self_clip_text_model_encoder_layers_0_self_attn_k_proj_bias[v_ax2]

@T.prim_func
def fused_matmul30_add45_add44(lv50: T.Buffer((T.int64(1), T.int64(77), T.int64(768)), "float32"), lv51: T.Buffer((T.int64(768), T.int64(768)), "float32"), self_clip_text_model_encoder_layers_0_self_attn_out_proj_bias: T.Buffer((T.int64(768),), "float32"), lv9: T.Buffer((T.int64(1), T.int64(77), T.int64(768)), "float32"), var_T_add_intermediate: T.Buffer((T.int64(1), T.int64(77), T.int64(768)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_matmul_intermediate = T.alloc_buffer((T.int64(1), T.int64(77), T.int64(768)))
    var_T_add_intermediate_1 = T.alloc_buffer((T.int64(1), T.int64(77), T.int64(768)))
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(77), T.int64(768), T.int64(768)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv50[v_i0, v_i1, v_k], lv51[v_k, v_i2])
            T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
            var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[v_i0, v_i1, v_i2] + lv50[v_i0, v_i1, v_k] * lv51[v_k, v_i2]
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(77), T.int64(768)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_matmul_intermediate[v_ax0, v_ax1, v_ax2], self_clip_text_model_encoder_layers_0_self_attn_out_proj_bias[v_ax2])
            T.writes(var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2])
            var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2] = var_matmul_intermediate[v_ax0, v_ax1, v_ax2] + self_clip_text_model_encoder_layers_0_self_attn_out_proj_bias[v_ax2]
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(77), T.int64(768)):
        with T.block("T_add_1"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(lv9[v_ax0, v_ax1, v_ax2], var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2] = lv9[v_ax0, v_ax1, v_ax2] + var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2]

@T.prim_func
def fused_matmul30_add45_multiply15(lv21: T.Buffer((T.int64(1), T.int64(77), T.int64(768)), "float32"), lv22: T.Buffer((T.int64(768), T.int64(768)), "float32"), self_clip_text_model_encoder_layers_0_self_attn_q_proj_bias: T.Buffer((T.int64(768),), "float32"), var_T_multiply_intermediate: T.Buffer((T.int64(1), T.int64(77), T.int64(768)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_matmul_intermediate = T.alloc_buffer((T.int64(1), T.int64(77), T.int64(768)))
    var_T_add_intermediate = T.alloc_buffer((T.int64(1), T.int64(77), T.int64(768)))
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(77), T.int64(768), T.int64(768)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv21[v_i0, v_i1, v_k], lv22[v_k, v_i2])
            T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
            var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[v_i0, v_i1, v_i2] + lv21[v_i0, v_i1, v_k] * lv22[v_k, v_i2]
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(77), T.int64(768)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_matmul_intermediate[v_ax0, v_ax1, v_ax2], self_clip_text_model_encoder_layers_0_self_attn_q_proj_bias[v_ax2])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2] = var_matmul_intermediate[v_ax0, v_ax1, v_ax2] + self_clip_text_model_encoder_layers_0_self_attn_q_proj_bias[v_ax2]
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(77), T.int64(768)):
        with T.block("T_multiply"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
            T.writes(var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2] = var_T_add_intermediate[v_ax0, v_ax1, v_ax2] * T.float32(0.125)

@T.prim_func
def fused_matmul33_add47_multiply16_tir_sigmoid_multiply17(lv55: T.Buffer((T.int64(1), T.int64(77), T.int64(768)), "float32"), lv56: T.Buffer((T.int64(768), T.int64(3072)), "float32"), self_clip_text_model_encoder_layers_0_mlp_fc1_bias: T.Buffer((T.int64(3072),), "float32"), var_T_multiply_intermediate: T.Buffer((T.int64(1), T.int64(77), T.int64(3072)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_matmul_intermediate = T.alloc_buffer((T.int64(1), T.int64(77), T.int64(3072)))
    var_T_add_intermediate = T.alloc_buffer((T.int64(1), T.int64(77), T.int64(3072)))
    var_T_multiply_intermediate_1 = T.alloc_buffer((T.int64(1), T.int64(77), T.int64(3072)))
    var_compute_intermediate = T.alloc_buffer((T.int64(1), T.int64(77), T.int64(3072)))
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(77), T.int64(3072), T.int64(768)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv55[v_i0, v_i1, v_k], lv56[v_k, v_i2])
            T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
            var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[v_i0, v_i1, v_i2] + lv55[v_i0, v_i1, v_k] * lv56[v_k, v_i2]
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(77), T.int64(3072)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_matmul_intermediate[v_ax0, v_ax1, v_ax2], self_clip_text_model_encoder_layers_0_mlp_fc1_bias[v_ax2])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2] = var_matmul_intermediate[v_ax0, v_ax1, v_ax2] + self_clip_text_model_encoder_layers_0_mlp_fc1_bias[v_ax2]
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(77), T.int64(3072)):
        with T.block("T_multiply"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
            T.writes(var_T_multiply_intermediate_1[v_ax0, v_ax1, v_ax2])
            var_T_multiply_intermediate_1[v_ax0, v_ax1, v_ax2] = T.float32(1.7020000219345093) * var_T_add_intermediate[v_ax0, v_ax1, v_ax2]
    for i0, i1, i2 in T.grid(T.int64(1), T.int64(77), T.int64(3072)):
        with T.block("compute"):
            v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
            T.reads(var_T_multiply_intermediate_1[v_i0, v_i1, v_i2])
            T.writes(var_compute_intermediate[v_i0, v_i1, v_i2])
            var_compute_intermediate[v_i0, v_i1, v_i2] = T.sigmoid(var_T_multiply_intermediate_1[v_i0, v_i1, v_i2])
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(77), T.int64(3072)):
        with T.block("T_multiply_1"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_T_add_intermediate[v_ax0, v_ax1, v_ax2], var_compute_intermediate[v_ax0, v_ax1, v_ax2])
            T.writes(var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2] = var_T_add_intermediate[v_ax0, v_ax1, v_ax2] * var_compute_intermediate[v_ax0, v_ax1, v_ax2]

@T.prim_func
def fused_matmul34_add45_add44(lv61: T.Buffer((T.int64(1), T.int64(77), T.int64(3072)), "float32"), lv62: T.Buffer((T.int64(3072), T.int64(768)), "float32"), self_clip_text_model_encoder_layers_0_mlp_fc2_bias: T.Buffer((T.int64(768),), "float32"), lv54: T.Buffer((T.int64(1), T.int64(77), T.int64(768)), "float32"), var_T_add_intermediate: T.Buffer((T.int64(1), T.int64(77), T.int64(768)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_matmul_intermediate = T.alloc_buffer((T.int64(1), T.int64(77), T.int64(768)))
    var_T_add_intermediate_1 = T.alloc_buffer((T.int64(1), T.int64(77), T.int64(768)))
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(77), T.int64(768), T.int64(3072)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv61[v_i0, v_i1, v_k], lv62[v_k, v_i2])
            T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
            var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[v_i0, v_i1, v_i2] + lv61[v_i0, v_i1, v_k] * lv62[v_k, v_i2]
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(77), T.int64(768)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_matmul_intermediate[v_ax0, v_ax1, v_ax2], self_clip_text_model_encoder_layers_0_mlp_fc2_bias[v_ax2])
            T.writes(var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2])
            var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2] = var_matmul_intermediate[v_ax0, v_ax1, v_ax2] + self_clip_text_model_encoder_layers_0_mlp_fc2_bias[v_ax2]
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(77), T.int64(768)):
        with T.block("T_add_1"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(lv54[v_ax0, v_ax1, v_ax2], var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2] = lv54[v_ax0, v_ax1, v_ax2] + var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2]

@T.prim_func
def fused_matmul3_add4_gelu(lv57: T.Buffer((T.int64(1), T.int64(77), T.int64(1280)), "float32"), lv58: T.Buffer((T.int64(1280), T.int64(5120)), "float32"), self_clip_text_model_encoder_layers_0_mlp_fc1_bias: T.Buffer((T.int64(5120),), "float32"), var_T_multiply_intermediate: T.Buffer((T.int64(1), T.int64(77), T.int64(5120)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_matmul_intermediate = T.alloc_buffer((T.int64(1), T.int64(77), T.int64(5120)))
    var_T_add_intermediate = T.alloc_buffer((T.int64(1), T.int64(77), T.int64(5120)))
    T_multiply = T.alloc_buffer((T.int64(1), T.int64(77), T.int64(5120)))
    compute = T.alloc_buffer((T.int64(1), T.int64(77), T.int64(5120)))
    T_multiply_1 = T.alloc_buffer((T.int64(1), T.int64(77), T.int64(5120)))
    T_add = T.alloc_buffer((T.int64(1), T.int64(77), T.int64(5120)))
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(77), T.int64(5120), T.int64(1280)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv57[v_i0, v_i1, v_k], lv58[v_k, v_i2])
            T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
            var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[v_i0, v_i1, v_i2] + lv57[v_i0, v_i1, v_k] * lv58[v_k, v_i2]
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(77), T.int64(5120)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_matmul_intermediate[v_ax0, v_ax1, v_ax2], self_clip_text_model_encoder_layers_0_mlp_fc1_bias[v_ax2])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2] = var_matmul_intermediate[v_ax0, v_ax1, v_ax2] + self_clip_text_model_encoder_layers_0_mlp_fc1_bias[v_ax2]
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(77), T.int64(5120)):
        with T.block("T_multiply"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
            T.writes(T_multiply[v_ax0, v_ax1, v_ax2])
            T_multiply[v_ax0, v_ax1, v_ax2] = var_T_add_intermediate[v_ax0, v_ax1, v_ax2] * T.float32(0.70710678118654757)
    for i0, i1, i2 in T.grid(T.int64(1), T.int64(77), T.int64(5120)):
        with T.block("compute"):
            v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
            T.reads(T_multiply[v_i0, v_i1, v_i2])
            T.writes(compute[v_i0, v_i1, v_i2])
            compute[v_i0, v_i1, v_i2] = T.erf(T_multiply[v_i0, v_i1, v_i2])
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(77), T.int64(5120)):
        with T.block("T_multiply_1"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(compute[v_ax0, v_ax1, v_ax2])
            T.writes(T_multiply_1[v_ax0, v_ax1, v_ax2])
            T_multiply_1[v_ax0, v_ax1, v_ax2] = compute[v_ax0, v_ax1, v_ax2] * T.float32(0.5)
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(77), T.int64(5120)):
        with T.block("T_add_1"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(T_multiply_1[v_ax0, v_ax1, v_ax2])
            T.writes(T_add[v_ax0, v_ax1, v_ax2])
            T_add[v_ax0, v_ax1, v_ax2] = T.float32(0.5) + T_multiply_1[v_ax0, v_ax1, v_ax2]
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(77), T.int64(5120)):
        with T.block("T_multiply_2"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_T_add_intermediate[v_ax0, v_ax1, v_ax2], T_add[v_ax0, v_ax1, v_ax2])
            T.writes(var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2] = var_T_add_intermediate[v_ax0, v_ax1, v_ax2] * T_add[v_ax0, v_ax1, v_ax2]

@T.prim_func
def fused_matmul4_add2_add(lv61: T.Buffer((T.int64(1), T.int64(77), T.int64(5120)), "float32"), lv62: T.Buffer((T.int64(5120), T.int64(1280)), "float32"), self_clip_text_model_encoder_layers_0_mlp_fc2_bias: T.Buffer((T.int64(1280),), "float32"), lv56: T.Buffer((T.int64(1), T.int64(77), T.int64(1280)), "float32"), var_T_add_intermediate: T.Buffer((T.int64(1), T.int64(77), T.int64(1280)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_matmul_intermediate = T.alloc_buffer((T.int64(1), T.int64(77), T.int64(1280)))
    var_T_add_intermediate_1 = T.alloc_buffer((T.int64(1), T.int64(77), T.int64(1280)))
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(77), T.int64(1280), T.int64(5120)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv61[v_i0, v_i1, v_k], lv62[v_k, v_i2])
            T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
            var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[v_i0, v_i1, v_i2] + lv61[v_i0, v_i1, v_k] * lv62[v_k, v_i2]
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(77), T.int64(1280)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_matmul_intermediate[v_ax0, v_ax1, v_ax2], self_clip_text_model_encoder_layers_0_mlp_fc2_bias[v_ax2])
            T.writes(var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2])
            var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2] = var_matmul_intermediate[v_ax0, v_ax1, v_ax2] + self_clip_text_model_encoder_layers_0_mlp_fc2_bias[v_ax2]
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(77), T.int64(1280)):
        with T.block("T_add_1"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(lv56[v_ax0, v_ax1, v_ax2], var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2] = lv56[v_ax0, v_ax1, v_ax2] + var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2]

@T.prim_func
def fused_matmul6_add5_silu(lv14: T.Buffer((T.int64(2), T.int64(320)), "float32"), lv15: T.Buffer((T.int64(320), T.int64(1280)), "float32"), unet_time_embedding_linear_1_bias: T.Buffer((T.int64(1280),), "float32"), var_T_multiply_intermediate: T.Buffer((T.int64(2), T.int64(1280)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_matmul_intermediate = T.alloc_buffer((T.int64(2), T.int64(1280)))
    var_T_add_intermediate = T.alloc_buffer((T.int64(2), T.int64(1280)))
    compute = T.alloc_buffer((T.int64(2), T.int64(1280)))
    for i0, i1, k in T.grid(T.int64(2), T.int64(1280), T.int64(320)):
        with T.block("matmul"):
            v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])
            T.reads(lv14[v_i0, v_k], lv15[v_k, v_i1])
            T.writes(var_matmul_intermediate[v_i0, v_i1])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1] = T.float32(0)
            var_matmul_intermediate[v_i0, v_i1] = var_matmul_intermediate[v_i0, v_i1] + lv14[v_i0, v_k] * lv15[v_k, v_i1]
    for ax0, ax1 in T.grid(T.int64(2), T.int64(1280)):
        with T.block("T_add"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(var_matmul_intermediate[v_ax0, v_ax1], unet_time_embedding_linear_1_bias[v_ax1])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1])
            var_T_add_intermediate[v_ax0, v_ax1] = var_matmul_intermediate[v_ax0, v_ax1] + unet_time_embedding_linear_1_bias[v_ax1]
    for i0, i1 in T.grid(T.int64(2), T.int64(1280)):
        with T.block("compute"):
            v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
            T.reads(var_T_add_intermediate[v_i0, v_i1])
            T.writes(compute[v_i0, v_i1])
            compute[v_i0, v_i1] = T.sigmoid(var_T_add_intermediate[v_i0, v_i1])
    for ax0, ax1 in T.grid(T.int64(2), T.int64(1280)):
        with T.block("T_multiply"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(var_T_add_intermediate[v_ax0, v_ax1], compute[v_ax0, v_ax1])
            T.writes(var_T_multiply_intermediate[v_ax0, v_ax1])
            var_T_multiply_intermediate[v_ax0, v_ax1] = var_T_add_intermediate[v_ax0, v_ax1] * compute[v_ax0, v_ax1]

@T.prim_func
def fused_matmul7_add5(lv41: T.Buffer((T.int64(2), T.int64(1280)), "float32"), lv42: T.Buffer((T.int64(1280), T.int64(1280)), "float32"), unet_add_embedding_linear_2_bias: T.Buffer((T.int64(1280),), "float32"), var_T_add_intermediate: T.Buffer((T.int64(2), T.int64(1280)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_matmul_intermediate = T.alloc_buffer((T.int64(2), T.int64(1280)))
    for i0, i1, k in T.grid(T.int64(2), T.int64(1280), T.int64(1280)):
        with T.block("matmul"):
            v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])
            T.reads(lv41[v_i0, v_k], lv42[v_k, v_i1])
            T.writes(var_matmul_intermediate[v_i0, v_i1])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1] = T.float32(0)
            var_matmul_intermediate[v_i0, v_i1] = var_matmul_intermediate[v_i0, v_i1] + lv41[v_i0, v_k] * lv42[v_k, v_i1]
    for ax0, ax1 in T.grid(T.int64(2), T.int64(1280)):
        with T.block("T_add"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(var_matmul_intermediate[v_ax0, v_ax1], unet_add_embedding_linear_2_bias[v_ax1])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1])
            var_T_add_intermediate[v_ax0, v_ax1] = var_matmul_intermediate[v_ax0, v_ax1] + unet_add_embedding_linear_2_bias[v_ax1]

@T.prim_func
def fused_matmul7_add5_add6(lv18: T.Buffer((T.int64(2), T.int64(1280)), "float32"), lv19: T.Buffer((T.int64(1280), T.int64(1280)), "float32"), unet_time_embedding_linear_2_bias: T.Buffer((T.int64(1280),), "float32"), lv44: T.Buffer((T.int64(2), T.int64(1280)), "float32"), var_T_add_intermediate: T.Buffer((T.int64(2), T.int64(1280)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_matmul_intermediate = T.alloc_buffer((T.int64(2), T.int64(1280)))
    var_T_add_intermediate_1 = T.alloc_buffer((T.int64(2), T.int64(1280)))
    for i0, i1, k in T.grid(T.int64(2), T.int64(1280), T.int64(1280)):
        with T.block("matmul"):
            v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])
            T.reads(lv18[v_i0, v_k], lv19[v_k, v_i1])
            T.writes(var_matmul_intermediate[v_i0, v_i1])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1] = T.float32(0)
            var_matmul_intermediate[v_i0, v_i1] = var_matmul_intermediate[v_i0, v_i1] + lv18[v_i0, v_k] * lv19[v_k, v_i1]
    for ax0, ax1 in T.grid(T.int64(2), T.int64(1280)):
        with T.block("T_add"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(var_matmul_intermediate[v_ax0, v_ax1], unet_time_embedding_linear_2_bias[v_ax1])
            T.writes(var_T_add_intermediate_1[v_ax0, v_ax1])
            var_T_add_intermediate_1[v_ax0, v_ax1] = var_matmul_intermediate[v_ax0, v_ax1] + unet_time_embedding_linear_2_bias[v_ax1]
    for ax0, ax1 in T.grid(T.int64(2), T.int64(1280)):
        with T.block("T_add_1"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(var_T_add_intermediate_1[v_ax0, v_ax1], lv44[v_ax0, v_ax1])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1])
            var_T_add_intermediate[v_ax0, v_ax1] = var_T_add_intermediate_1[v_ax0, v_ax1] + lv44[v_ax0, v_ax1]

@T.prim_func
def fused_matmul7_add5_strided_slice8(lv431: T.Buffer((T.int64(2), T.int64(1280)), "float32"), lv432: T.Buffer((T.int64(1280), T.int64(1280)), "float32"), unet_down_blocks_2_resnets_0_time_emb_proj_bias: T.Buffer((T.int64(1280),), "float32"), var_T_strided_slice_with_axes_intermediate: T.Buffer((T.int64(2), T.int64(1280)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_matmul_intermediate = T.alloc_buffer((T.int64(2), T.int64(1280)))
    var_T_add_intermediate = T.alloc_buffer((T.int64(2), T.int64(1280)))
    for i0, i1, k in T.grid(T.int64(2), T.int64(1280), T.int64(1280)):
        with T.block("matmul"):
            v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])
            T.reads(lv431[v_i0, v_k], lv432[v_k, v_i1])
            T.writes(var_matmul_intermediate[v_i0, v_i1])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1] = T.float32(0)
            var_matmul_intermediate[v_i0, v_i1] = var_matmul_intermediate[v_i0, v_i1] + lv431[v_i0, v_k] * lv432[v_k, v_i1]
    for ax0, ax1 in T.grid(T.int64(2), T.int64(1280)):
        with T.block("T_add"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(var_matmul_intermediate[v_ax0, v_ax1], unet_down_blocks_2_resnets_0_time_emb_proj_bias[v_ax1])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1])
            var_T_add_intermediate[v_ax0, v_ax1] = var_matmul_intermediate[v_ax0, v_ax1] + unet_down_blocks_2_resnets_0_time_emb_proj_bias[v_ax1]
    for ax0, ax1 in T.grid(T.int64(2), T.int64(1280)):
        with T.block("T_strided_slice_with_axes"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(var_T_add_intermediate[v_ax0, v_ax1])
            T.writes(var_T_strided_slice_with_axes_intermediate[v_ax0, v_ax1])
            var_T_strided_slice_with_axes_intermediate[v_ax0, v_ax1] = var_T_add_intermediate[v_ax0, v_ax1]

@T.prim_func
def fused_matmul8_add5_silu(lv37: T.Buffer((T.int64(2), T.int64(2816)), "float32"), lv38: T.Buffer((T.int64(2816), T.int64(1280)), "float32"), unet_add_embedding_linear_1_bias: T.Buffer((T.int64(1280),), "float32"), var_T_multiply_intermediate: T.Buffer((T.int64(2), T.int64(1280)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_matmul_intermediate = T.alloc_buffer((T.int64(2), T.int64(1280)))
    var_T_add_intermediate = T.alloc_buffer((T.int64(2), T.int64(1280)))
    compute = T.alloc_buffer((T.int64(2), T.int64(1280)))
    for i0, i1, k in T.grid(T.int64(2), T.int64(1280), T.int64(2816)):
        with T.block("matmul"):
            v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])
            T.reads(lv37[v_i0, v_k], lv38[v_k, v_i1])
            T.writes(var_matmul_intermediate[v_i0, v_i1])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1] = T.float32(0)
            var_matmul_intermediate[v_i0, v_i1] = var_matmul_intermediate[v_i0, v_i1] + lv37[v_i0, v_k] * lv38[v_k, v_i1]
    for ax0, ax1 in T.grid(T.int64(2), T.int64(1280)):
        with T.block("T_add"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(var_matmul_intermediate[v_ax0, v_ax1], unet_add_embedding_linear_1_bias[v_ax1])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1])
            var_T_add_intermediate[v_ax0, v_ax1] = var_matmul_intermediate[v_ax0, v_ax1] + unet_add_embedding_linear_1_bias[v_ax1]
    for i0, i1 in T.grid(T.int64(2), T.int64(1280)):
        with T.block("compute"):
            v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
            T.reads(var_T_add_intermediate[v_i0, v_i1])
            T.writes(compute[v_i0, v_i1])
            compute[v_i0, v_i1] = T.sigmoid(var_T_add_intermediate[v_i0, v_i1])
    for ax0, ax1 in T.grid(T.int64(2), T.int64(1280)):
        with T.block("T_multiply"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(var_T_add_intermediate[v_ax0, v_ax1], compute[v_ax0, v_ax1])
            T.writes(var_T_multiply_intermediate[v_ax0, v_ax1])
            var_T_multiply_intermediate[v_ax0, v_ax1] = var_T_add_intermediate[v_ax0, v_ax1] * compute[v_ax0, v_ax1]

@T.prim_func
def fused_matmul9_add8_cast4(lv54: T.Buffer((T.int64(2), T.int64(1280)), "float32"), lv55: T.Buffer((T.int64(1280), T.int64(320)), "float32"), unet_down_blocks_0_resnets_0_time_emb_proj_bias: T.Buffer((T.int64(320),), "float32"), var_compute_intermediate: T.Buffer((T.int64(2), T.int64(320)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_matmul_intermediate = T.alloc_buffer((T.int64(2), T.int64(320)))
    var_T_add_intermediate = T.alloc_buffer((T.int64(2), T.int64(320)))
    for i0, i1, k in T.grid(T.int64(2), T.int64(320), T.int64(1280)):
        with T.block("matmul"):
            v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])
            T.reads(lv54[v_i0, v_k], lv55[v_k, v_i1])
            T.writes(var_matmul_intermediate[v_i0, v_i1])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1] = T.float32(0)
            var_matmul_intermediate[v_i0, v_i1] = var_matmul_intermediate[v_i0, v_i1] + lv54[v_i0, v_k] * lv55[v_k, v_i1]
    for ax0, ax1 in T.grid(T.int64(2), T.int64(320)):
        with T.block("T_add"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(var_matmul_intermediate[v_ax0, v_ax1], unet_down_blocks_0_resnets_0_time_emb_proj_bias[v_ax1])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1])
            var_T_add_intermediate[v_ax0, v_ax1] = var_matmul_intermediate[v_ax0, v_ax1] + unet_down_blocks_0_resnets_0_time_emb_proj_bias[v_ax1]
    for i0, i1 in T.grid(T.int64(2), T.int64(320)):
        with T.block("compute"):
            v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
            T.reads(var_T_add_intermediate[v_i0, v_i1])
            T.writes(var_compute_intermediate[v_i0, v_i1])
            var_compute_intermediate[v_i0, v_i1] = var_T_add_intermediate[v_i0, v_i1]

@T.prim_func
def fused_matmul_add2(lv21: T.Buffer((T.int64(1), T.int64(77), T.int64(1280)), "float32"), lv26: T.Buffer((T.int64(1280), T.int64(1280)), "float32"), self_clip_text_model_encoder_layers_0_self_attn_k_proj_bias: T.Buffer((T.int64(1280),), "float32"), var_T_add_intermediate: T.Buffer((T.int64(1), T.int64(77), T.int64(1280)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_matmul_intermediate = T.alloc_buffer((T.int64(1), T.int64(77), T.int64(1280)))
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(77), T.int64(1280), T.int64(1280)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv21[v_i0, v_i1, v_k], lv26[v_k, v_i2])
            T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
            var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[v_i0, v_i1, v_i2] + lv21[v_i0, v_i1, v_k] * lv26[v_k, v_i2]
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(77), T.int64(1280)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_matmul_intermediate[v_ax0, v_ax1, v_ax2], self_clip_text_model_encoder_layers_0_self_attn_k_proj_bias[v_ax2])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2] = var_matmul_intermediate[v_ax0, v_ax1, v_ax2] + self_clip_text_model_encoder_layers_0_self_attn_k_proj_bias[v_ax2]

@T.prim_func
def fused_matmul_add2_add(lv52: T.Buffer((T.int64(1), T.int64(77), T.int64(1280)), "float32"), lv53: T.Buffer((T.int64(1280), T.int64(1280)), "float32"), self_clip_text_model_encoder_layers_0_self_attn_out_proj_bias: T.Buffer((T.int64(1280),), "float32"), lv9: T.Buffer((T.int64(1), T.int64(77), T.int64(1280)), "float32"), var_T_add_intermediate: T.Buffer((T.int64(1), T.int64(77), T.int64(1280)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_matmul_intermediate = T.alloc_buffer((T.int64(1), T.int64(77), T.int64(1280)))
    var_T_add_intermediate_1 = T.alloc_buffer((T.int64(1), T.int64(77), T.int64(1280)))
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(77), T.int64(1280), T.int64(1280)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv52[v_i0, v_i1, v_k], lv53[v_k, v_i2])
            T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
            var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[v_i0, v_i1, v_i2] + lv52[v_i0, v_i1, v_k] * lv53[v_k, v_i2]
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(77), T.int64(1280)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_matmul_intermediate[v_ax0, v_ax1, v_ax2], self_clip_text_model_encoder_layers_0_self_attn_out_proj_bias[v_ax2])
            T.writes(var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2])
            var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2] = var_matmul_intermediate[v_ax0, v_ax1, v_ax2] + self_clip_text_model_encoder_layers_0_self_attn_out_proj_bias[v_ax2]
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(77), T.int64(1280)):
        with T.block("T_add_1"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(lv9[v_ax0, v_ax1, v_ax2], var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2] = lv9[v_ax0, v_ax1, v_ax2] + var_T_add_intermediate_1[v_ax0, v_ax1, v_ax2]

@T.prim_func
def fused_matmul_add2_multiply(lv21: T.Buffer((T.int64(1), T.int64(77), T.int64(1280)), "float32"), lv22: T.Buffer((T.int64(1280), T.int64(1280)), "float32"), self_clip_text_model_encoder_layers_0_self_attn_q_proj_bias: T.Buffer((T.int64(1280),), "float32"), var_T_multiply_intermediate: T.Buffer((T.int64(1), T.int64(77), T.int64(1280)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_matmul_intermediate = T.alloc_buffer((T.int64(1), T.int64(77), T.int64(1280)))
    var_T_add_intermediate = T.alloc_buffer((T.int64(1), T.int64(77), T.int64(1280)))
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(77), T.int64(1280), T.int64(1280)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv21[v_i0, v_i1, v_k], lv22[v_k, v_i2])
            T.writes(var_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_matmul_intermediate[v_i0, v_i1, v_i2] = T.float32(0)
            var_matmul_intermediate[v_i0, v_i1, v_i2] = var_matmul_intermediate[v_i0, v_i1, v_i2] + lv21[v_i0, v_i1, v_k] * lv22[v_k, v_i2]
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(77), T.int64(1280)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_matmul_intermediate[v_ax0, v_ax1, v_ax2], self_clip_text_model_encoder_layers_0_self_attn_q_proj_bias[v_ax2])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2] = var_matmul_intermediate[v_ax0, v_ax1, v_ax2] + self_clip_text_model_encoder_layers_0_self_attn_q_proj_bias[v_ax2]
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(77), T.int64(1280)):
        with T.block("T_multiply"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
            T.writes(var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2] = var_T_add_intermediate[v_ax0, v_ax1, v_ax2] * T.float32(0.125)

@T.prim_func
def fused_reshape14_strided_slice4_reshape15_cast5_multiply3_multiply4_tir_sin1_tir_cos1_concatenate2_strided_slice5_reshape16_strided_slice6_reshape16_concatenate2_reshape17_concatenate3(inp_4: T.Buffer((T.int64(2), T.int64(6)), "float32"), param_0: T.Buffer((T.int64(1), T.int64(128)), "float32"), inp_3: T.Buffer((T.int64(2), T.int64(1280)), "float32"), var_T_concat_intermediate: T.Buffer((T.int64(2), T.int64(2816)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_T_reshape_intermediate = T.alloc_buffer((T.int64(12),))
    var_T_strided_slice_with_axes_intermediate = T.alloc_buffer((T.int64(12),))
    var_T_reshape_intermediate_1 = T.alloc_buffer((T.int64(12), T.int64(1)))
    var_compute_intermediate = T.alloc_buffer((T.int64(12), T.int64(1)))
    var_T_multiply_intermediate = T.alloc_buffer((T.int64(12), T.int64(128)))
    var_T_multiply_intermediate_1 = T.alloc_buffer((T.int64(12), T.int64(128)))
    var_compute_intermediate_1 = T.alloc_buffer((T.int64(12), T.int64(128)))
    var_compute_intermediate_2 = T.alloc_buffer((T.int64(12), T.int64(128)))
    var_T_concat_intermediate_1 = T.alloc_buffer((T.int64(12), T.int64(256)))
    var_T_strided_slice_with_axes_intermediate_1 = T.alloc_buffer((T.int64(12), T.int64(128)))
    var_T_reshape_intermediate_2 = T.alloc_buffer((T.int64(12), T.int64(128)))
    var_T_strided_slice_with_axes_intermediate_2 = T.alloc_buffer((T.int64(12), T.int64(128)))
    var_T_reshape_intermediate_3 = T.alloc_buffer((T.int64(12), T.int64(128)))
    var_T_concat_intermediate_2 = T.alloc_buffer((T.int64(12), T.int64(256)))
    var_T_reshape_intermediate_4 = T.alloc_buffer((T.int64(2), T.int64(1536)))
    for ax0 in range(T.int64(12)):
        with T.block("T_reshape"):
            v_ax0 = T.axis.spatial(T.int64(12), ax0)
            T.reads(inp_4[v_ax0 % T.int64(12) // T.int64(6), v_ax0 % T.int64(6)])
            T.writes(var_T_reshape_intermediate[v_ax0])
            var_T_reshape_intermediate[v_ax0] = inp_4[v_ax0 % T.int64(12) // T.int64(6), v_ax0 % T.int64(6)]
    for ax0 in range(T.int64(12)):
        with T.block("T_strided_slice_with_axes"):
            v_ax0 = T.axis.spatial(T.int64(12), ax0)
            T.reads(var_T_reshape_intermediate[v_ax0])
            T.writes(var_T_strided_slice_with_axes_intermediate[v_ax0])
            var_T_strided_slice_with_axes_intermediate[v_ax0] = var_T_reshape_intermediate[v_ax0]
    for ax0, ax1 in T.grid(T.int64(12), T.int64(1)):
        with T.block("T_reshape_1"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(var_T_strided_slice_with_axes_intermediate[(v_ax0 + v_ax1) % T.int64(12)])
            T.writes(var_T_reshape_intermediate_1[v_ax0, v_ax1])
            var_T_reshape_intermediate_1[v_ax0, v_ax1] = var_T_strided_slice_with_axes_intermediate[(v_ax0 + v_ax1) % T.int64(12)]
    for i0, i1 in T.grid(T.int64(12), T.int64(1)):
        with T.block("compute"):
            v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
            T.reads(var_T_reshape_intermediate_1[v_i0, v_i1])
            T.writes(var_compute_intermediate[v_i0, v_i1])
            var_compute_intermediate[v_i0, v_i1] = var_T_reshape_intermediate_1[v_i0, v_i1]
    for ax0, ax1 in T.grid(T.int64(12), T.int64(128)):
        with T.block("T_multiply"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(var_compute_intermediate[v_ax0, T.int64(0)], param_0[T.int64(0), v_ax1])
            T.writes(var_T_multiply_intermediate[v_ax0, v_ax1])
            var_T_multiply_intermediate[v_ax0, v_ax1] = var_compute_intermediate[v_ax0, T.int64(0)] * param_0[T.int64(0), v_ax1]
    for ax0, ax1 in T.grid(T.int64(12), T.int64(128)):
        with T.block("T_multiply_1"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(var_T_multiply_intermediate[v_ax0, v_ax1])
            T.writes(var_T_multiply_intermediate_1[v_ax0, v_ax1])
            var_T_multiply_intermediate_1[v_ax0, v_ax1] = var_T_multiply_intermediate[v_ax0, v_ax1]
    for i0, i1 in T.grid(T.int64(12), T.int64(128)):
        with T.block("compute_1"):
            v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
            T.reads(var_T_multiply_intermediate_1[v_i0, v_i1])
            T.writes(var_compute_intermediate_1[v_i0, v_i1])
            var_compute_intermediate_1[v_i0, v_i1] = T.sin(var_T_multiply_intermediate_1[v_i0, v_i1])
    for i0, i1 in T.grid(T.int64(12), T.int64(128)):
        with T.block("compute_2"):
            v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
            T.reads(var_T_multiply_intermediate_1[v_i0, v_i1])
            T.writes(var_compute_intermediate_2[v_i0, v_i1])
            var_compute_intermediate_2[v_i0, v_i1] = T.cos(var_T_multiply_intermediate_1[v_i0, v_i1])
    for ax0, ax1 in T.grid(T.int64(12), T.int64(256)):
        with T.block("T_concat"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(var_compute_intermediate_2[v_ax0, v_ax1 - T.int64(128)], var_compute_intermediate_1[v_ax0, v_ax1])
            T.writes(var_T_concat_intermediate_1[v_ax0, v_ax1])
            var_T_concat_intermediate_1[v_ax0, v_ax1] = T.if_then_else(T.int64(128) <= v_ax1, var_compute_intermediate_2[v_ax0, v_ax1 - T.int64(128)], var_compute_intermediate_1[v_ax0, v_ax1])
    for ax0, ax1 in T.grid(T.int64(12), T.int64(128)):
        with T.block("T_strided_slice_with_axes_1"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(var_T_concat_intermediate_1[v_ax0, v_ax1 + T.int64(128)])
            T.writes(var_T_strided_slice_with_axes_intermediate_1[v_ax0, v_ax1])
            var_T_strided_slice_with_axes_intermediate_1[v_ax0, v_ax1] = var_T_concat_intermediate_1[v_ax0, v_ax1 + T.int64(128)]
    for ax0, ax1 in T.grid(T.int64(12), T.int64(128)):
        with T.block("T_reshape_2"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(var_T_strided_slice_with_axes_intermediate_1[(v_ax1 // T.int64(128) + v_ax0) % T.int64(12), v_ax1 % T.int64(128)])
            T.writes(var_T_reshape_intermediate_2[v_ax0, v_ax1])
            var_T_reshape_intermediate_2[v_ax0, v_ax1] = var_T_strided_slice_with_axes_intermediate_1[(v_ax1 // T.int64(128) + v_ax0) % T.int64(12), v_ax1 % T.int64(128)]
    for ax0, ax1 in T.grid(T.int64(12), T.int64(128)):
        with T.block("T_strided_slice_with_axes_2"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(var_T_concat_intermediate_1[v_ax0, v_ax1])
            T.writes(var_T_strided_slice_with_axes_intermediate_2[v_ax0, v_ax1])
            var_T_strided_slice_with_axes_intermediate_2[v_ax0, v_ax1] = var_T_concat_intermediate_1[v_ax0, v_ax1]
    for ax0, ax1 in T.grid(T.int64(12), T.int64(128)):
        with T.block("T_reshape_3"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(var_T_strided_slice_with_axes_intermediate_2[(v_ax1 // T.int64(128) + v_ax0) % T.int64(12), v_ax1 % T.int64(128)])
            T.writes(var_T_reshape_intermediate_3[v_ax0, v_ax1])
            var_T_reshape_intermediate_3[v_ax0, v_ax1] = var_T_strided_slice_with_axes_intermediate_2[(v_ax1 // T.int64(128) + v_ax0) % T.int64(12), v_ax1 % T.int64(128)]
    for ax0, ax1 in T.grid(T.int64(12), T.int64(256)):
        with T.block("T_concat_1"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(var_T_reshape_intermediate_3[v_ax0, v_ax1 - T.int64(128)], var_T_reshape_intermediate_2[v_ax0, v_ax1])
            T.writes(var_T_concat_intermediate_2[v_ax0, v_ax1])
            var_T_concat_intermediate_2[v_ax0, v_ax1] = T.if_then_else(T.int64(128) <= v_ax1, var_T_reshape_intermediate_3[v_ax0, v_ax1 - T.int64(128)], var_T_reshape_intermediate_2[v_ax0, v_ax1])
    for ax0, ax1 in T.grid(T.int64(2), T.int64(1536)):
        with T.block("T_reshape_4"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(var_T_concat_intermediate_2[(v_ax0 * T.int64(6) + v_ax1 // T.int64(256)) % T.int64(12), v_ax1 % T.int64(256)])
            T.writes(var_T_reshape_intermediate_4[v_ax0, v_ax1])
            var_T_reshape_intermediate_4[v_ax0, v_ax1] = var_T_concat_intermediate_2[(v_ax0 * T.int64(6) + v_ax1 // T.int64(256)) % T.int64(12), v_ax1 % T.int64(256)]
    for ax0, ax1 in T.grid(T.int64(2), T.int64(2816)):
        with T.block("T_concat_2"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(var_T_reshape_intermediate_4[v_ax0, v_ax1 - T.int64(1280)], inp_3[v_ax0, v_ax1])
            T.writes(var_T_concat_intermediate[v_ax0, v_ax1])
            var_T_concat_intermediate[v_ax0, v_ax1] = T.if_then_else(T.int64(1280) <= v_ax1, var_T_reshape_intermediate_4[v_ax0, v_ax1 - T.int64(1280)], inp_3[v_ax0, v_ax1])

@T.prim_func
def fused_reshape23_transpose12(lv120: T.Buffer((T.int64(2), T.int64(4096), T.int64(640)), "float32"), var_T_transpose_intermediate: T.Buffer((T.int64(2), T.int64(10), T.int64(4096), T.int64(64)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_T_reshape_intermediate = T.alloc_buffer((T.int64(2), T.int64(4096), T.int64(10), T.int64(64)))
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(4096), T.int64(10), T.int64(64)):
        with T.block("T_reshape"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(lv120[(((v_ax2 * T.int64(64) + v_ax3) // T.int64(640) + v_ax1) // T.int64(4096) + v_ax0) % T.int64(2), ((v_ax2 * T.int64(64) + v_ax3) // T.int64(640) + v_ax1) % T.int64(4096), (v_ax2 * T.int64(64) + v_ax3) % T.int64(640)])
            T.writes(var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = lv120[(((v_ax2 * T.int64(64) + v_ax3) // T.int64(640) + v_ax1) // T.int64(4096) + v_ax0) % T.int64(2), ((v_ax2 * T.int64(64) + v_ax3) // T.int64(640) + v_ax1) % T.int64(4096), (v_ax2 * T.int64(64) + v_ax3) % T.int64(640)]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(10), T.int64(4096), T.int64(64)):
        with T.block("T_transpose"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_reshape_intermediate[v_ax0, v_ax2, v_ax1, v_ax3])
            T.writes(var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_T_reshape_intermediate[v_ax0, v_ax2, v_ax1, v_ax3]

@T.prim_func
def fused_reshape23_transpose12_transpose13(lv122: T.Buffer((T.int64(2), T.int64(4096), T.int64(640)), "float32"), var_T_transpose_intermediate: T.Buffer((T.int64(2), T.int64(10), T.int64(64), T.int64(4096)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_T_reshape_intermediate = T.alloc_buffer((T.int64(2), T.int64(4096), T.int64(10), T.int64(64)))
    var_T_transpose_intermediate_1 = T.alloc_buffer((T.int64(2), T.int64(10), T.int64(4096), T.int64(64)))
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(4096), T.int64(10), T.int64(64)):
        with T.block("T_reshape"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(lv122[(((v_ax2 * T.int64(64) + v_ax3) // T.int64(640) + v_ax1) // T.int64(4096) + v_ax0) % T.int64(2), ((v_ax2 * T.int64(64) + v_ax3) // T.int64(640) + v_ax1) % T.int64(4096), (v_ax2 * T.int64(64) + v_ax3) % T.int64(640)])
            T.writes(var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = lv122[(((v_ax2 * T.int64(64) + v_ax3) // T.int64(640) + v_ax1) // T.int64(4096) + v_ax0) % T.int64(2), ((v_ax2 * T.int64(64) + v_ax3) // T.int64(640) + v_ax1) % T.int64(4096), (v_ax2 * T.int64(64) + v_ax3) % T.int64(640)]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(10), T.int64(4096), T.int64(64)):
        with T.block("T_transpose"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_reshape_intermediate[v_ax0, v_ax2, v_ax1, v_ax3])
            T.writes(var_T_transpose_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_transpose_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3] = var_T_reshape_intermediate[v_ax0, v_ax2, v_ax1, v_ax3]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(10), T.int64(64), T.int64(4096)):
        with T.block("T_transpose_1"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_transpose_intermediate_1[v_ax0, v_ax1, v_ax3, v_ax2])
            T.writes(var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_T_transpose_intermediate_1[v_ax0, v_ax1, v_ax3, v_ax2]

@T.prim_func
def fused_reshape25_transpose16(lv151: T.Buffer((T.int64(2), T.int64(77), T.int64(640)), "float32"), var_T_transpose_intermediate: T.Buffer((T.int64(2), T.int64(10), T.int64(77), T.int64(64)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_T_reshape_intermediate = T.alloc_buffer((T.int64(2), T.int64(77), T.int64(10), T.int64(64)))
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(77), T.int64(10), T.int64(64)):
        with T.block("T_reshape"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(lv151[(((v_ax2 * T.int64(64) + v_ax3) // T.int64(640) + v_ax1) // T.int64(77) + v_ax0) % T.int64(2), ((v_ax2 * T.int64(64) + v_ax3) // T.int64(640) + v_ax1) % T.int64(77), (v_ax2 * T.int64(64) + v_ax3) % T.int64(640)])
            T.writes(var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = lv151[(((v_ax2 * T.int64(64) + v_ax3) // T.int64(640) + v_ax1) // T.int64(77) + v_ax0) % T.int64(2), ((v_ax2 * T.int64(64) + v_ax3) // T.int64(640) + v_ax1) % T.int64(77), (v_ax2 * T.int64(64) + v_ax3) % T.int64(640)]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(10), T.int64(77), T.int64(64)):
        with T.block("T_transpose"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_reshape_intermediate[v_ax0, v_ax2, v_ax1, v_ax3])
            T.writes(var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_T_reshape_intermediate[v_ax0, v_ax2, v_ax1, v_ax3]

@T.prim_func
def fused_reshape25_transpose16_transpose17(lv149: T.Buffer((T.int64(2), T.int64(77), T.int64(640)), "float32"), var_T_transpose_intermediate: T.Buffer((T.int64(2), T.int64(10), T.int64(64), T.int64(77)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_T_reshape_intermediate = T.alloc_buffer((T.int64(2), T.int64(77), T.int64(10), T.int64(64)))
    var_T_transpose_intermediate_1 = T.alloc_buffer((T.int64(2), T.int64(10), T.int64(77), T.int64(64)))
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(77), T.int64(10), T.int64(64)):
        with T.block("T_reshape"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(lv149[(((v_ax2 * T.int64(64) + v_ax3) // T.int64(640) + v_ax1) // T.int64(77) + v_ax0) % T.int64(2), ((v_ax2 * T.int64(64) + v_ax3) // T.int64(640) + v_ax1) % T.int64(77), (v_ax2 * T.int64(64) + v_ax3) % T.int64(640)])
            T.writes(var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = lv149[(((v_ax2 * T.int64(64) + v_ax3) // T.int64(640) + v_ax1) // T.int64(77) + v_ax0) % T.int64(2), ((v_ax2 * T.int64(64) + v_ax3) // T.int64(640) + v_ax1) % T.int64(77), (v_ax2 * T.int64(64) + v_ax3) % T.int64(640)]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(10), T.int64(77), T.int64(64)):
        with T.block("T_transpose"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_reshape_intermediate[v_ax0, v_ax2, v_ax1, v_ax3])
            T.writes(var_T_transpose_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_transpose_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3] = var_T_reshape_intermediate[v_ax0, v_ax2, v_ax1, v_ax3]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(10), T.int64(64), T.int64(77)):
        with T.block("T_transpose_1"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_transpose_intermediate_1[v_ax0, v_ax1, v_ax3, v_ax2])
            T.writes(var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_T_transpose_intermediate_1[v_ax0, v_ax1, v_ax3, v_ax2]

@T.prim_func
def fused_reshape26_transpose20_add15(lv254: T.Buffer((T.int64(2), T.int64(4096), T.int64(640)), "float32"), lv111: T.Buffer((T.int64(2), T.int64(640), T.int64(64), T.int64(64)), "float32"), var_T_add_intermediate: T.Buffer((T.int64(2), T.int64(640), T.int64(64), T.int64(64)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_T_reshape_intermediate = T.alloc_buffer((T.int64(2), T.int64(64), T.int64(64), T.int64(640)))
    var_T_transpose_intermediate = T.alloc_buffer((T.int64(2), T.int64(640), T.int64(64), T.int64(64)))
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(64), T.int64(64), T.int64(640)):
        with T.block("T_reshape"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(lv254[((v_ax1 * T.int64(64) + v_ax3 // T.int64(640) + v_ax2) // T.int64(4096) + v_ax0) % T.int64(2), (v_ax1 * T.int64(64) + v_ax3 // T.int64(640) + v_ax2) % T.int64(4096), v_ax3 % T.int64(640)])
            T.writes(var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = lv254[((v_ax1 * T.int64(64) + v_ax3 // T.int64(640) + v_ax2) // T.int64(4096) + v_ax0) % T.int64(2), (v_ax1 * T.int64(64) + v_ax3 // T.int64(640) + v_ax2) % T.int64(4096), v_ax3 % T.int64(640)]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(640), T.int64(64), T.int64(64)):
        with T.block("T_transpose"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_reshape_intermediate[v_ax0, v_ax2, v_ax3, v_ax1])
            T.writes(var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_T_reshape_intermediate[v_ax0, v_ax2, v_ax3, v_ax1]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(640), T.int64(64), T.int64(64)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], lv111[v_ax0, v_ax1, v_ax2, v_ax3])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] + lv111[v_ax0, v_ax1, v_ax2, v_ax3]

@T.prim_func
def fused_reshape26_transpose20_add15_concatenate7(lv4835: T.Buffer((T.int64(2), T.int64(4096), T.int64(640)), "float32"), lv4692: T.Buffer((T.int64(2), T.int64(640), T.int64(64), T.int64(64)), "float32"), lv257: T.Buffer((T.int64(2), T.int64(640), T.int64(64), T.int64(64)), "float32"), var_T_concat_intermediate: T.Buffer((T.int64(2), T.int64(1280), T.int64(64), T.int64(64)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_T_reshape_intermediate = T.alloc_buffer((T.int64(2), T.int64(64), T.int64(64), T.int64(640)))
    var_T_transpose_intermediate = T.alloc_buffer((T.int64(2), T.int64(640), T.int64(64), T.int64(64)))
    var_T_add_intermediate = T.alloc_buffer((T.int64(2), T.int64(640), T.int64(64), T.int64(64)))
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(64), T.int64(64), T.int64(640)):
        with T.block("T_reshape"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(lv4835[((v_ax1 * T.int64(64) + v_ax3 // T.int64(640) + v_ax2) // T.int64(4096) + v_ax0) % T.int64(2), (v_ax1 * T.int64(64) + v_ax3 // T.int64(640) + v_ax2) % T.int64(4096), v_ax3 % T.int64(640)])
            T.writes(var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = lv4835[((v_ax1 * T.int64(64) + v_ax3 // T.int64(640) + v_ax2) // T.int64(4096) + v_ax0) % T.int64(2), (v_ax1 * T.int64(64) + v_ax3 // T.int64(640) + v_ax2) % T.int64(4096), v_ax3 % T.int64(640)]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(640), T.int64(64), T.int64(64)):
        with T.block("T_transpose"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_reshape_intermediate[v_ax0, v_ax2, v_ax3, v_ax1])
            T.writes(var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_T_reshape_intermediate[v_ax0, v_ax2, v_ax3, v_ax1]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(640), T.int64(64), T.int64(64)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], lv4692[v_ax0, v_ax1, v_ax2, v_ax3])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] + lv4692[v_ax0, v_ax1, v_ax2, v_ax3]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(1280), T.int64(64), T.int64(64)):
        with T.block("T_concat"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(lv257[v_ax0, v_ax1 - T.int64(640), v_ax2, v_ax3], var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            T.writes(var_T_concat_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_concat_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = T.if_then_else(T.int64(640) <= v_ax1, lv257[v_ax0, v_ax1 - T.int64(640), v_ax2, v_ax3], var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])

@T.prim_func
def fused_reshape26_transpose20_add15_concatenate8(lv5004: T.Buffer((T.int64(2), T.int64(4096), T.int64(640)), "float32"), lv4861: T.Buffer((T.int64(2), T.int64(640), T.int64(64), T.int64(64)), "float32"), lv89: T.Buffer((T.int64(2), T.int64(320), T.int64(64), T.int64(64)), "float32"), var_T_concat_intermediate: T.Buffer((T.int64(2), T.int64(960), T.int64(64), T.int64(64)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_T_reshape_intermediate = T.alloc_buffer((T.int64(2), T.int64(64), T.int64(64), T.int64(640)))
    var_T_transpose_intermediate = T.alloc_buffer((T.int64(2), T.int64(640), T.int64(64), T.int64(64)))
    var_T_add_intermediate = T.alloc_buffer((T.int64(2), T.int64(640), T.int64(64), T.int64(64)))
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(64), T.int64(64), T.int64(640)):
        with T.block("T_reshape"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(lv5004[((v_ax1 * T.int64(64) + v_ax3 // T.int64(640) + v_ax2) // T.int64(4096) + v_ax0) % T.int64(2), (v_ax1 * T.int64(64) + v_ax3 // T.int64(640) + v_ax2) % T.int64(4096), v_ax3 % T.int64(640)])
            T.writes(var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = lv5004[((v_ax1 * T.int64(64) + v_ax3 // T.int64(640) + v_ax2) // T.int64(4096) + v_ax0) % T.int64(2), (v_ax1 * T.int64(64) + v_ax3 // T.int64(640) + v_ax2) % T.int64(4096), v_ax3 % T.int64(640)]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(640), T.int64(64), T.int64(64)):
        with T.block("T_transpose"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_reshape_intermediate[v_ax0, v_ax2, v_ax3, v_ax1])
            T.writes(var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_T_reshape_intermediate[v_ax0, v_ax2, v_ax3, v_ax1]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(640), T.int64(64), T.int64(64)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], lv4861[v_ax0, v_ax1, v_ax2, v_ax3])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] + lv4861[v_ax0, v_ax1, v_ax2, v_ax3]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(960), T.int64(64), T.int64(64)):
        with T.block("T_concat"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(lv89[v_ax0, v_ax1 - T.int64(640), v_ax2, v_ax3], var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            T.writes(var_T_concat_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_concat_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = T.if_then_else(T.int64(640) <= v_ax1, lv89[v_ax0, v_ax1 - T.int64(640), v_ax2, v_ax3], var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])

@T.prim_func
def fused_reshape26_transpose20_add15_resize2d1(lv5173: T.Buffer((T.int64(2), T.int64(4096), T.int64(640)), "float32"), lv5030: T.Buffer((T.int64(2), T.int64(640), T.int64(64), T.int64(64)), "float32"), var_resize_intermediate: T.Buffer((T.int64(2), T.int64(640), T.int64(128), T.int64(128)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_T_reshape_intermediate = T.alloc_buffer((T.int64(2), T.int64(64), T.int64(64), T.int64(640)))
    var_T_transpose_intermediate = T.alloc_buffer((T.int64(2), T.int64(640), T.int64(64), T.int64(64)))
    var_T_add_intermediate = T.alloc_buffer((T.int64(2), T.int64(640), T.int64(64), T.int64(64)))
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(64), T.int64(64), T.int64(640)):
        with T.block("T_reshape"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(lv5173[((v_ax1 * T.int64(64) + v_ax3 // T.int64(640) + v_ax2) // T.int64(4096) + v_ax0) % T.int64(2), (v_ax1 * T.int64(64) + v_ax3 // T.int64(640) + v_ax2) % T.int64(4096), v_ax3 % T.int64(640)])
            T.writes(var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = lv5173[((v_ax1 * T.int64(64) + v_ax3 // T.int64(640) + v_ax2) // T.int64(4096) + v_ax0) % T.int64(2), (v_ax1 * T.int64(64) + v_ax3 // T.int64(640) + v_ax2) % T.int64(4096), v_ax3 % T.int64(640)]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(640), T.int64(64), T.int64(64)):
        with T.block("T_transpose"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_reshape_intermediate[v_ax0, v_ax2, v_ax3, v_ax1])
            T.writes(var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_T_reshape_intermediate[v_ax0, v_ax2, v_ax3, v_ax1]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(640), T.int64(64), T.int64(64)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], lv5030[v_ax0, v_ax1, v_ax2, v_ax3])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] + lv5030[v_ax0, v_ax1, v_ax2, v_ax3]
    for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(640), T.int64(128), T.int64(128)):
        with T.block("resize"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(var_T_add_intermediate[v_i0, v_i1, T.max(T.min(T.Div(v_i2, T.int64(2)), T.int64(63)), T.int64(0)), T.max(T.min(T.Div(v_i3, T.int64(2)), T.int64(63)), T.int64(0))])
            T.writes(var_resize_intermediate[v_i0, v_i1, v_i2, v_i3])
            var_resize_intermediate[v_i0, v_i1, v_i2, v_i3] = var_T_add_intermediate[v_i0, v_i1, T.max(T.min(T.Div(v_i2, T.int64(2)), T.int64(63)), T.int64(0)), T.max(T.min(T.Div(v_i3, T.int64(2)), T.int64(63)), T.int64(0))]

@T.prim_func
def fused_reshape2_reshape2_add(lv3: T.Buffer((T.int64(77), T.int64(1280)), "float32"), lv7: T.Buffer((T.int64(77), T.int64(1280)), "float32"), var_T_add_intermediate: T.Buffer((T.int64(1), T.int64(77), T.int64(1280)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_T_reshape_intermediate = T.alloc_buffer((T.int64(1), T.int64(77), T.int64(1280)))
    var_T_reshape_intermediate_1 = T.alloc_buffer((T.int64(1), T.int64(77), T.int64(1280)))
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(77), T.int64(1280)):
        with T.block("T_reshape"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(lv3[(v_ax2 // T.int64(1280) + v_ax1) % T.int64(77), v_ax2 % T.int64(1280)])
            T.writes(var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2] = lv3[(v_ax2 // T.int64(1280) + v_ax1) % T.int64(77), v_ax2 % T.int64(1280)]
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(77), T.int64(1280)):
        with T.block("T_reshape_1"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(lv7[(v_ax2 // T.int64(1280) + v_ax1) % T.int64(77), v_ax2 % T.int64(1280)])
            T.writes(var_T_reshape_intermediate_1[v_ax0, v_ax1, v_ax2])
            var_T_reshape_intermediate_1[v_ax0, v_ax1, v_ax2] = lv7[(v_ax2 // T.int64(1280) + v_ax1) % T.int64(77), v_ax2 % T.int64(1280)]
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(77), T.int64(1280)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2], var_T_reshape_intermediate_1[v_ax0, v_ax1, v_ax2])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2] = var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2] + var_T_reshape_intermediate_1[v_ax0, v_ax1, v_ax2]

@T.prim_func
def fused_reshape30_transpose22(lv456: T.Buffer((T.int64(2), T.int64(1024), T.int64(1280)), "float32"), var_T_transpose_intermediate: T.Buffer((T.int64(2), T.int64(20), T.int64(1024), T.int64(64)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_T_reshape_intermediate = T.alloc_buffer((T.int64(2), T.int64(1024), T.int64(20), T.int64(64)))
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(1024), T.int64(20), T.int64(64)):
        with T.block("T_reshape"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(lv456[(((v_ax2 * T.int64(64) + v_ax3) // T.int64(1280) + v_ax1) // T.int64(1024) + v_ax0) % T.int64(2), ((v_ax2 * T.int64(64) + v_ax3) // T.int64(1280) + v_ax1) % T.int64(1024), (v_ax2 * T.int64(64) + v_ax3) % T.int64(1280)])
            T.writes(var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = lv456[(((v_ax2 * T.int64(64) + v_ax3) // T.int64(1280) + v_ax1) // T.int64(1024) + v_ax0) % T.int64(2), ((v_ax2 * T.int64(64) + v_ax3) // T.int64(1280) + v_ax1) % T.int64(1024), (v_ax2 * T.int64(64) + v_ax3) % T.int64(1280)]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(20), T.int64(1024), T.int64(64)):
        with T.block("T_transpose"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_reshape_intermediate[v_ax0, v_ax2, v_ax1, v_ax3])
            T.writes(var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_T_reshape_intermediate[v_ax0, v_ax2, v_ax1, v_ax3]

@T.prim_func
def fused_reshape30_transpose22_transpose23(lv458: T.Buffer((T.int64(2), T.int64(1024), T.int64(1280)), "float32"), var_T_transpose_intermediate: T.Buffer((T.int64(2), T.int64(20), T.int64(64), T.int64(1024)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_T_reshape_intermediate = T.alloc_buffer((T.int64(2), T.int64(1024), T.int64(20), T.int64(64)))
    var_T_transpose_intermediate_1 = T.alloc_buffer((T.int64(2), T.int64(20), T.int64(1024), T.int64(64)))
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(1024), T.int64(20), T.int64(64)):
        with T.block("T_reshape"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(lv458[(((v_ax2 * T.int64(64) + v_ax3) // T.int64(1280) + v_ax1) // T.int64(1024) + v_ax0) % T.int64(2), ((v_ax2 * T.int64(64) + v_ax3) // T.int64(1280) + v_ax1) % T.int64(1024), (v_ax2 * T.int64(64) + v_ax3) % T.int64(1280)])
            T.writes(var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = lv458[(((v_ax2 * T.int64(64) + v_ax3) // T.int64(1280) + v_ax1) // T.int64(1024) + v_ax0) % T.int64(2), ((v_ax2 * T.int64(64) + v_ax3) // T.int64(1280) + v_ax1) % T.int64(1024), (v_ax2 * T.int64(64) + v_ax3) % T.int64(1280)]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(20), T.int64(1024), T.int64(64)):
        with T.block("T_transpose"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_reshape_intermediate[v_ax0, v_ax2, v_ax1, v_ax3])
            T.writes(var_T_transpose_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_transpose_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3] = var_T_reshape_intermediate[v_ax0, v_ax2, v_ax1, v_ax3]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(20), T.int64(64), T.int64(1024)):
        with T.block("T_transpose_1"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_transpose_intermediate_1[v_ax0, v_ax1, v_ax3, v_ax2])
            T.writes(var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_T_transpose_intermediate_1[v_ax0, v_ax1, v_ax3, v_ax2]

@T.prim_func
def fused_reshape32_transpose26(lv487: T.Buffer((T.int64(2), T.int64(77), T.int64(1280)), "float32"), var_T_transpose_intermediate: T.Buffer((T.int64(2), T.int64(20), T.int64(77), T.int64(64)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_T_reshape_intermediate = T.alloc_buffer((T.int64(2), T.int64(77), T.int64(20), T.int64(64)))
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(77), T.int64(20), T.int64(64)):
        with T.block("T_reshape"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(lv487[(((v_ax2 * T.int64(64) + v_ax3) // T.int64(1280) + v_ax1) // T.int64(77) + v_ax0) % T.int64(2), ((v_ax2 * T.int64(64) + v_ax3) // T.int64(1280) + v_ax1) % T.int64(77), (v_ax2 * T.int64(64) + v_ax3) % T.int64(1280)])
            T.writes(var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = lv487[(((v_ax2 * T.int64(64) + v_ax3) // T.int64(1280) + v_ax1) // T.int64(77) + v_ax0) % T.int64(2), ((v_ax2 * T.int64(64) + v_ax3) // T.int64(1280) + v_ax1) % T.int64(77), (v_ax2 * T.int64(64) + v_ax3) % T.int64(1280)]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(20), T.int64(77), T.int64(64)):
        with T.block("T_transpose"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_reshape_intermediate[v_ax0, v_ax2, v_ax1, v_ax3])
            T.writes(var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_T_reshape_intermediate[v_ax0, v_ax2, v_ax1, v_ax3]

@T.prim_func
def fused_reshape32_transpose26_transpose27(lv485: T.Buffer((T.int64(2), T.int64(77), T.int64(1280)), "float32"), var_T_transpose_intermediate: T.Buffer((T.int64(2), T.int64(20), T.int64(64), T.int64(77)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_T_reshape_intermediate = T.alloc_buffer((T.int64(2), T.int64(77), T.int64(20), T.int64(64)))
    var_T_transpose_intermediate_1 = T.alloc_buffer((T.int64(2), T.int64(20), T.int64(77), T.int64(64)))
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(77), T.int64(20), T.int64(64)):
        with T.block("T_reshape"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(lv485[(((v_ax2 * T.int64(64) + v_ax3) // T.int64(1280) + v_ax1) // T.int64(77) + v_ax0) % T.int64(2), ((v_ax2 * T.int64(64) + v_ax3) // T.int64(1280) + v_ax1) % T.int64(77), (v_ax2 * T.int64(64) + v_ax3) % T.int64(1280)])
            T.writes(var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = lv485[(((v_ax2 * T.int64(64) + v_ax3) // T.int64(1280) + v_ax1) // T.int64(77) + v_ax0) % T.int64(2), ((v_ax2 * T.int64(64) + v_ax3) // T.int64(1280) + v_ax1) % T.int64(77), (v_ax2 * T.int64(64) + v_ax3) % T.int64(1280)]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(20), T.int64(77), T.int64(64)):
        with T.block("T_transpose"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_reshape_intermediate[v_ax0, v_ax2, v_ax1, v_ax3])
            T.writes(var_T_transpose_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_transpose_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3] = var_T_reshape_intermediate[v_ax0, v_ax2, v_ax1, v_ax3]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(20), T.int64(64), T.int64(77)):
        with T.block("T_transpose_1"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_transpose_intermediate_1[v_ax0, v_ax1, v_ax3, v_ax2])
            T.writes(var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_T_transpose_intermediate_1[v_ax0, v_ax1, v_ax3, v_ax2]

@T.prim_func
def fused_reshape33_transpose29_add22(lv1126: T.Buffer((T.int64(2), T.int64(1024), T.int64(1280)), "float32"), lv447: T.Buffer((T.int64(2), T.int64(1280), T.int64(32), T.int64(32)), "float32"), var_T_add_intermediate: T.Buffer((T.int64(2), T.int64(1280), T.int64(32), T.int64(32)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_T_reshape_intermediate = T.alloc_buffer((T.int64(2), T.int64(32), T.int64(32), T.int64(1280)))
    var_T_transpose_intermediate = T.alloc_buffer((T.int64(2), T.int64(1280), T.int64(32), T.int64(32)))
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(32), T.int64(32), T.int64(1280)):
        with T.block("T_reshape"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(lv1126[((v_ax1 * T.int64(32) + v_ax3 // T.int64(1280) + v_ax2) // T.int64(1024) + v_ax0) % T.int64(2), (v_ax1 * T.int64(32) + v_ax3 // T.int64(1280) + v_ax2) % T.int64(1024), v_ax3 % T.int64(1280)])
            T.writes(var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = lv1126[((v_ax1 * T.int64(32) + v_ax3 // T.int64(1280) + v_ax2) // T.int64(1024) + v_ax0) % T.int64(2), (v_ax1 * T.int64(32) + v_ax3 // T.int64(1280) + v_ax2) % T.int64(1024), v_ax3 % T.int64(1280)]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(1280), T.int64(32), T.int64(32)):
        with T.block("T_transpose"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_reshape_intermediate[v_ax0, v_ax2, v_ax3, v_ax1])
            T.writes(var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_T_reshape_intermediate[v_ax0, v_ax2, v_ax3, v_ax1]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(1280), T.int64(32), T.int64(32)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], lv447[v_ax0, v_ax1, v_ax2, v_ax3])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] + lv447[v_ax0, v_ax1, v_ax2, v_ax3]

@T.prim_func
def fused_reshape33_transpose29_add22_concatenate4(lv3252: T.Buffer((T.int64(2), T.int64(1024), T.int64(1280)), "float32"), lv2573: T.Buffer((T.int64(2), T.int64(1280), T.int64(32), T.int64(32)), "float32"), lv1129: T.Buffer((T.int64(2), T.int64(1280), T.int64(32), T.int64(32)), "float32"), var_T_concat_intermediate: T.Buffer((T.int64(2), T.int64(2560), T.int64(32), T.int64(32)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_T_reshape_intermediate = T.alloc_buffer((T.int64(2), T.int64(32), T.int64(32), T.int64(1280)))
    var_T_transpose_intermediate = T.alloc_buffer((T.int64(2), T.int64(1280), T.int64(32), T.int64(32)))
    var_T_add_intermediate = T.alloc_buffer((T.int64(2), T.int64(1280), T.int64(32), T.int64(32)))
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(32), T.int64(32), T.int64(1280)):
        with T.block("T_reshape"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(lv3252[((v_ax1 * T.int64(32) + v_ax3 // T.int64(1280) + v_ax2) // T.int64(1024) + v_ax0) % T.int64(2), (v_ax1 * T.int64(32) + v_ax3 // T.int64(1280) + v_ax2) % T.int64(1024), v_ax3 % T.int64(1280)])
            T.writes(var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = lv3252[((v_ax1 * T.int64(32) + v_ax3 // T.int64(1280) + v_ax2) // T.int64(1024) + v_ax0) % T.int64(2), (v_ax1 * T.int64(32) + v_ax3 // T.int64(1280) + v_ax2) % T.int64(1024), v_ax3 % T.int64(1280)]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(1280), T.int64(32), T.int64(32)):
        with T.block("T_transpose"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_reshape_intermediate[v_ax0, v_ax2, v_ax3, v_ax1])
            T.writes(var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_T_reshape_intermediate[v_ax0, v_ax2, v_ax3, v_ax1]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(1280), T.int64(32), T.int64(32)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], lv2573[v_ax0, v_ax1, v_ax2, v_ax3])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] + lv2573[v_ax0, v_ax1, v_ax2, v_ax3]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(2560), T.int64(32), T.int64(32)):
        with T.block("T_concat"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(lv1129[v_ax0, v_ax1 - T.int64(1280), v_ax2, v_ax3], var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            T.writes(var_T_concat_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_concat_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = T.if_then_else(T.int64(1280) <= v_ax1, lv1129[v_ax0, v_ax1 - T.int64(1280), v_ax2, v_ax3], var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])

@T.prim_func
def fused_reshape33_transpose29_add22_concatenate5(lv3957: T.Buffer((T.int64(2), T.int64(1024), T.int64(1280)), "float32"), lv3278: T.Buffer((T.int64(2), T.int64(1280), T.int64(32), T.int64(32)), "float32"), lv425: T.Buffer((T.int64(2), T.int64(640), T.int64(32), T.int64(32)), "float32"), var_T_concat_intermediate: T.Buffer((T.int64(2), T.int64(1920), T.int64(32), T.int64(32)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_T_reshape_intermediate = T.alloc_buffer((T.int64(2), T.int64(32), T.int64(32), T.int64(1280)))
    var_T_transpose_intermediate = T.alloc_buffer((T.int64(2), T.int64(1280), T.int64(32), T.int64(32)))
    var_T_add_intermediate = T.alloc_buffer((T.int64(2), T.int64(1280), T.int64(32), T.int64(32)))
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(32), T.int64(32), T.int64(1280)):
        with T.block("T_reshape"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(lv3957[((v_ax1 * T.int64(32) + v_ax3 // T.int64(1280) + v_ax2) // T.int64(1024) + v_ax0) % T.int64(2), (v_ax1 * T.int64(32) + v_ax3 // T.int64(1280) + v_ax2) % T.int64(1024), v_ax3 % T.int64(1280)])
            T.writes(var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = lv3957[((v_ax1 * T.int64(32) + v_ax3 // T.int64(1280) + v_ax2) // T.int64(1024) + v_ax0) % T.int64(2), (v_ax1 * T.int64(32) + v_ax3 // T.int64(1280) + v_ax2) % T.int64(1024), v_ax3 % T.int64(1280)]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(1280), T.int64(32), T.int64(32)):
        with T.block("T_transpose"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_reshape_intermediate[v_ax0, v_ax2, v_ax3, v_ax1])
            T.writes(var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_T_reshape_intermediate[v_ax0, v_ax2, v_ax3, v_ax1]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(1280), T.int64(32), T.int64(32)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], lv3278[v_ax0, v_ax1, v_ax2, v_ax3])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] + lv3278[v_ax0, v_ax1, v_ax2, v_ax3]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(1920), T.int64(32), T.int64(32)):
        with T.block("T_concat"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(lv425[v_ax0, v_ax1 - T.int64(1280), v_ax2, v_ax3], var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            T.writes(var_T_concat_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_concat_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = T.if_then_else(T.int64(1280) <= v_ax1, lv425[v_ax0, v_ax1 - T.int64(1280), v_ax2, v_ax3], var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])

@T.prim_func
def fused_reshape33_transpose29_add22_resize2d(lv4662: T.Buffer((T.int64(2), T.int64(1024), T.int64(1280)), "float32"), lv3983: T.Buffer((T.int64(2), T.int64(1280), T.int64(32), T.int64(32)), "float32"), var_resize_intermediate: T.Buffer((T.int64(2), T.int64(1280), T.int64(64), T.int64(64)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_T_reshape_intermediate = T.alloc_buffer((T.int64(2), T.int64(32), T.int64(32), T.int64(1280)))
    var_T_transpose_intermediate = T.alloc_buffer((T.int64(2), T.int64(1280), T.int64(32), T.int64(32)))
    var_T_add_intermediate = T.alloc_buffer((T.int64(2), T.int64(1280), T.int64(32), T.int64(32)))
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(32), T.int64(32), T.int64(1280)):
        with T.block("T_reshape"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(lv4662[((v_ax1 * T.int64(32) + v_ax3 // T.int64(1280) + v_ax2) // T.int64(1024) + v_ax0) % T.int64(2), (v_ax1 * T.int64(32) + v_ax3 // T.int64(1280) + v_ax2) % T.int64(1024), v_ax3 % T.int64(1280)])
            T.writes(var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = lv4662[((v_ax1 * T.int64(32) + v_ax3 // T.int64(1280) + v_ax2) // T.int64(1024) + v_ax0) % T.int64(2), (v_ax1 * T.int64(32) + v_ax3 // T.int64(1280) + v_ax2) % T.int64(1024), v_ax3 % T.int64(1280)]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(1280), T.int64(32), T.int64(32)):
        with T.block("T_transpose"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_reshape_intermediate[v_ax0, v_ax2, v_ax3, v_ax1])
            T.writes(var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_T_reshape_intermediate[v_ax0, v_ax2, v_ax3, v_ax1]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(1280), T.int64(32), T.int64(32)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], lv3983[v_ax0, v_ax1, v_ax2, v_ax3])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] + lv3983[v_ax0, v_ax1, v_ax2, v_ax3]
    for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(1280), T.int64(64), T.int64(64)):
        with T.block("resize"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(var_T_add_intermediate[v_i0, v_i1, T.max(T.min(T.Div(v_i2, T.int64(2)), T.int64(31)), T.int64(0)), T.max(T.min(T.Div(v_i3, T.int64(2)), T.int64(31)), T.int64(0))])
            T.writes(var_resize_intermediate[v_i0, v_i1, v_i2, v_i3])
            var_resize_intermediate[v_i0, v_i1, v_i2, v_i3] = var_T_add_intermediate[v_i0, v_i1, T.max(T.min(T.Div(v_i2, T.int64(2)), T.int64(31)), T.int64(0)), T.max(T.min(T.Div(v_i3, T.int64(2)), T.int64(31)), T.int64(0))]

@T.prim_func
def fused_reshape36_transpose30_transpose31(lv18: T.Buffer((T.int64(1), T.int64(512), T.int64(128), T.int64(128)), "float32"), var_T_transpose_intermediate: T.Buffer((T.int64(1), T.int64(512), T.int64(16384)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_T_reshape_intermediate = T.alloc_buffer((T.int64(1), T.int64(512), T.int64(16384)))
    var_T_transpose_intermediate_1 = T.alloc_buffer((T.int64(1), T.int64(16384), T.int64(512)))
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(512), T.int64(16384)):
        with T.block("T_reshape"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(lv18[T.int64(0), (v_ax2 // T.int64(16384) + v_ax1) % T.int64(512), v_ax2 % T.int64(16384) // T.int64(128), v_ax2 % T.int64(128)])
            T.writes(var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2] = lv18[T.int64(0), (v_ax2 // T.int64(16384) + v_ax1) % T.int64(512), v_ax2 % T.int64(16384) // T.int64(128), v_ax2 % T.int64(128)]
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(16384), T.int64(512)):
        with T.block("T_transpose"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_T_reshape_intermediate[v_ax0, v_ax2, v_ax1])
            T.writes(var_T_transpose_intermediate_1[v_ax0, v_ax1, v_ax2])
            var_T_transpose_intermediate_1[v_ax0, v_ax1, v_ax2] = var_T_reshape_intermediate[v_ax0, v_ax2, v_ax1]
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(512), T.int64(16384)):
        with T.block("T_transpose_1"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_T_transpose_intermediate_1[v_ax0, v_ax2, v_ax1])
            T.writes(var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2] = var_T_transpose_intermediate_1[v_ax0, v_ax2, v_ax1]

@T.prim_func
def fused_reshape37_transpose33(lv26: T.Buffer((T.int64(1), T.int64(16384), T.int64(512)), "float32"), var_T_transpose_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(16384), T.int64(512)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_T_reshape_intermediate = T.alloc_buffer((T.int64(1), T.int64(16384), T.int64(1), T.int64(512)))
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(16384), T.int64(1), T.int64(512)):
        with T.block("T_reshape"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(lv26[T.int64(0), (v_ax3 // T.int64(512) + v_ax1 + v_ax2) % T.int64(16384), v_ax3 % T.int64(512)])
            T.writes(var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = lv26[T.int64(0), (v_ax3 // T.int64(512) + v_ax1 + v_ax2) % T.int64(16384), v_ax3 % T.int64(512)]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(1), T.int64(16384), T.int64(512)):
        with T.block("T_transpose"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_reshape_intermediate[v_ax0, v_ax2, v_ax1, v_ax3])
            T.writes(var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_T_reshape_intermediate[v_ax0, v_ax2, v_ax1, v_ax3]

@T.prim_func
def fused_reshape37_transpose33_transpose34(lv29: T.Buffer((T.int64(1), T.int64(16384), T.int64(512)), "float32"), var_T_transpose_intermediate: T.Buffer((T.int64(1), T.int64(1), T.int64(512), T.int64(16384)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_T_reshape_intermediate = T.alloc_buffer((T.int64(1), T.int64(16384), T.int64(1), T.int64(512)))
    var_T_transpose_intermediate_1 = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(16384), T.int64(512)))
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(16384), T.int64(1), T.int64(512)):
        with T.block("T_reshape"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(lv29[T.int64(0), (v_ax3 // T.int64(512) + v_ax1 + v_ax2) % T.int64(16384), v_ax3 % T.int64(512)])
            T.writes(var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = lv29[T.int64(0), (v_ax3 // T.int64(512) + v_ax1 + v_ax2) % T.int64(16384), v_ax3 % T.int64(512)]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(1), T.int64(16384), T.int64(512)):
        with T.block("T_transpose"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_reshape_intermediate[v_ax0, v_ax2, v_ax1, v_ax3])
            T.writes(var_T_transpose_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_transpose_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3] = var_T_reshape_intermediate[v_ax0, v_ax2, v_ax1, v_ax3]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(1), T.int64(512), T.int64(16384)):
        with T.block("T_transpose_1"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_transpose_intermediate_1[v_ax0, v_ax1, v_ax3, v_ax2])
            T.writes(var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_T_transpose_intermediate_1[v_ax0, v_ax1, v_ax3, v_ax2]

@T.prim_func
def fused_reshape43_reshape43_add44(lv3: T.Buffer((T.int64(77), T.int64(768)), "float32"), lv7: T.Buffer((T.int64(77), T.int64(768)), "float32"), var_T_add_intermediate: T.Buffer((T.int64(1), T.int64(77), T.int64(768)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_T_reshape_intermediate = T.alloc_buffer((T.int64(1), T.int64(77), T.int64(768)))
    var_T_reshape_intermediate_1 = T.alloc_buffer((T.int64(1), T.int64(77), T.int64(768)))
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(77), T.int64(768)):
        with T.block("T_reshape"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(lv3[(v_ax2 // T.int64(768) + v_ax1) % T.int64(77), v_ax2 % T.int64(768)])
            T.writes(var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2] = lv3[(v_ax2 // T.int64(768) + v_ax1) % T.int64(77), v_ax2 % T.int64(768)]
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(77), T.int64(768)):
        with T.block("T_reshape_1"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(lv7[(v_ax2 // T.int64(768) + v_ax1) % T.int64(77), v_ax2 % T.int64(768)])
            T.writes(var_T_reshape_intermediate_1[v_ax0, v_ax1, v_ax2])
            var_T_reshape_intermediate_1[v_ax0, v_ax1, v_ax2] = lv7[(v_ax2 // T.int64(768) + v_ax1) % T.int64(77), v_ax2 % T.int64(768)]
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(77), T.int64(768)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2], var_T_reshape_intermediate_1[v_ax0, v_ax1, v_ax2])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2] = var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2] + var_T_reshape_intermediate_1[v_ax0, v_ax1, v_ax2]

@T.prim_func
def fused_reshape44_transpose38_reshape45(lv33: T.Buffer((T.int64(1), T.int64(77), T.int64(768)), "float32"), var_T_reshape_intermediate: T.Buffer((T.int64(12), T.int64(77), T.int64(64)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_T_reshape_intermediate_1 = T.alloc_buffer((T.int64(1), T.int64(77), T.int64(12), T.int64(64)))
    var_T_transpose_intermediate = T.alloc_buffer((T.int64(1), T.int64(12), T.int64(77), T.int64(64)))
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(77), T.int64(12), T.int64(64)):
        with T.block("T_reshape"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(lv33[T.int64(0), ((v_ax2 * T.int64(64) + v_ax3) // T.int64(768) + v_ax1) % T.int64(77), (v_ax2 * T.int64(64) + v_ax3) % T.int64(768)])
            T.writes(var_T_reshape_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_reshape_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3] = lv33[T.int64(0), ((v_ax2 * T.int64(64) + v_ax3) // T.int64(768) + v_ax1) % T.int64(77), (v_ax2 * T.int64(64) + v_ax3) % T.int64(768)]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(12), T.int64(77), T.int64(64)):
        with T.block("T_transpose"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_reshape_intermediate_1[v_ax0, v_ax2, v_ax1, v_ax3])
            T.writes(var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_T_reshape_intermediate_1[v_ax0, v_ax2, v_ax1, v_ax3]
    for ax0, ax1, ax2 in T.grid(T.int64(12), T.int64(77), T.int64(64)):
        with T.block("T_reshape_1"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_T_transpose_intermediate[T.int64(0), ((v_ax2 // T.int64(64) + v_ax1) // T.int64(77) + v_ax0) % T.int64(12), (v_ax2 // T.int64(64) + v_ax1) % T.int64(77), v_ax2 % T.int64(64)])
            T.writes(var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2] = var_T_transpose_intermediate[T.int64(0), ((v_ax2 // T.int64(64) + v_ax1) // T.int64(77) + v_ax0) % T.int64(12), (v_ax2 // T.int64(64) + v_ax1) % T.int64(77), v_ax2 % T.int64(64)]

@T.prim_func
def fused_reshape44_transpose38_reshape45_transpose39(lv28: T.Buffer((T.int64(1), T.int64(77), T.int64(768)), "float32"), var_T_transpose_intermediate: T.Buffer((T.int64(12), T.int64(64), T.int64(77)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_T_reshape_intermediate = T.alloc_buffer((T.int64(1), T.int64(77), T.int64(12), T.int64(64)))
    var_T_transpose_intermediate_1 = T.alloc_buffer((T.int64(1), T.int64(12), T.int64(77), T.int64(64)))
    var_T_reshape_intermediate_1 = T.alloc_buffer((T.int64(12), T.int64(77), T.int64(64)))
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(77), T.int64(12), T.int64(64)):
        with T.block("T_reshape"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(lv28[T.int64(0), ((v_ax2 * T.int64(64) + v_ax3) // T.int64(768) + v_ax1) % T.int64(77), (v_ax2 * T.int64(64) + v_ax3) % T.int64(768)])
            T.writes(var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = lv28[T.int64(0), ((v_ax2 * T.int64(64) + v_ax3) // T.int64(768) + v_ax1) % T.int64(77), (v_ax2 * T.int64(64) + v_ax3) % T.int64(768)]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(12), T.int64(77), T.int64(64)):
        with T.block("T_transpose"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_reshape_intermediate[v_ax0, v_ax2, v_ax1, v_ax3])
            T.writes(var_T_transpose_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_transpose_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3] = var_T_reshape_intermediate[v_ax0, v_ax2, v_ax1, v_ax3]
    for ax0, ax1, ax2 in T.grid(T.int64(12), T.int64(77), T.int64(64)):
        with T.block("T_reshape_1"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_T_transpose_intermediate_1[T.int64(0), ((v_ax2 // T.int64(64) + v_ax1) // T.int64(77) + v_ax0) % T.int64(12), (v_ax2 // T.int64(64) + v_ax1) % T.int64(77), v_ax2 % T.int64(64)])
            T.writes(var_T_reshape_intermediate_1[v_ax0, v_ax1, v_ax2])
            var_T_reshape_intermediate_1[v_ax0, v_ax1, v_ax2] = var_T_transpose_intermediate_1[T.int64(0), ((v_ax2 // T.int64(64) + v_ax1) // T.int64(77) + v_ax0) % T.int64(12), (v_ax2 // T.int64(64) + v_ax1) % T.int64(77), v_ax2 % T.int64(64)]
    for ax0, ax1, ax2 in T.grid(T.int64(12), T.int64(64), T.int64(77)):
        with T.block("T_transpose_1"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_T_reshape_intermediate_1[v_ax0, v_ax2, v_ax1])
            T.writes(var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2] = var_T_reshape_intermediate_1[v_ax0, v_ax2, v_ax1]

@T.prim_func
def fused_reshape46_add46_reshape47(lv42: T.Buffer((T.int64(12), T.int64(77), T.int64(77)), "float32"), param_0: T.Buffer((T.int64(1), T.int64(1), T.int64(77), T.int64(77)), "float32"), var_T_reshape_intermediate: T.Buffer((T.int64(12), T.int64(77), T.int64(77)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_T_reshape_intermediate_1 = T.alloc_buffer((T.int64(1), T.int64(12), T.int64(77), T.int64(77)))
    var_T_add_intermediate = T.alloc_buffer((T.int64(1), T.int64(12), T.int64(77), T.int64(77)))
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(12), T.int64(77), T.int64(77)):
        with T.block("T_reshape"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(lv42[((v_ax3 // T.int64(77) + v_ax2) // T.int64(77) + v_ax1) % T.int64(12), (v_ax3 // T.int64(77) + v_ax2) % T.int64(77), v_ax3 % T.int64(77)])
            T.writes(var_T_reshape_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_reshape_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3] = lv42[((v_ax3 // T.int64(77) + v_ax2) // T.int64(77) + v_ax1) % T.int64(12), (v_ax3 // T.int64(77) + v_ax2) % T.int64(77), v_ax3 % T.int64(77)]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(12), T.int64(77), T.int64(77)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_reshape_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3], param_0[v_ax0, T.int64(0), v_ax2, v_ax3])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_T_reshape_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3] + param_0[v_ax0, T.int64(0), v_ax2, v_ax3]
    for ax0, ax1, ax2 in T.grid(T.int64(12), T.int64(77), T.int64(77)):
        with T.block("T_reshape_1"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_T_add_intermediate[T.int64(0), ((v_ax2 // T.int64(77) + v_ax1) // T.int64(77) + v_ax0) % T.int64(12), (v_ax2 // T.int64(77) + v_ax1) % T.int64(77), v_ax2 % T.int64(77)])
            T.writes(var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2] = var_T_add_intermediate[T.int64(0), ((v_ax2 // T.int64(77) + v_ax1) // T.int64(77) + v_ax0) % T.int64(12), (v_ax2 // T.int64(77) + v_ax1) % T.int64(77), v_ax2 % T.int64(77)]

@T.prim_func
def fused_reshape48_transpose40_reshape49(lv47: T.Buffer((T.int64(12), T.int64(77), T.int64(64)), "float32"), var_T_reshape_intermediate: T.Buffer((T.int64(1), T.int64(77), T.int64(768)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_T_reshape_intermediate_1 = T.alloc_buffer((T.int64(1), T.int64(12), T.int64(77), T.int64(64)))
    var_T_transpose_intermediate = T.alloc_buffer((T.int64(1), T.int64(77), T.int64(12), T.int64(64)))
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(12), T.int64(77), T.int64(64)):
        with T.block("T_reshape"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(lv47[((v_ax3 // T.int64(64) + v_ax2) // T.int64(77) + v_ax1) % T.int64(12), (v_ax3 // T.int64(64) + v_ax2) % T.int64(77), v_ax3 % T.int64(64)])
            T.writes(var_T_reshape_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_reshape_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3] = lv47[((v_ax3 // T.int64(64) + v_ax2) // T.int64(77) + v_ax1) % T.int64(12), (v_ax3 // T.int64(64) + v_ax2) % T.int64(77), v_ax3 % T.int64(64)]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(77), T.int64(12), T.int64(64)):
        with T.block("T_transpose"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_reshape_intermediate_1[v_ax0, v_ax2, v_ax1, v_ax3])
            T.writes(var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_T_reshape_intermediate_1[v_ax0, v_ax2, v_ax1, v_ax3]
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(77), T.int64(768)):
        with T.block("T_reshape_1"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_T_transpose_intermediate[T.int64(0), (v_ax2 // T.int64(768) + v_ax1) % T.int64(77), v_ax2 % T.int64(768) // T.int64(64), v_ax2 % T.int64(64)])
            T.writes(var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2] = var_T_transpose_intermediate[T.int64(0), (v_ax2 // T.int64(768) + v_ax1) % T.int64(77), v_ax2 % T.int64(768) // T.int64(64), v_ax2 % T.int64(64)]

@T.prim_func
def fused_reshape5_transpose1_reshape6(lv33: T.Buffer((T.int64(1), T.int64(77), T.int64(1280)), "float32"), var_T_reshape_intermediate: T.Buffer((T.int64(20), T.int64(77), T.int64(64)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_T_reshape_intermediate_1 = T.alloc_buffer((T.int64(1), T.int64(77), T.int64(20), T.int64(64)))
    var_T_transpose_intermediate = T.alloc_buffer((T.int64(1), T.int64(20), T.int64(77), T.int64(64)))
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(77), T.int64(20), T.int64(64)):
        with T.block("T_reshape"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(lv33[T.int64(0), ((v_ax2 * T.int64(64) + v_ax3) // T.int64(1280) + v_ax1) % T.int64(77), (v_ax2 * T.int64(64) + v_ax3) % T.int64(1280)])
            T.writes(var_T_reshape_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_reshape_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3] = lv33[T.int64(0), ((v_ax2 * T.int64(64) + v_ax3) // T.int64(1280) + v_ax1) % T.int64(77), (v_ax2 * T.int64(64) + v_ax3) % T.int64(1280)]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(20), T.int64(77), T.int64(64)):
        with T.block("T_transpose"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_reshape_intermediate_1[v_ax0, v_ax2, v_ax1, v_ax3])
            T.writes(var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_T_reshape_intermediate_1[v_ax0, v_ax2, v_ax1, v_ax3]
    for ax0, ax1, ax2 in T.grid(T.int64(20), T.int64(77), T.int64(64)):
        with T.block("T_reshape_1"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_T_transpose_intermediate[T.int64(0), ((v_ax2 // T.int64(64) + v_ax1) // T.int64(77) + v_ax0) % T.int64(20), (v_ax2 // T.int64(64) + v_ax1) % T.int64(77), v_ax2 % T.int64(64)])
            T.writes(var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2] = var_T_transpose_intermediate[T.int64(0), ((v_ax2 // T.int64(64) + v_ax1) // T.int64(77) + v_ax0) % T.int64(20), (v_ax2 // T.int64(64) + v_ax1) % T.int64(77), v_ax2 % T.int64(64)]

@T.prim_func
def fused_reshape5_transpose1_reshape6_transpose2(lv28: T.Buffer((T.int64(1), T.int64(77), T.int64(1280)), "float32"), var_T_transpose_intermediate: T.Buffer((T.int64(20), T.int64(64), T.int64(77)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_T_reshape_intermediate = T.alloc_buffer((T.int64(1), T.int64(77), T.int64(20), T.int64(64)))
    var_T_transpose_intermediate_1 = T.alloc_buffer((T.int64(1), T.int64(20), T.int64(77), T.int64(64)))
    var_T_reshape_intermediate_1 = T.alloc_buffer((T.int64(20), T.int64(77), T.int64(64)))
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(77), T.int64(20), T.int64(64)):
        with T.block("T_reshape"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(lv28[T.int64(0), ((v_ax2 * T.int64(64) + v_ax3) // T.int64(1280) + v_ax1) % T.int64(77), (v_ax2 * T.int64(64) + v_ax3) % T.int64(1280)])
            T.writes(var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = lv28[T.int64(0), ((v_ax2 * T.int64(64) + v_ax3) // T.int64(1280) + v_ax1) % T.int64(77), (v_ax2 * T.int64(64) + v_ax3) % T.int64(1280)]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(20), T.int64(77), T.int64(64)):
        with T.block("T_transpose"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_reshape_intermediate[v_ax0, v_ax2, v_ax1, v_ax3])
            T.writes(var_T_transpose_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_transpose_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3] = var_T_reshape_intermediate[v_ax0, v_ax2, v_ax1, v_ax3]
    for ax0, ax1, ax2 in T.grid(T.int64(20), T.int64(77), T.int64(64)):
        with T.block("T_reshape_1"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_T_transpose_intermediate_1[T.int64(0), ((v_ax2 // T.int64(64) + v_ax1) // T.int64(77) + v_ax0) % T.int64(20), (v_ax2 // T.int64(64) + v_ax1) % T.int64(77), v_ax2 % T.int64(64)])
            T.writes(var_T_reshape_intermediate_1[v_ax0, v_ax1, v_ax2])
            var_T_reshape_intermediate_1[v_ax0, v_ax1, v_ax2] = var_T_transpose_intermediate_1[T.int64(0), ((v_ax2 // T.int64(64) + v_ax1) // T.int64(77) + v_ax0) % T.int64(20), (v_ax2 // T.int64(64) + v_ax1) % T.int64(77), v_ax2 % T.int64(64)]
    for ax0, ax1, ax2 in T.grid(T.int64(20), T.int64(64), T.int64(77)):
        with T.block("T_transpose_1"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_T_reshape_intermediate_1[v_ax0, v_ax2, v_ax1])
            T.writes(var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2] = var_T_reshape_intermediate_1[v_ax0, v_ax2, v_ax1]

@T.prim_func
def fused_reshape7_add3_reshape8(lv42: T.Buffer((T.int64(20), T.int64(77), T.int64(77)), "float32"), param_0: T.Buffer((T.int64(1), T.int64(1), T.int64(77), T.int64(77)), "float32"), var_T_reshape_intermediate: T.Buffer((T.int64(20), T.int64(77), T.int64(77)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_T_reshape_intermediate_1 = T.alloc_buffer((T.int64(1), T.int64(20), T.int64(77), T.int64(77)))
    var_T_add_intermediate = T.alloc_buffer((T.int64(1), T.int64(20), T.int64(77), T.int64(77)))
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(20), T.int64(77), T.int64(77)):
        with T.block("T_reshape"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(lv42[((v_ax3 // T.int64(77) + v_ax2) // T.int64(77) + v_ax1) % T.int64(20), (v_ax3 // T.int64(77) + v_ax2) % T.int64(77), v_ax3 % T.int64(77)])
            T.writes(var_T_reshape_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_reshape_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3] = lv42[((v_ax3 // T.int64(77) + v_ax2) // T.int64(77) + v_ax1) % T.int64(20), (v_ax3 // T.int64(77) + v_ax2) % T.int64(77), v_ax3 % T.int64(77)]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(20), T.int64(77), T.int64(77)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_reshape_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3], param_0[v_ax0, T.int64(0), v_ax2, v_ax3])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_T_reshape_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3] + param_0[v_ax0, T.int64(0), v_ax2, v_ax3]
    for ax0, ax1, ax2 in T.grid(T.int64(20), T.int64(77), T.int64(77)):
        with T.block("T_reshape_1"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_T_add_intermediate[T.int64(0), ((v_ax2 // T.int64(77) + v_ax1) // T.int64(77) + v_ax0) % T.int64(20), (v_ax2 // T.int64(77) + v_ax1) % T.int64(77), v_ax2 % T.int64(77)])
            T.writes(var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2] = var_T_add_intermediate[T.int64(0), ((v_ax2 // T.int64(77) + v_ax1) // T.int64(77) + v_ax0) % T.int64(20), (v_ax2 // T.int64(77) + v_ax1) % T.int64(77), v_ax2 % T.int64(77)]

@T.prim_func
def fused_reshape7_reshape8(lv46: T.Buffer((T.int64(20), T.int64(77), T.int64(77)), "float32"), var_T_reshape_intermediate: T.Buffer((T.int64(20), T.int64(77), T.int64(77)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_T_reshape_intermediate_1 = T.alloc_buffer((T.int64(1), T.int64(20), T.int64(77), T.int64(77)))
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(20), T.int64(77), T.int64(77)):
        with T.block("T_reshape"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(lv46[((v_ax3 // T.int64(77) + v_ax2) // T.int64(77) + v_ax1) % T.int64(20), (v_ax3 // T.int64(77) + v_ax2) % T.int64(77), v_ax3 % T.int64(77)])
            T.writes(var_T_reshape_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_reshape_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3] = lv46[((v_ax3 // T.int64(77) + v_ax2) // T.int64(77) + v_ax1) % T.int64(20), (v_ax3 // T.int64(77) + v_ax2) % T.int64(77), v_ax3 % T.int64(77)]
    for ax0, ax1, ax2 in T.grid(T.int64(20), T.int64(77), T.int64(77)):
        with T.block("T_reshape_1"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_T_reshape_intermediate_1[T.int64(0), ((v_ax2 // T.int64(77) + v_ax1) // T.int64(77) + v_ax0) % T.int64(20), (v_ax2 // T.int64(77) + v_ax1) % T.int64(77), v_ax2 % T.int64(77)])
            T.writes(var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2] = var_T_reshape_intermediate_1[T.int64(0), ((v_ax2 // T.int64(77) + v_ax1) // T.int64(77) + v_ax0) % T.int64(20), (v_ax2 // T.int64(77) + v_ax1) % T.int64(77), v_ax2 % T.int64(77)]

@T.prim_func
def fused_reshape9_transpose3_reshape10(lv49: T.Buffer((T.int64(20), T.int64(77), T.int64(64)), "float32"), var_T_reshape_intermediate: T.Buffer((T.int64(1), T.int64(77), T.int64(1280)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_T_reshape_intermediate_1 = T.alloc_buffer((T.int64(1), T.int64(20), T.int64(77), T.int64(64)))
    var_T_transpose_intermediate = T.alloc_buffer((T.int64(1), T.int64(77), T.int64(20), T.int64(64)))
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(20), T.int64(77), T.int64(64)):
        with T.block("T_reshape"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(lv49[((v_ax3 // T.int64(64) + v_ax2) // T.int64(77) + v_ax1) % T.int64(20), (v_ax3 // T.int64(64) + v_ax2) % T.int64(77), v_ax3 % T.int64(64)])
            T.writes(var_T_reshape_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_reshape_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3] = lv49[((v_ax3 // T.int64(64) + v_ax2) // T.int64(77) + v_ax1) % T.int64(20), (v_ax3 // T.int64(64) + v_ax2) % T.int64(77), v_ax3 % T.int64(64)]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(77), T.int64(20), T.int64(64)):
        with T.block("T_transpose"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_reshape_intermediate_1[v_ax0, v_ax2, v_ax1, v_ax3])
            T.writes(var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_T_reshape_intermediate_1[v_ax0, v_ax2, v_ax1, v_ax3]
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(77), T.int64(1280)):
        with T.block("T_reshape_1"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_T_transpose_intermediate[T.int64(0), (v_ax2 // T.int64(1280) + v_ax1) % T.int64(77), v_ax2 % T.int64(1280) // T.int64(64), v_ax2 % T.int64(64)])
            T.writes(var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2] = var_T_transpose_intermediate[T.int64(0), (v_ax2 // T.int64(1280) + v_ax1) % T.int64(77), v_ax2 % T.int64(1280) // T.int64(64), v_ax2 % T.int64(64)]

@T.prim_func
def fused_reshape_cast_reshape1(inp_0: T.Buffer((T.int64(1), T.int64(77)), "int32"), var_T_reshape_intermediate: T.Buffer((T.int64(77),), "int32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_T_reshape_intermediate_1 = T.alloc_buffer((T.int64(1), T.int64(77)), "int32")
    var_compute_intermediate = T.alloc_buffer((T.int64(1), T.int64(77)), "int32")
    for ax0, ax1 in T.grid(T.int64(1), T.int64(77)):
        with T.block("T_reshape"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(inp_0[T.int64(0), v_ax1 % T.int64(77)])
            T.writes(var_T_reshape_intermediate_1[v_ax0, v_ax1])
            var_T_reshape_intermediate_1[v_ax0, v_ax1] = inp_0[T.int64(0), v_ax1 % T.int64(77)]
    for i0, i1 in T.grid(T.int64(1), T.int64(77)):
        with T.block("compute"):
            v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
            T.reads(var_T_reshape_intermediate_1[v_i0, v_i1])
            T.writes(var_compute_intermediate[v_i0, v_i1])
            var_compute_intermediate[v_i0, v_i1] = var_T_reshape_intermediate_1[v_i0, v_i1]
    for ax0 in range(T.int64(77)):
        with T.block("T_reshape_1"):
            v_ax0 = T.axis.spatial(T.int64(77), ax0)
            T.reads(var_compute_intermediate[T.int64(0), v_ax0 % T.int64(77)])
            T.writes(var_T_reshape_intermediate[v_ax0])
            var_T_reshape_intermediate[v_ax0] = var_compute_intermediate[T.int64(0), v_ax0 % T.int64(77)]

@T.prim_func
def fused_split1_gelu2_multiply10(lv511: T.Buffer((T.int64(2), T.int64(1024), T.int64(10240)), "float32"), var_T_multiply_intermediate: T.Buffer((T.int64(2), T.int64(1024), T.int64(5120)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_T_split_sections_intermediate = T.alloc_buffer((T.int64(2), T.int64(1024), T.int64(5120)))
    var_T_split_sections_intermediate_1 = T.alloc_buffer((T.int64(2), T.int64(1024), T.int64(5120)))
    T_multiply = T.alloc_buffer((T.int64(2), T.int64(1024), T.int64(5120)))
    compute = T.alloc_buffer((T.int64(2), T.int64(1024), T.int64(5120)))
    T_multiply_1 = T.alloc_buffer((T.int64(2), T.int64(1024), T.int64(5120)))
    T_add = T.alloc_buffer((T.int64(2), T.int64(1024), T.int64(5120)))
    var_T_multiply_intermediate_1 = T.alloc_buffer((T.int64(2), T.int64(1024), T.int64(5120)))
    for ax0, ax1, ax2 in T.grid(T.int64(2), T.int64(1024), T.int64(5120)):
        with T.block("T_split_sections"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(lv511[v_ax0, v_ax1, v_ax2])
            T.writes(var_T_split_sections_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_split_sections_intermediate[v_ax0, v_ax1, v_ax2] = lv511[v_ax0, v_ax1, v_ax2]
    for ax0, ax1, ax2 in T.grid(T.int64(2), T.int64(1024), T.int64(5120)):
        with T.block("T_split_sections_1"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(lv511[v_ax0, v_ax1, v_ax2 + T.int64(5120)])
            T.writes(var_T_split_sections_intermediate_1[v_ax0, v_ax1, v_ax2])
            var_T_split_sections_intermediate_1[v_ax0, v_ax1, v_ax2] = lv511[v_ax0, v_ax1, v_ax2 + T.int64(5120)]
    for ax0, ax1, ax2 in T.grid(T.int64(2), T.int64(1024), T.int64(5120)):
        with T.block("T_multiply"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_T_split_sections_intermediate_1[v_ax0, v_ax1, v_ax2])
            T.writes(T_multiply[v_ax0, v_ax1, v_ax2])
            T_multiply[v_ax0, v_ax1, v_ax2] = var_T_split_sections_intermediate_1[v_ax0, v_ax1, v_ax2] * T.float32(0.70710678118654757)
    for i0, i1, i2 in T.grid(T.int64(2), T.int64(1024), T.int64(5120)):
        with T.block("compute"):
            v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
            T.reads(T_multiply[v_i0, v_i1, v_i2])
            T.writes(compute[v_i0, v_i1, v_i2])
            compute[v_i0, v_i1, v_i2] = T.erf(T_multiply[v_i0, v_i1, v_i2])
    for ax0, ax1, ax2 in T.grid(T.int64(2), T.int64(1024), T.int64(5120)):
        with T.block("T_multiply_1"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(compute[v_ax0, v_ax1, v_ax2])
            T.writes(T_multiply_1[v_ax0, v_ax1, v_ax2])
            T_multiply_1[v_ax0, v_ax1, v_ax2] = compute[v_ax0, v_ax1, v_ax2] * T.float32(0.5)
    for ax0, ax1, ax2 in T.grid(T.int64(2), T.int64(1024), T.int64(5120)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(T_multiply_1[v_ax0, v_ax1, v_ax2])
            T.writes(T_add[v_ax0, v_ax1, v_ax2])
            T_add[v_ax0, v_ax1, v_ax2] = T.float32(0.5) + T_multiply_1[v_ax0, v_ax1, v_ax2]
    for ax0, ax1, ax2 in T.grid(T.int64(2), T.int64(1024), T.int64(5120)):
        with T.block("T_multiply_2"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_T_split_sections_intermediate_1[v_ax0, v_ax1, v_ax2], T_add[v_ax0, v_ax1, v_ax2])
            T.writes(var_T_multiply_intermediate_1[v_ax0, v_ax1, v_ax2])
            var_T_multiply_intermediate_1[v_ax0, v_ax1, v_ax2] = var_T_split_sections_intermediate_1[v_ax0, v_ax1, v_ax2] * T_add[v_ax0, v_ax1, v_ax2]
    for ax0, ax1, ax2 in T.grid(T.int64(2), T.int64(1024), T.int64(5120)):
        with T.block("T_multiply_3"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_T_split_sections_intermediate[v_ax0, v_ax1, v_ax2], var_T_multiply_intermediate_1[v_ax0, v_ax1, v_ax2])
            T.writes(var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2] = var_T_split_sections_intermediate[v_ax0, v_ax1, v_ax2] * var_T_multiply_intermediate_1[v_ax0, v_ax1, v_ax2]

@T.prim_func
def fused_split2_subtract_multiply11_add29(lv5254: T.Buffer((T.int64(2), T.int64(4), T.int64(128), T.int64(128)), "float32"), var_T_add_intermediate: T.Buffer((T.int64(1), T.int64(4), T.int64(128), T.int64(128)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_T_split_sections_intermediate = T.alloc_buffer((T.int64(1), T.int64(4), T.int64(128), T.int64(128)))
    var_T_split_sections_intermediate_1 = T.alloc_buffer((T.int64(1), T.int64(4), T.int64(128), T.int64(128)))
    var_T_subtract_intermediate = T.alloc_buffer((T.int64(1), T.int64(4), T.int64(128), T.int64(128)))
    var_T_multiply_intermediate = T.alloc_buffer((T.int64(1), T.int64(4), T.int64(128), T.int64(128)))
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(4), T.int64(128), T.int64(128)):
        with T.block("T_split_sections"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(lv5254[v_ax0, v_ax1, v_ax2, v_ax3])
            T.writes(var_T_split_sections_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_split_sections_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = lv5254[v_ax0, v_ax1, v_ax2, v_ax3]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(4), T.int64(128), T.int64(128)):
        with T.block("T_split_sections_1"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(lv5254[v_ax0 + T.int64(1), v_ax1, v_ax2, v_ax3])
            T.writes(var_T_split_sections_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_split_sections_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3] = lv5254[v_ax0 + T.int64(1), v_ax1, v_ax2, v_ax3]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(4), T.int64(128), T.int64(128)):
        with T.block("T_subtract"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_split_sections_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3], var_T_split_sections_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            T.writes(var_T_subtract_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_subtract_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_T_split_sections_intermediate_1[v_ax0, v_ax1, v_ax2, v_ax3] - var_T_split_sections_intermediate[v_ax0, v_ax1, v_ax2, v_ax3]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(4), T.int64(128), T.int64(128)):
        with T.block("T_multiply"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_subtract_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            T.writes(var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = T.float32(5) * var_T_subtract_intermediate[v_ax0, v_ax1, v_ax2, v_ax3]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(4), T.int64(128), T.int64(128)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_split_sections_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_T_split_sections_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] + var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2, v_ax3]

@T.prim_func
def fused_split_gelu1_multiply7(lv175: T.Buffer((T.int64(2), T.int64(4096), T.int64(5120)), "float32"), var_T_multiply_intermediate: T.Buffer((T.int64(2), T.int64(4096), T.int64(2560)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_T_split_sections_intermediate = T.alloc_buffer((T.int64(2), T.int64(4096), T.int64(2560)))
    var_T_split_sections_intermediate_1 = T.alloc_buffer((T.int64(2), T.int64(4096), T.int64(2560)))
    T_multiply = T.alloc_buffer((T.int64(2), T.int64(4096), T.int64(2560)))
    compute = T.alloc_buffer((T.int64(2), T.int64(4096), T.int64(2560)))
    T_multiply_1 = T.alloc_buffer((T.int64(2), T.int64(4096), T.int64(2560)))
    T_add = T.alloc_buffer((T.int64(2), T.int64(4096), T.int64(2560)))
    var_T_multiply_intermediate_1 = T.alloc_buffer((T.int64(2), T.int64(4096), T.int64(2560)))
    for ax0, ax1, ax2 in T.grid(T.int64(2), T.int64(4096), T.int64(2560)):
        with T.block("T_split_sections"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(lv175[v_ax0, v_ax1, v_ax2])
            T.writes(var_T_split_sections_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_split_sections_intermediate[v_ax0, v_ax1, v_ax2] = lv175[v_ax0, v_ax1, v_ax2]
    for ax0, ax1, ax2 in T.grid(T.int64(2), T.int64(4096), T.int64(2560)):
        with T.block("T_split_sections_1"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(lv175[v_ax0, v_ax1, v_ax2 + T.int64(2560)])
            T.writes(var_T_split_sections_intermediate_1[v_ax0, v_ax1, v_ax2])
            var_T_split_sections_intermediate_1[v_ax0, v_ax1, v_ax2] = lv175[v_ax0, v_ax1, v_ax2 + T.int64(2560)]
    for ax0, ax1, ax2 in T.grid(T.int64(2), T.int64(4096), T.int64(2560)):
        with T.block("T_multiply"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_T_split_sections_intermediate_1[v_ax0, v_ax1, v_ax2])
            T.writes(T_multiply[v_ax0, v_ax1, v_ax2])
            T_multiply[v_ax0, v_ax1, v_ax2] = var_T_split_sections_intermediate_1[v_ax0, v_ax1, v_ax2] * T.float32(0.70710678118654757)
    for i0, i1, i2 in T.grid(T.int64(2), T.int64(4096), T.int64(2560)):
        with T.block("compute"):
            v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
            T.reads(T_multiply[v_i0, v_i1, v_i2])
            T.writes(compute[v_i0, v_i1, v_i2])
            compute[v_i0, v_i1, v_i2] = T.erf(T_multiply[v_i0, v_i1, v_i2])
    for ax0, ax1, ax2 in T.grid(T.int64(2), T.int64(4096), T.int64(2560)):
        with T.block("T_multiply_1"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(compute[v_ax0, v_ax1, v_ax2])
            T.writes(T_multiply_1[v_ax0, v_ax1, v_ax2])
            T_multiply_1[v_ax0, v_ax1, v_ax2] = compute[v_ax0, v_ax1, v_ax2] * T.float32(0.5)
    for ax0, ax1, ax2 in T.grid(T.int64(2), T.int64(4096), T.int64(2560)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(T_multiply_1[v_ax0, v_ax1, v_ax2])
            T.writes(T_add[v_ax0, v_ax1, v_ax2])
            T_add[v_ax0, v_ax1, v_ax2] = T.float32(0.5) + T_multiply_1[v_ax0, v_ax1, v_ax2]
    for ax0, ax1, ax2 in T.grid(T.int64(2), T.int64(4096), T.int64(2560)):
        with T.block("T_multiply_2"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_T_split_sections_intermediate_1[v_ax0, v_ax1, v_ax2], T_add[v_ax0, v_ax1, v_ax2])
            T.writes(var_T_multiply_intermediate_1[v_ax0, v_ax1, v_ax2])
            var_T_multiply_intermediate_1[v_ax0, v_ax1, v_ax2] = var_T_split_sections_intermediate_1[v_ax0, v_ax1, v_ax2] * T_add[v_ax0, v_ax1, v_ax2]
    for ax0, ax1, ax2 in T.grid(T.int64(2), T.int64(4096), T.int64(2560)):
        with T.block("T_multiply_3"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_T_split_sections_intermediate[v_ax0, v_ax1, v_ax2], var_T_multiply_intermediate_1[v_ax0, v_ax1, v_ax2])
            T.writes(var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2] = var_T_split_sections_intermediate[v_ax0, v_ax1, v_ax2] * var_T_multiply_intermediate_1[v_ax0, v_ax1, v_ax2]

@T.prim_func
def fused_strided_slice_reshape11(lv1465: T.Buffer((T.int64(1), T.int64(1280)), "float32"), var_T_reshape_intermediate: T.Buffer((T.int64(1), T.int64(1280)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_T_strided_slice_with_axes_intermediate = T.alloc_buffer((T.int64(1), T.int64(1280)))
    for ax0, ax1 in T.grid(T.int64(1), T.int64(1280)):
        with T.block("T_strided_slice_with_axes"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(lv1465[v_ax0, v_ax1])
            T.writes(var_T_strided_slice_with_axes_intermediate[v_ax0, v_ax1])
            var_T_strided_slice_with_axes_intermediate[v_ax0, v_ax1] = lv1465[v_ax0, v_ax1]
    for ax0, ax1 in T.grid(T.int64(1), T.int64(1280)):
        with T.block("T_reshape"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(var_T_strided_slice_with_axes_intermediate[T.int64(0), v_ax1 % T.int64(1280)])
            T.writes(var_T_reshape_intermediate[v_ax0, v_ax1])
            var_T_reshape_intermediate[v_ax0, v_ax1] = var_T_strided_slice_with_axes_intermediate[T.int64(0), v_ax1 % T.int64(1280)]

@T.prim_func
def fused_transpose10_reshape22(lv112: T.Buffer((T.int64(2), T.int64(640), T.int64(64), T.int64(64)), "float32"), var_T_reshape_intermediate: T.Buffer((T.int64(2), T.int64(4096), T.int64(640)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_T_transpose_intermediate = T.alloc_buffer((T.int64(2), T.int64(64), T.int64(64), T.int64(640)))
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(64), T.int64(64), T.int64(640)):
        with T.block("T_transpose"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(lv112[v_ax0, v_ax3, v_ax1, v_ax2])
            T.writes(var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = lv112[v_ax0, v_ax3, v_ax1, v_ax2]
    for ax0, ax1, ax2 in T.grid(T.int64(2), T.int64(4096), T.int64(640)):
        with T.block("T_reshape"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_T_transpose_intermediate[((v_ax2 // T.int64(640) + v_ax1) // T.int64(4096) + v_ax0) % T.int64(2), (v_ax2 // T.int64(640) + v_ax1) % T.int64(4096) // T.int64(64), (v_ax2 // T.int64(640) + v_ax1) % T.int64(64), v_ax2 % T.int64(640)])
            T.writes(var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2] = var_T_transpose_intermediate[((v_ax2 // T.int64(640) + v_ax1) // T.int64(4096) + v_ax0) % T.int64(2), (v_ax2 // T.int64(640) + v_ax1) % T.int64(4096) // T.int64(64), (v_ax2 // T.int64(640) + v_ax1) % T.int64(64), v_ax2 % T.int64(640)]

@T.prim_func
def fused_transpose14_reshape24(lv137: T.Buffer((T.int64(2), T.int64(10), T.int64(4096), T.int64(64)), "float32"), var_T_reshape_intermediate: T.Buffer((T.int64(2), T.int64(4096), T.int64(640)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_T_transpose_intermediate = T.alloc_buffer((T.int64(2), T.int64(4096), T.int64(10), T.int64(64)))
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(4096), T.int64(10), T.int64(64)):
        with T.block("T_transpose"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(lv137[v_ax0, v_ax2, v_ax1, v_ax3])
            T.writes(var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = lv137[v_ax0, v_ax2, v_ax1, v_ax3]
    for ax0, ax1, ax2 in T.grid(T.int64(2), T.int64(4096), T.int64(640)):
        with T.block("T_reshape"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_T_transpose_intermediate[((v_ax2 // T.int64(640) + v_ax1) // T.int64(4096) + v_ax0) % T.int64(2), (v_ax2 // T.int64(640) + v_ax1) % T.int64(4096), v_ax2 % T.int64(640) // T.int64(64), v_ax2 % T.int64(64)])
            T.writes(var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2] = var_T_transpose_intermediate[((v_ax2 // T.int64(640) + v_ax1) // T.int64(4096) + v_ax0) % T.int64(2), (v_ax2 // T.int64(640) + v_ax1) % T.int64(4096), v_ax2 % T.int64(640) // T.int64(64), v_ax2 % T.int64(64)]

@T.prim_func
def fused_transpose21_reshape29(lv448: T.Buffer((T.int64(2), T.int64(1280), T.int64(32), T.int64(32)), "float32"), var_T_reshape_intermediate: T.Buffer((T.int64(2), T.int64(1024), T.int64(1280)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_T_transpose_intermediate = T.alloc_buffer((T.int64(2), T.int64(32), T.int64(32), T.int64(1280)))
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(32), T.int64(32), T.int64(1280)):
        with T.block("T_transpose"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(lv448[v_ax0, v_ax3, v_ax1, v_ax2])
            T.writes(var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = lv448[v_ax0, v_ax3, v_ax1, v_ax2]
    for ax0, ax1, ax2 in T.grid(T.int64(2), T.int64(1024), T.int64(1280)):
        with T.block("T_reshape"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_T_transpose_intermediate[((v_ax2 // T.int64(1280) + v_ax1) // T.int64(1024) + v_ax0) % T.int64(2), (v_ax2 // T.int64(1280) + v_ax1) % T.int64(1024) // T.int64(32), (v_ax2 // T.int64(1280) + v_ax1) % T.int64(32), v_ax2 % T.int64(1280)])
            T.writes(var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2] = var_T_transpose_intermediate[((v_ax2 // T.int64(1280) + v_ax1) // T.int64(1024) + v_ax0) % T.int64(2), (v_ax2 // T.int64(1280) + v_ax1) % T.int64(1024) // T.int64(32), (v_ax2 // T.int64(1280) + v_ax1) % T.int64(32), v_ax2 % T.int64(1280)]

@T.prim_func
def fused_transpose24_reshape31(lv473: T.Buffer((T.int64(2), T.int64(20), T.int64(1024), T.int64(64)), "float32"), var_T_reshape_intermediate: T.Buffer((T.int64(2), T.int64(1024), T.int64(1280)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_T_transpose_intermediate = T.alloc_buffer((T.int64(2), T.int64(1024), T.int64(20), T.int64(64)))
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(1024), T.int64(20), T.int64(64)):
        with T.block("T_transpose"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(lv473[v_ax0, v_ax2, v_ax1, v_ax3])
            T.writes(var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = lv473[v_ax0, v_ax2, v_ax1, v_ax3]
    for ax0, ax1, ax2 in T.grid(T.int64(2), T.int64(1024), T.int64(1280)):
        with T.block("T_reshape"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_T_transpose_intermediate[((v_ax2 // T.int64(1280) + v_ax1) // T.int64(1024) + v_ax0) % T.int64(2), (v_ax2 // T.int64(1280) + v_ax1) % T.int64(1024), v_ax2 % T.int64(1280) // T.int64(64), v_ax2 % T.int64(64)])
            T.writes(var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2] = var_T_transpose_intermediate[((v_ax2 // T.int64(1280) + v_ax1) // T.int64(1024) + v_ax0) % T.int64(2), (v_ax2 // T.int64(1280) + v_ax1) % T.int64(1024), v_ax2 % T.int64(1280) // T.int64(64), v_ax2 % T.int64(64)]

@T.prim_func
def fused_transpose31_reshape39_add32_divide6(lv50: T.Buffer((T.int64(1), T.int64(16384), T.int64(512)), "float32"), lv18: T.Buffer((T.int64(1), T.int64(512), T.int64(128), T.int64(128)), "float32"), var_T_divide_intermediate: T.Buffer((T.int64(1), T.int64(512), T.int64(128), T.int64(128)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_T_transpose_intermediate = T.alloc_buffer((T.int64(1), T.int64(512), T.int64(16384)))
    var_T_reshape_intermediate = T.alloc_buffer((T.int64(1), T.int64(512), T.int64(128), T.int64(128)))
    var_T_add_intermediate = T.alloc_buffer((T.int64(1), T.int64(512), T.int64(128), T.int64(128)))
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(512), T.int64(16384)):
        with T.block("T_transpose"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(lv50[v_ax0, v_ax2, v_ax1])
            T.writes(var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2] = lv50[v_ax0, v_ax2, v_ax1]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(512), T.int64(128), T.int64(128)):
        with T.block("T_reshape"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_transpose_intermediate[T.int64(0), ((v_ax2 * T.int64(128) + v_ax3) // T.int64(16384) + v_ax1) % T.int64(512), (v_ax2 * T.int64(128) + v_ax3) % T.int64(16384)])
            T.writes(var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_T_transpose_intermediate[T.int64(0), ((v_ax2 * T.int64(128) + v_ax3) // T.int64(16384) + v_ax1) % T.int64(512), (v_ax2 * T.int64(128) + v_ax3) % T.int64(16384)]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(512), T.int64(128), T.int64(128)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3], lv18[v_ax0, v_ax1, v_ax2, v_ax3])
            T.writes(var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] + lv18[v_ax0, v_ax1, v_ax2, v_ax3]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(512), T.int64(128), T.int64(128)):
        with T.block("T_divide"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            T.writes(var_T_divide_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_divide_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_T_add_intermediate[v_ax0, v_ax1, v_ax2, v_ax3]

@T.prim_func
def fused_transpose35_reshape38(lv45: T.Buffer((T.int64(1), T.int64(1), T.int64(16384), T.int64(512)), "float32"), var_T_reshape_intermediate: T.Buffer((T.int64(1), T.int64(16384), T.int64(512)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_T_transpose_intermediate = T.alloc_buffer((T.int64(1), T.int64(16384), T.int64(1), T.int64(512)))
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(16384), T.int64(1), T.int64(512)):
        with T.block("T_transpose"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(lv45[v_ax0, v_ax2, v_ax1, v_ax3])
            T.writes(var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = lv45[v_ax0, v_ax2, v_ax1, v_ax3]
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(16384), T.int64(512)):
        with T.block("T_reshape"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_T_transpose_intermediate[T.int64(0), (v_ax2 // T.int64(512) + v_ax1) % T.int64(16384), T.int64(0), v_ax2 % T.int64(512)])
            T.writes(var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_reshape_intermediate[v_ax0, v_ax1, v_ax2] = var_T_transpose_intermediate[T.int64(0), (v_ax2 // T.int64(512) + v_ax1) % T.int64(16384), T.int64(0), v_ax2 % T.int64(512)]

@T.prim_func
def fused_transpose36_multiply14_tir_round(lv237: T.Buffer((T.int64(1), T.int64(3), T.int64(1024), T.int64(1024)), "float32"), var_compute_intermediate: T.Buffer((T.int64(1), T.int64(1024), T.int64(1024), T.int64(3)), "float32")):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    var_T_transpose_intermediate = T.alloc_buffer((T.int64(1), T.int64(1024), T.int64(1024), T.int64(3)))
    var_T_multiply_intermediate = T.alloc_buffer((T.int64(1), T.int64(1024), T.int64(1024), T.int64(3)))
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(1024), T.int64(1024), T.int64(3)):
        with T.block("T_transpose"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(lv237[v_ax0, v_ax3, v_ax1, v_ax2])
            T.writes(var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = lv237[v_ax0, v_ax3, v_ax1, v_ax2]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(1024), T.int64(1024), T.int64(3)):
        with T.block("T_multiply"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            T.writes(var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2, v_ax3])
            var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] = var_T_transpose_intermediate[v_ax0, v_ax1, v_ax2, v_ax3] * T.float32(255)
    for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(1024), T.int64(1024), T.int64(3)):
        with T.block("compute"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(var_T_multiply_intermediate[v_i0, v_i1, v_i2, v_i3])
            T.writes(var_compute_intermediate[v_i0, v_i1, v_i2, v_i3])
            var_compute_intermediate[v_i0, v_i1, v_i2, v_i3] = T.round(var_T_multiply_intermediate[v_i0, v_i1, v_i2, v_i3])

@T.prim_func
def group_norm15(A: T.Buffer((T.int64(1), T.int64(512), T.int64(16384)), "float32"), B: T.Buffer((T.int64(512),), "float32"), C: T.Buffer((T.int64(512),), "float32"), T_reshape: T.Buffer((T.int64(1), T.int64(512), T.int64(16384)), "float32")):
    T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    T_reshape_1 = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(16), T.int64(16384)))
    A_red_temp_v0 = T.alloc_buffer((T.int64(1), T.int64(32)))
    A_red_temp_v1 = T.alloc_buffer((T.int64(1), T.int64(32)))
    T_reshape_2 = T.alloc_buffer((T.int64(32), T.int64(16)))
    T_reshape_3 = T.alloc_buffer((T.int64(32), T.int64(16)))
    T_group_norm = T.alloc_buffer((T.int64(1), T.int64(32), T.int64(16), T.int64(16384)))
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(32), T.int64(16), T.int64(16384)):
        with T.block("T_reshape"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(A[T.int64(0), (v_ax1 * T.int64(16) + v_ax3 // T.int64(16384) + v_ax2) % T.int64(512), v_ax3 % T.int64(16384)])
            T.writes(T_reshape_1[v_ax0, v_ax1, v_ax2, v_ax3])
            T_reshape_1[v_ax0, v_ax1, v_ax2, v_ax3] = A[T.int64(0), (v_ax1 * T.int64(16) + v_ax3 // T.int64(16384) + v_ax2) % T.int64(512), v_ax3 % T.int64(16384)]
    for ax0, ax1, k2, k3 in T.grid(T.int64(1), T.int64(32), T.int64(16), T.int64(16384)):
        with T.block("A_red_temp"):
            v_ax0, v_ax1, v_k2, v_k3 = T.axis.remap("SSRR", [ax0, ax1, k2, k3])
            T.reads(T_reshape_1[v_ax0, v_ax1, v_k2, v_k3])
            T.writes(A_red_temp_v0[v_ax0, v_ax1], A_red_temp_v1[v_ax0, v_ax1])
            with T.init():
                A_red_temp_v0[v_ax0, v_ax1] = T.float32(0)
                A_red_temp_v1[v_ax0, v_ax1] = T.float32(0)
            v_A_red_temp_v0: T.float32 = A_red_temp_v0[v_ax0, v_ax1] + T_reshape_1[v_ax0, v_ax1, v_k2, v_k3]
            v_A_red_temp_v1: T.float32 = A_red_temp_v1[v_ax0, v_ax1] + T_reshape_1[v_ax0, v_ax1, v_k2, v_k3] * T_reshape_1[v_ax0, v_ax1, v_k2, v_k3]
            A_red_temp_v0[v_ax0, v_ax1] = v_A_red_temp_v0
            A_red_temp_v1[v_ax0, v_ax1] = v_A_red_temp_v1
    for ax0, ax1 in T.grid(T.int64(32), T.int64(16)):
        with T.block("T_reshape_1"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(B[(v_ax0 * T.int64(16) + v_ax1) % T.int64(512)])
            T.writes(T_reshape_2[v_ax0, v_ax1])
            T_reshape_2[v_ax0, v_ax1] = B[(v_ax0 * T.int64(16) + v_ax1) % T.int64(512)]
    for ax0, ax1 in T.grid(T.int64(32), T.int64(16)):
        with T.block("T_reshape_2"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(C[(v_ax0 * T.int64(16) + v_ax1) % T.int64(512)])
            T.writes(T_reshape_3[v_ax0, v_ax1])
            T_reshape_3[v_ax0, v_ax1] = C[(v_ax0 * T.int64(16) + v_ax1) % T.int64(512)]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(32), T.int64(16), T.int64(16384)):
        with T.block("T_group_norm"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(T_reshape_1[v_ax0, v_ax1, v_ax2, v_ax3], A_red_temp_v0[v_ax0, v_ax1], A_red_temp_v1[v_ax0, v_ax1], T_reshape_2[v_ax1, v_ax2], T_reshape_3[v_ax1, v_ax2])
            T.writes(T_group_norm[v_ax0, v_ax1, v_ax2, v_ax3])
            T_group_norm[v_ax0, v_ax1, v_ax2, v_ax3] = (T_reshape_1[v_ax0, v_ax1, v_ax2, v_ax3] - A_red_temp_v0[v_ax0, v_ax1] * T.float32(3.814697265625e-06)) * T.rsqrt(A_red_temp_v1[v_ax0, v_ax1] * T.float32(3.814697265625e-06) - A_red_temp_v0[v_ax0, v_ax1] * T.float32(3.814697265625e-06) * (A_red_temp_v0[v_ax0, v_ax1] * T.float32(3.814697265625e-06)) + T.float32(9.9999999999999995e-07)) * T_reshape_2[v_ax1, v_ax2] + T_reshape_3[v_ax1, v_ax2]
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(512), T.int64(16384)):
        with T.block("T_reshape_3"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(T_group_norm[T.int64(0), (v_ax2 // T.int64(16384) + v_ax1) % T.int64(512) // T.int64(16), (v_ax2 // T.int64(16384) + v_ax1) % T.int64(16), v_ax2 % T.int64(16384)])
            T.writes(T_reshape[v_ax0, v_ax1, v_ax2])
            T_reshape[v_ax0, v_ax1, v_ax2] = T_group_norm[T.int64(0), (v_ax2 // T.int64(16384) + v_ax1) % T.int64(512) // T.int64(16), (v_ax2 // T.int64(16384) + v_ax1) % T.int64(16), v_ax2 % T.int64(16384)]

@T.prim_func
def group_norm3(A: T.Buffer((T.int64(2), T.int64(640), T.int64(64), T.int64(64)), "float32"), B: T.Buffer((T.int64(640),), "float32"), C: T.Buffer((T.int64(640),), "float32"), T_reshape: T.Buffer((T.int64(2), T.int64(640), T.int64(64), T.int64(64)), "float32")):
    T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    T_reshape_1 = T.alloc_buffer((T.int64(2), T.int64(32), T.int64(20), T.int64(64), T.int64(64)))
    A_red_temp_v0 = T.alloc_buffer((T.int64(2), T.int64(32)))
    A_red_temp_v1 = T.alloc_buffer((T.int64(2), T.int64(32)))
    T_reshape_2 = T.alloc_buffer((T.int64(32), T.int64(20)))
    T_reshape_3 = T.alloc_buffer((T.int64(32), T.int64(20)))
    T_group_norm = T.alloc_buffer((T.int64(2), T.int64(32), T.int64(20), T.int64(64), T.int64(64)))
    for ax0, ax1, ax2, ax3, ax4 in T.grid(T.int64(2), T.int64(32), T.int64(20), T.int64(64), T.int64(64)):
        with T.block("T_reshape"):
            v_ax0, v_ax1, v_ax2, v_ax3, v_ax4 = T.axis.remap("SSSSS", [ax0, ax1, ax2, ax3, ax4])
            T.reads(A[((v_ax1 * T.int64(20) + (v_ax4 // T.int64(64) + v_ax3) // T.int64(64) + v_ax2) // T.int64(640) + v_ax0) % T.int64(2), (v_ax1 * T.int64(20) + (v_ax4 // T.int64(64) + v_ax3) // T.int64(64) + v_ax2) % T.int64(640), (v_ax4 // T.int64(64) + v_ax3) % T.int64(64), v_ax4 % T.int64(64)])
            T.writes(T_reshape_1[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4])
            T_reshape_1[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] = A[((v_ax1 * T.int64(20) + (v_ax4 // T.int64(64) + v_ax3) // T.int64(64) + v_ax2) // T.int64(640) + v_ax0) % T.int64(2), (v_ax1 * T.int64(20) + (v_ax4 // T.int64(64) + v_ax3) // T.int64(64) + v_ax2) % T.int64(640), (v_ax4 // T.int64(64) + v_ax3) % T.int64(64), v_ax4 % T.int64(64)]
    for ax0, ax1, k2, k3, k4 in T.grid(T.int64(2), T.int64(32), T.int64(20), T.int64(64), T.int64(64)):
        with T.block("A_red_temp"):
            v_ax0, v_ax1, v_k2, v_k3, v_k4 = T.axis.remap("SSRRR", [ax0, ax1, k2, k3, k4])
            T.reads(T_reshape_1[v_ax0, v_ax1, v_k2, v_k3, v_k4])
            T.writes(A_red_temp_v0[v_ax0, v_ax1], A_red_temp_v1[v_ax0, v_ax1])
            with T.init():
                A_red_temp_v0[v_ax0, v_ax1] = T.float32(0)
                A_red_temp_v1[v_ax0, v_ax1] = T.float32(0)
            v_A_red_temp_v0: T.float32 = A_red_temp_v0[v_ax0, v_ax1] + T_reshape_1[v_ax0, v_ax1, v_k2, v_k3, v_k4]
            v_A_red_temp_v1: T.float32 = A_red_temp_v1[v_ax0, v_ax1] + T_reshape_1[v_ax0, v_ax1, v_k2, v_k3, v_k4] * T_reshape_1[v_ax0, v_ax1, v_k2, v_k3, v_k4]
            A_red_temp_v0[v_ax0, v_ax1] = v_A_red_temp_v0
            A_red_temp_v1[v_ax0, v_ax1] = v_A_red_temp_v1
    for ax0, ax1 in T.grid(T.int64(32), T.int64(20)):
        with T.block("T_reshape_1"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(B[(v_ax0 * T.int64(20) + v_ax1) % T.int64(640)])
            T.writes(T_reshape_2[v_ax0, v_ax1])
            T_reshape_2[v_ax0, v_ax1] = B[(v_ax0 * T.int64(20) + v_ax1) % T.int64(640)]
    for ax0, ax1 in T.grid(T.int64(32), T.int64(20)):
        with T.block("T_reshape_2"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(C[(v_ax0 * T.int64(20) + v_ax1) % T.int64(640)])
            T.writes(T_reshape_3[v_ax0, v_ax1])
            T_reshape_3[v_ax0, v_ax1] = C[(v_ax0 * T.int64(20) + v_ax1) % T.int64(640)]
    for ax0, ax1, ax2, ax3, ax4 in T.grid(T.int64(2), T.int64(32), T.int64(20), T.int64(64), T.int64(64)):
        with T.block("T_group_norm"):
            v_ax0, v_ax1, v_ax2, v_ax3, v_ax4 = T.axis.remap("SSSSS", [ax0, ax1, ax2, ax3, ax4])
            T.reads(T_reshape_1[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4], A_red_temp_v0[v_ax0, v_ax1], A_red_temp_v1[v_ax0, v_ax1], T_reshape_2[v_ax1, v_ax2], T_reshape_3[v_ax1, v_ax2])
            T.writes(T_group_norm[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4])
            T_group_norm[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] = (T_reshape_1[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] - A_red_temp_v0[v_ax0, v_ax1] * T.float32(1.2207031250000001e-05)) * T.rsqrt(A_red_temp_v1[v_ax0, v_ax1] * T.float32(1.2207031250000001e-05) - A_red_temp_v0[v_ax0, v_ax1] * T.float32(1.2207031250000001e-05) * (A_red_temp_v0[v_ax0, v_ax1] * T.float32(1.2207031250000001e-05)) + T.float32(9.9999999999999995e-07)) * T_reshape_2[v_ax1, v_ax2] + T_reshape_3[v_ax1, v_ax2]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(640), T.int64(64), T.int64(64)):
        with T.block("T_reshape_3"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(T_group_norm[(((v_ax3 // T.int64(64) + v_ax2) // T.int64(64) + v_ax1) // T.int64(640) + v_ax0) % T.int64(2), ((v_ax3 // T.int64(64) + v_ax2) // T.int64(64) + v_ax1) % T.int64(640) // T.int64(20), ((v_ax3 // T.int64(64) + v_ax2) // T.int64(64) + v_ax1) % T.int64(20), (v_ax3 // T.int64(64) + v_ax2) % T.int64(64), v_ax3 % T.int64(64)])
            T.writes(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3])
            T_reshape[v_ax0, v_ax1, v_ax2, v_ax3] = T_group_norm[(((v_ax3 // T.int64(64) + v_ax2) // T.int64(64) + v_ax1) // T.int64(640) + v_ax0) % T.int64(2), ((v_ax3 // T.int64(64) + v_ax2) // T.int64(64) + v_ax1) % T.int64(640) // T.int64(20), ((v_ax3 // T.int64(64) + v_ax2) // T.int64(64) + v_ax1) % T.int64(20), (v_ax3 // T.int64(64) + v_ax2) % T.int64(64), v_ax3 % T.int64(64)]

@T.prim_func
def group_norm6(A: T.Buffer((T.int64(2), T.int64(1280), T.int64(32), T.int64(32)), "float32"), B: T.Buffer((T.int64(1280),), "float32"), C: T.Buffer((T.int64(1280),), "float32"), T_reshape: T.Buffer((T.int64(2), T.int64(1280), T.int64(32), T.int64(32)), "float32")):
    T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    T_reshape_1 = T.alloc_buffer((T.int64(2), T.int64(32), T.int64(40), T.int64(32), T.int64(32)))
    A_red_temp_v0 = T.alloc_buffer((T.int64(2), T.int64(32)))
    A_red_temp_v1 = T.alloc_buffer((T.int64(2), T.int64(32)))
    T_reshape_2 = T.alloc_buffer((T.int64(32), T.int64(40)))
    T_reshape_3 = T.alloc_buffer((T.int64(32), T.int64(40)))
    T_group_norm = T.alloc_buffer((T.int64(2), T.int64(32), T.int64(40), T.int64(32), T.int64(32)))
    for ax0, ax1, ax2, ax3, ax4 in T.grid(T.int64(2), T.int64(32), T.int64(40), T.int64(32), T.int64(32)):
        with T.block("T_reshape"):
            v_ax0, v_ax1, v_ax2, v_ax3, v_ax4 = T.axis.remap("SSSSS", [ax0, ax1, ax2, ax3, ax4])
            T.reads(A[((v_ax1 * T.int64(40) + (v_ax4 // T.int64(32) + v_ax3) // T.int64(32) + v_ax2) // T.int64(1280) + v_ax0) % T.int64(2), (v_ax1 * T.int64(40) + (v_ax4 // T.int64(32) + v_ax3) // T.int64(32) + v_ax2) % T.int64(1280), (v_ax4 // T.int64(32) + v_ax3) % T.int64(32), v_ax4 % T.int64(32)])
            T.writes(T_reshape_1[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4])
            T_reshape_1[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] = A[((v_ax1 * T.int64(40) + (v_ax4 // T.int64(32) + v_ax3) // T.int64(32) + v_ax2) // T.int64(1280) + v_ax0) % T.int64(2), (v_ax1 * T.int64(40) + (v_ax4 // T.int64(32) + v_ax3) // T.int64(32) + v_ax2) % T.int64(1280), (v_ax4 // T.int64(32) + v_ax3) % T.int64(32), v_ax4 % T.int64(32)]
    for ax0, ax1, k2, k3, k4 in T.grid(T.int64(2), T.int64(32), T.int64(40), T.int64(32), T.int64(32)):
        with T.block("A_red_temp"):
            v_ax0, v_ax1, v_k2, v_k3, v_k4 = T.axis.remap("SSRRR", [ax0, ax1, k2, k3, k4])
            T.reads(T_reshape_1[v_ax0, v_ax1, v_k2, v_k3, v_k4])
            T.writes(A_red_temp_v0[v_ax0, v_ax1], A_red_temp_v1[v_ax0, v_ax1])
            with T.init():
                A_red_temp_v0[v_ax0, v_ax1] = T.float32(0)
                A_red_temp_v1[v_ax0, v_ax1] = T.float32(0)
            v_A_red_temp_v0: T.float32 = A_red_temp_v0[v_ax0, v_ax1] + T_reshape_1[v_ax0, v_ax1, v_k2, v_k3, v_k4]
            v_A_red_temp_v1: T.float32 = A_red_temp_v1[v_ax0, v_ax1] + T_reshape_1[v_ax0, v_ax1, v_k2, v_k3, v_k4] * T_reshape_1[v_ax0, v_ax1, v_k2, v_k3, v_k4]
            A_red_temp_v0[v_ax0, v_ax1] = v_A_red_temp_v0
            A_red_temp_v1[v_ax0, v_ax1] = v_A_red_temp_v1
    for ax0, ax1 in T.grid(T.int64(32), T.int64(40)):
        with T.block("T_reshape_1"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(B[(v_ax0 * T.int64(40) + v_ax1) % T.int64(1280)])
            T.writes(T_reshape_2[v_ax0, v_ax1])
            T_reshape_2[v_ax0, v_ax1] = B[(v_ax0 * T.int64(40) + v_ax1) % T.int64(1280)]
    for ax0, ax1 in T.grid(T.int64(32), T.int64(40)):
        with T.block("T_reshape_2"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(C[(v_ax0 * T.int64(40) + v_ax1) % T.int64(1280)])
            T.writes(T_reshape_3[v_ax0, v_ax1])
            T_reshape_3[v_ax0, v_ax1] = C[(v_ax0 * T.int64(40) + v_ax1) % T.int64(1280)]
    for ax0, ax1, ax2, ax3, ax4 in T.grid(T.int64(2), T.int64(32), T.int64(40), T.int64(32), T.int64(32)):
        with T.block("T_group_norm"):
            v_ax0, v_ax1, v_ax2, v_ax3, v_ax4 = T.axis.remap("SSSSS", [ax0, ax1, ax2, ax3, ax4])
            T.reads(T_reshape_1[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4], A_red_temp_v0[v_ax0, v_ax1], A_red_temp_v1[v_ax0, v_ax1], T_reshape_2[v_ax1, v_ax2], T_reshape_3[v_ax1, v_ax2])
            T.writes(T_group_norm[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4])
            T_group_norm[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] = (T_reshape_1[v_ax0, v_ax1, v_ax2, v_ax3, v_ax4] - A_red_temp_v0[v_ax0, v_ax1] * T.float32(2.4414062500000001e-05)) * T.rsqrt(A_red_temp_v1[v_ax0, v_ax1] * T.float32(2.4414062500000001e-05) - A_red_temp_v0[v_ax0, v_ax1] * T.float32(2.4414062500000001e-05) * (A_red_temp_v0[v_ax0, v_ax1] * T.float32(2.4414062500000001e-05)) + T.float32(9.9999999999999995e-07)) * T_reshape_2[v_ax1, v_ax2] + T_reshape_3[v_ax1, v_ax2]
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(1280), T.int64(32), T.int64(32)):
        with T.block("T_reshape_3"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(T_group_norm[(((v_ax3 // T.int64(32) + v_ax2) // T.int64(32) + v_ax1) // T.int64(1280) + v_ax0) % T.int64(2), ((v_ax3 // T.int64(32) + v_ax2) // T.int64(32) + v_ax1) % T.int64(1280) // T.int64(40), ((v_ax3 // T.int64(32) + v_ax2) // T.int64(32) + v_ax1) % T.int64(40), (v_ax3 // T.int64(32) + v_ax2) % T.int64(32), v_ax3 % T.int64(32)])
            T.writes(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3])
            T_reshape[v_ax0, v_ax1, v_ax2, v_ax3] = T_group_norm[(((v_ax3 // T.int64(32) + v_ax2) // T.int64(32) + v_ax1) // T.int64(1280) + v_ax0) % T.int64(2), ((v_ax3 // T.int64(32) + v_ax2) // T.int64(32) + v_ax1) % T.int64(1280) // T.int64(40), ((v_ax3 // T.int64(32) + v_ax2) // T.int64(32) + v_ax1) % T.int64(40), (v_ax3 // T.int64(32) + v_ax2) % T.int64(32), v_ax3 % T.int64(32)]

@T.prim_func
def layer_norm(A: T.Buffer((T.int64(1), T.int64(77), T.int64(1280)), "float32"), B: T.Buffer((T.int64(1280),), "float32"), C: T.Buffer((T.int64(1280),), "float32"), T_layer_norm: T.Buffer((T.int64(1), T.int64(77), T.int64(1280)), "float32")):
    T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    A_red_temp_v0 = T.alloc_buffer((T.int64(1), T.int64(77)))
    A_red_temp_v1 = T.alloc_buffer((T.int64(1), T.int64(77)))
    for ax0, ax1, k2 in T.grid(T.int64(1), T.int64(77), T.int64(1280)):
        with T.block("A_red_temp"):
            v_ax0, v_ax1, v_k2 = T.axis.remap("SSR", [ax0, ax1, k2])
            T.reads(A[v_ax0, v_ax1, v_k2])
            T.writes(A_red_temp_v0[v_ax0, v_ax1], A_red_temp_v1[v_ax0, v_ax1])
            with T.init():
                A_red_temp_v0[v_ax0, v_ax1] = T.float32(0)
                A_red_temp_v1[v_ax0, v_ax1] = T.float32(0)
            v_A_red_temp_v0: T.float32 = A_red_temp_v0[v_ax0, v_ax1] + A[v_ax0, v_ax1, v_k2]
            v_A_red_temp_v1: T.float32 = A_red_temp_v1[v_ax0, v_ax1] + A[v_ax0, v_ax1, v_k2] * A[v_ax0, v_ax1, v_k2]
            A_red_temp_v0[v_ax0, v_ax1] = v_A_red_temp_v0
            A_red_temp_v1[v_ax0, v_ax1] = v_A_red_temp_v1
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(77), T.int64(1280)):
        with T.block("T_layer_norm"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(A[v_ax0, v_ax1, v_ax2], A_red_temp_v0[v_ax0, v_ax1], A_red_temp_v1[v_ax0, v_ax1], B[v_ax2], C[v_ax2])
            T.writes(T_layer_norm[v_ax0, v_ax1, v_ax2])
            T_layer_norm[v_ax0, v_ax1, v_ax2] = (A[v_ax0, v_ax1, v_ax2] - A_red_temp_v0[v_ax0, v_ax1] * T.float32(0.00078125000000000004)) * T.rsqrt(A_red_temp_v1[v_ax0, v_ax1] * T.float32(0.00078125000000000004) - A_red_temp_v0[v_ax0, v_ax1] * T.float32(0.00078125000000000004) * (A_red_temp_v0[v_ax0, v_ax1] * T.float32(0.00078125000000000004)) + T.float32(1.0000000000000001e-05)) * B[v_ax2] + C[v_ax2]

@T.prim_func
def layer_norm1(A: T.Buffer((T.int64(2), T.int64(4096), T.int64(640)), "float32"), B: T.Buffer((T.int64(640),), "float32"), C: T.Buffer((T.int64(640),), "float32"), T_layer_norm: T.Buffer((T.int64(2), T.int64(4096), T.int64(640)), "float32")):
    T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    A_red_temp_v0 = T.alloc_buffer((T.int64(2), T.int64(4096)))
    A_red_temp_v1 = T.alloc_buffer((T.int64(2), T.int64(4096)))
    for ax0, ax1, k2 in T.grid(T.int64(2), T.int64(4096), T.int64(640)):
        with T.block("A_red_temp"):
            v_ax0, v_ax1, v_k2 = T.axis.remap("SSR", [ax0, ax1, k2])
            T.reads(A[v_ax0, v_ax1, v_k2])
            T.writes(A_red_temp_v0[v_ax0, v_ax1], A_red_temp_v1[v_ax0, v_ax1])
            with T.init():
                A_red_temp_v0[v_ax0, v_ax1] = T.float32(0)
                A_red_temp_v1[v_ax0, v_ax1] = T.float32(0)
            v_A_red_temp_v0: T.float32 = A_red_temp_v0[v_ax0, v_ax1] + A[v_ax0, v_ax1, v_k2]
            v_A_red_temp_v1: T.float32 = A_red_temp_v1[v_ax0, v_ax1] + A[v_ax0, v_ax1, v_k2] * A[v_ax0, v_ax1, v_k2]
            A_red_temp_v0[v_ax0, v_ax1] = v_A_red_temp_v0
            A_red_temp_v1[v_ax0, v_ax1] = v_A_red_temp_v1
    for ax0, ax1, ax2 in T.grid(T.int64(2), T.int64(4096), T.int64(640)):
        with T.block("T_layer_norm"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(A[v_ax0, v_ax1, v_ax2], A_red_temp_v0[v_ax0, v_ax1], A_red_temp_v1[v_ax0, v_ax1], B[v_ax2], C[v_ax2])
            T.writes(T_layer_norm[v_ax0, v_ax1, v_ax2])
            T_layer_norm[v_ax0, v_ax1, v_ax2] = (A[v_ax0, v_ax1, v_ax2] - A_red_temp_v0[v_ax0, v_ax1] * T.float32(0.0015625000000000001)) * T.rsqrt(A_red_temp_v1[v_ax0, v_ax1] * T.float32(0.0015625000000000001) - A_red_temp_v0[v_ax0, v_ax1] * T.float32(0.0015625000000000001) * (A_red_temp_v0[v_ax0, v_ax1] * T.float32(0.0015625000000000001)) + T.float32(1.0000000000000001e-05)) * B[v_ax2] + C[v_ax2]

@T.prim_func
def layer_norm2(A: T.Buffer((T.int64(2), T.int64(1024), T.int64(1280)), "float32"), B: T.Buffer((T.int64(1280),), "float32"), C: T.Buffer((T.int64(1280),), "float32"), T_layer_norm: T.Buffer((T.int64(2), T.int64(1024), T.int64(1280)), "float32")):
    T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    A_red_temp_v0 = T.alloc_buffer((T.int64(2), T.int64(1024)))
    A_red_temp_v1 = T.alloc_buffer((T.int64(2), T.int64(1024)))
    for ax0, ax1, k2 in T.grid(T.int64(2), T.int64(1024), T.int64(1280)):
        with T.block("A_red_temp"):
            v_ax0, v_ax1, v_k2 = T.axis.remap("SSR", [ax0, ax1, k2])
            T.reads(A[v_ax0, v_ax1, v_k2])
            T.writes(A_red_temp_v0[v_ax0, v_ax1], A_red_temp_v1[v_ax0, v_ax1])
            with T.init():
                A_red_temp_v0[v_ax0, v_ax1] = T.float32(0)
                A_red_temp_v1[v_ax0, v_ax1] = T.float32(0)
            v_A_red_temp_v0: T.float32 = A_red_temp_v0[v_ax0, v_ax1] + A[v_ax0, v_ax1, v_k2]
            v_A_red_temp_v1: T.float32 = A_red_temp_v1[v_ax0, v_ax1] + A[v_ax0, v_ax1, v_k2] * A[v_ax0, v_ax1, v_k2]
            A_red_temp_v0[v_ax0, v_ax1] = v_A_red_temp_v0
            A_red_temp_v1[v_ax0, v_ax1] = v_A_red_temp_v1
    for ax0, ax1, ax2 in T.grid(T.int64(2), T.int64(1024), T.int64(1280)):
        with T.block("T_layer_norm"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(A[v_ax0, v_ax1, v_ax2], A_red_temp_v0[v_ax0, v_ax1], A_red_temp_v1[v_ax0, v_ax1], B[v_ax2], C[v_ax2])
            T.writes(T_layer_norm[v_ax0, v_ax1, v_ax2])
            T_layer_norm[v_ax0, v_ax1, v_ax2] = (A[v_ax0, v_ax1, v_ax2] - A_red_temp_v0[v_ax0, v_ax1] * T.float32(0.00078125000000000004)) * T.rsqrt(A_red_temp_v1[v_ax0, v_ax1] * T.float32(0.00078125000000000004) - A_red_temp_v0[v_ax0, v_ax1] * T.float32(0.00078125000000000004) * (A_red_temp_v0[v_ax0, v_ax1] * T.float32(0.00078125000000000004)) + T.float32(1.0000000000000001e-05)) * B[v_ax2] + C[v_ax2]

@T.prim_func
def layer_norm3(A: T.Buffer((T.int64(1), T.int64(77), T.int64(768)), "float32"), B: T.Buffer((T.int64(768),), "float32"), C: T.Buffer((T.int64(768),), "float32"), T_layer_norm: T.Buffer((T.int64(1), T.int64(77), T.int64(768)), "float32")):
    T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    A_red_temp_v0 = T.alloc_buffer((T.int64(1), T.int64(77)))
    A_red_temp_v1 = T.alloc_buffer((T.int64(1), T.int64(77)))
    for ax0, ax1, k2 in T.grid(T.int64(1), T.int64(77), T.int64(768)):
        with T.block("A_red_temp"):
            v_ax0, v_ax1, v_k2 = T.axis.remap("SSR", [ax0, ax1, k2])
            T.reads(A[v_ax0, v_ax1, v_k2])
            T.writes(A_red_temp_v0[v_ax0, v_ax1], A_red_temp_v1[v_ax0, v_ax1])
            with T.init():
                A_red_temp_v0[v_ax0, v_ax1] = T.float32(0)
                A_red_temp_v1[v_ax0, v_ax1] = T.float32(0)
            v_A_red_temp_v0: T.float32 = A_red_temp_v0[v_ax0, v_ax1] + A[v_ax0, v_ax1, v_k2]
            v_A_red_temp_v1: T.float32 = A_red_temp_v1[v_ax0, v_ax1] + A[v_ax0, v_ax1, v_k2] * A[v_ax0, v_ax1, v_k2]
            A_red_temp_v0[v_ax0, v_ax1] = v_A_red_temp_v0
            A_red_temp_v1[v_ax0, v_ax1] = v_A_red_temp_v1
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(77), T.int64(768)):
        with T.block("T_layer_norm"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(A[v_ax0, v_ax1, v_ax2], A_red_temp_v0[v_ax0, v_ax1], A_red_temp_v1[v_ax0, v_ax1], B[v_ax2], C[v_ax2])
            T.writes(T_layer_norm[v_ax0, v_ax1, v_ax2])
            T_layer_norm[v_ax0, v_ax1, v_ax2] = (A[v_ax0, v_ax1, v_ax2] - A_red_temp_v0[v_ax0, v_ax1] * T.float32(0.0013020833333333333)) * T.rsqrt(A_red_temp_v1[v_ax0, v_ax1] * T.float32(0.0013020833333333333) - A_red_temp_v0[v_ax0, v_ax1] * T.float32(0.0013020833333333333) * (A_red_temp_v0[v_ax0, v_ax1] * T.float32(0.0013020833333333333)) + T.float32(1.0000000000000001e-05)) * B[v_ax2] + C[v_ax2]

@T.prim_func
def matmul1(A: T.Buffer((T.int64(20), T.int64(77), T.int64(64)), "float32"), B: T.Buffer((T.int64(20), T.int64(64), T.int64(77)), "float32"), matmul: T.Buffer((T.int64(20), T.int64(77), T.int64(77)), "float32")):
    T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    for i0, i1, i2, k in T.grid(T.int64(20), T.int64(77), T.int64(77), T.int64(64)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(A[v_i0, v_i1, v_k], B[v_i0, v_k, v_i2])
            T.writes(matmul[v_i0, v_i1, v_i2])
            with T.init():
                matmul[v_i0, v_i1, v_i2] = T.float32(0)
            matmul[v_i0, v_i1, v_i2] = matmul[v_i0, v_i1, v_i2] + A[v_i0, v_i1, v_k] * B[v_i0, v_k, v_i2]

@T.prim_func
def matmul11(A: T.Buffer((T.int64(2), T.int64(4096), T.int64(640)), "float32"), B: T.Buffer((T.int64(640), T.int64(640)), "float32"), matmul: T.Buffer((T.int64(2), T.int64(4096), T.int64(640)), "float32")):
    T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    for i0, i1, i2, k in T.grid(T.int64(2), T.int64(4096), T.int64(640), T.int64(640)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(A[v_i0, v_i1, v_k], B[v_k, v_i2])
            T.writes(matmul[v_i0, v_i1, v_i2])
            with T.init():
                matmul[v_i0, v_i1, v_i2] = T.float32(0)
            matmul[v_i0, v_i1, v_i2] = matmul[v_i0, v_i1, v_i2] + A[v_i0, v_i1, v_k] * B[v_k, v_i2]

@T.prim_func
def matmul13(A: T.Buffer((T.int64(2), T.int64(10), T.int64(4096), T.int64(4096)), "float32"), B: T.Buffer((T.int64(2), T.int64(10), T.int64(4096), T.int64(64)), "float32"), matmul: T.Buffer((T.int64(2), T.int64(10), T.int64(4096), T.int64(64)), "float32")):
    T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    for i0, i1, i2, i3, k in T.grid(T.int64(2), T.int64(10), T.int64(4096), T.int64(64), T.int64(4096)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_i3, v_k = T.axis.remap("SSSSR", [i0, i1, i2, i3, k])
            T.reads(A[v_i0, v_i1, v_i2, v_k], B[v_i0, v_i1, v_k, v_i3])
            T.writes(matmul[v_i0, v_i1, v_i2, v_i3])
            with T.init():
                matmul[v_i0, v_i1, v_i2, v_i3] = T.float32(0)
            matmul[v_i0, v_i1, v_i2, v_i3] = matmul[v_i0, v_i1, v_i2, v_i3] + A[v_i0, v_i1, v_i2, v_k] * B[v_i0, v_i1, v_k, v_i3]

@T.prim_func
def matmul14(A: T.Buffer((T.int64(2), T.int64(77), T.int64(2048)), "float32"), B: T.Buffer((T.int64(2048), T.int64(640)), "float32"), matmul: T.Buffer((T.int64(2), T.int64(77), T.int64(640)), "float32")):
    T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    for i0, i1, i2, k in T.grid(T.int64(2), T.int64(77), T.int64(640), T.int64(2048)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(A[v_i0, v_i1, v_k], B[v_k, v_i2])
            T.writes(matmul[v_i0, v_i1, v_i2])
            with T.init():
                matmul[v_i0, v_i1, v_i2] = T.float32(0)
            matmul[v_i0, v_i1, v_i2] = matmul[v_i0, v_i1, v_i2] + A[v_i0, v_i1, v_k] * B[v_k, v_i2]

@T.prim_func
def matmul16(A: T.Buffer((T.int64(2), T.int64(10), T.int64(4096), T.int64(77)), "float32"), B: T.Buffer((T.int64(2), T.int64(10), T.int64(77), T.int64(64)), "float32"), matmul: T.Buffer((T.int64(2), T.int64(10), T.int64(4096), T.int64(64)), "float32")):
    T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    for i0, i1, i2, i3, k in T.grid(T.int64(2), T.int64(10), T.int64(4096), T.int64(64), T.int64(77)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_i3, v_k = T.axis.remap("SSSSR", [i0, i1, i2, i3, k])
            T.reads(A[v_i0, v_i1, v_i2, v_k], B[v_i0, v_i1, v_k, v_i3])
            T.writes(matmul[v_i0, v_i1, v_i2, v_i3])
            with T.init():
                matmul[v_i0, v_i1, v_i2, v_i3] = T.float32(0)
            matmul[v_i0, v_i1, v_i2, v_i3] = matmul[v_i0, v_i1, v_i2, v_i3] + A[v_i0, v_i1, v_i2, v_k] * B[v_i0, v_i1, v_k, v_i3]

@T.prim_func
def matmul19(A: T.Buffer((T.int64(2), T.int64(1024), T.int64(1280)), "float32"), B: T.Buffer((T.int64(1280), T.int64(1280)), "float32"), matmul: T.Buffer((T.int64(2), T.int64(1024), T.int64(1280)), "float32")):
    T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    for i0, i1, i2, k in T.grid(T.int64(2), T.int64(1024), T.int64(1280), T.int64(1280)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(A[v_i0, v_i1, v_k], B[v_k, v_i2])
            T.writes(matmul[v_i0, v_i1, v_i2])
            with T.init():
                matmul[v_i0, v_i1, v_i2] = T.float32(0)
            matmul[v_i0, v_i1, v_i2] = matmul[v_i0, v_i1, v_i2] + A[v_i0, v_i1, v_k] * B[v_k, v_i2]

@T.prim_func
def matmul2(A: T.Buffer((T.int64(20), T.int64(77), T.int64(77)), "float32"), B: T.Buffer((T.int64(20), T.int64(77), T.int64(64)), "float32"), matmul: T.Buffer((T.int64(20), T.int64(77), T.int64(64)), "float32")):
    T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    for i0, i1, i2, k in T.grid(T.int64(20), T.int64(77), T.int64(64), T.int64(77)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(A[v_i0, v_i1, v_k], B[v_i0, v_k, v_i2])
            T.writes(matmul[v_i0, v_i1, v_i2])
            with T.init():
                matmul[v_i0, v_i1, v_i2] = T.float32(0)
            matmul[v_i0, v_i1, v_i2] = matmul[v_i0, v_i1, v_i2] + A[v_i0, v_i1, v_k] * B[v_i0, v_k, v_i2]

@T.prim_func
def matmul21(A: T.Buffer((T.int64(2), T.int64(20), T.int64(1024), T.int64(1024)), "float32"), B: T.Buffer((T.int64(2), T.int64(20), T.int64(1024), T.int64(64)), "float32"), matmul: T.Buffer((T.int64(2), T.int64(20), T.int64(1024), T.int64(64)), "float32")):
    T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    for i0, i1, i2, i3, k in T.grid(T.int64(2), T.int64(20), T.int64(1024), T.int64(64), T.int64(1024)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_i3, v_k = T.axis.remap("SSSSR", [i0, i1, i2, i3, k])
            T.reads(A[v_i0, v_i1, v_i2, v_k], B[v_i0, v_i1, v_k, v_i3])
            T.writes(matmul[v_i0, v_i1, v_i2, v_i3])
            with T.init():
                matmul[v_i0, v_i1, v_i2, v_i3] = T.float32(0)
            matmul[v_i0, v_i1, v_i2, v_i3] = matmul[v_i0, v_i1, v_i2, v_i3] + A[v_i0, v_i1, v_i2, v_k] * B[v_i0, v_i1, v_k, v_i3]

@T.prim_func
def matmul22(A: T.Buffer((T.int64(2), T.int64(77), T.int64(2048)), "float32"), B: T.Buffer((T.int64(2048), T.int64(1280)), "float32"), matmul: T.Buffer((T.int64(2), T.int64(77), T.int64(1280)), "float32")):
    T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    for i0, i1, i2, k in T.grid(T.int64(2), T.int64(77), T.int64(1280), T.int64(2048)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(A[v_i0, v_i1, v_k], B[v_k, v_i2])
            T.writes(matmul[v_i0, v_i1, v_i2])
            with T.init():
                matmul[v_i0, v_i1, v_i2] = T.float32(0)
            matmul[v_i0, v_i1, v_i2] = matmul[v_i0, v_i1, v_i2] + A[v_i0, v_i1, v_k] * B[v_k, v_i2]

@T.prim_func
def matmul24(A: T.Buffer((T.int64(2), T.int64(20), T.int64(1024), T.int64(77)), "float32"), B: T.Buffer((T.int64(2), T.int64(20), T.int64(77), T.int64(64)), "float32"), matmul: T.Buffer((T.int64(2), T.int64(20), T.int64(1024), T.int64(64)), "float32")):
    T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    for i0, i1, i2, i3, k in T.grid(T.int64(2), T.int64(20), T.int64(1024), T.int64(64), T.int64(77)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_i3, v_k = T.axis.remap("SSSSR", [i0, i1, i2, i3, k])
            T.reads(A[v_i0, v_i1, v_i2, v_k], B[v_i0, v_i1, v_k, v_i3])
            T.writes(matmul[v_i0, v_i1, v_i2, v_i3])
            with T.init():
                matmul[v_i0, v_i1, v_i2, v_i3] = T.float32(0)
            matmul[v_i0, v_i1, v_i2, v_i3] = matmul[v_i0, v_i1, v_i2, v_i3] + A[v_i0, v_i1, v_i2, v_k] * B[v_i0, v_i1, v_k, v_i3]

@T.prim_func
def matmul29(A: T.Buffer((T.int64(1), T.int64(1), T.int64(16384), T.int64(16384)), "float32"), B: T.Buffer((T.int64(1), T.int64(1), T.int64(16384), T.int64(512)), "float32"), matmul: T.Buffer((T.int64(1), T.int64(1), T.int64(16384), T.int64(512)), "float32")):
    T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    for i0, i1, i2, i3, k in T.grid(T.int64(1), T.int64(1), T.int64(16384), T.int64(512), T.int64(16384)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_i3, v_k = T.axis.remap("SSSSR", [i0, i1, i2, i3, k])
            T.reads(A[v_i0, v_i1, v_i2, v_k], B[v_i0, v_i1, v_k, v_i3])
            T.writes(matmul[v_i0, v_i1, v_i2, v_i3])
            with T.init():
                matmul[v_i0, v_i1, v_i2, v_i3] = T.float32(0)
            matmul[v_i0, v_i1, v_i2, v_i3] = matmul[v_i0, v_i1, v_i2, v_i3] + A[v_i0, v_i1, v_i2, v_k] * B[v_i0, v_i1, v_k, v_i3]

@T.prim_func
def matmul31(A: T.Buffer((T.int64(12), T.int64(77), T.int64(64)), "float32"), B: T.Buffer((T.int64(12), T.int64(64), T.int64(77)), "float32"), matmul: T.Buffer((T.int64(12), T.int64(77), T.int64(77)), "float32")):
    T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    for i0, i1, i2, k in T.grid(T.int64(12), T.int64(77), T.int64(77), T.int64(64)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(A[v_i0, v_i1, v_k], B[v_i0, v_k, v_i2])
            T.writes(matmul[v_i0, v_i1, v_i2])
            with T.init():
                matmul[v_i0, v_i1, v_i2] = T.float32(0)
            matmul[v_i0, v_i1, v_i2] = matmul[v_i0, v_i1, v_i2] + A[v_i0, v_i1, v_k] * B[v_i0, v_k, v_i2]

@T.prim_func
def matmul32(A: T.Buffer((T.int64(12), T.int64(77), T.int64(77)), "float32"), B: T.Buffer((T.int64(12), T.int64(77), T.int64(64)), "float32"), matmul: T.Buffer((T.int64(12), T.int64(77), T.int64(64)), "float32")):
    T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    for i0, i1, i2, k in T.grid(T.int64(12), T.int64(77), T.int64(64), T.int64(77)):
        with T.block("matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(A[v_i0, v_i1, v_k], B[v_i0, v_k, v_i2])
            T.writes(matmul[v_i0, v_i1, v_i2])
            with T.init():
                matmul[v_i0, v_i1, v_i2] = T.float32(0)
            matmul[v_i0, v_i1, v_i2] = matmul[v_i0, v_i1, v_i2] + A[v_i0, v_i1, v_k] * B[v_i0, v_k, v_i2]

@T.prim_func
def matmul5(A: T.Buffer((T.int64(1), T.int64(1280)), "float32"), B: T.Buffer((T.int64(1280), T.int64(1280)), "float32"), matmul: T.Buffer((T.int64(1), T.int64(1280)), "float32")):
    T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    for i0, i1, k in T.grid(T.int64(1), T.int64(1280), T.int64(1280)):
        with T.block("matmul"):
            v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])
            T.reads(A[v_i0, v_k], B[v_k, v_i1])
            T.writes(matmul[v_i0, v_i1])
            with T.init():
                matmul[v_i0, v_i1] = T.float32(0)
            matmul[v_i0, v_i1] = matmul[v_i0, v_i1] + A[v_i0, v_k] * B[v_k, v_i1]

@T.prim_func
def multiply12(A: T.Buffer((T.int64(1), T.int64(4), T.int64(128), T.int64(128)), "float32"), T_multiply: T.Buffer((T.int64(1), T.int64(4), T.int64(128), T.int64(128)), "float32")):
    T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(4), T.int64(128), T.int64(128)):
        with T.block("T_multiply"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3])
            T.writes(T_multiply[v_ax0, v_ax1, v_ax2, v_ax3])
            T_multiply[v_ax0, v_ax1, v_ax2, v_ax3] = T.float32(7.6775431632995605) * A[v_ax0, v_ax1, v_ax2, v_ax3]

@T.prim_func
def multiply18(A: T.Buffer((T.int64(1), T.int64(4), T.int64(128), T.int64(128)), "float32"), B: T.Buffer((), "float32"), T_multiply: T.Buffer((T.int64(1), T.int64(4), T.int64(128), T.int64(128)), "float32")):
    T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), T.int64(4), T.int64(128), T.int64(128)):
        with T.block("T_multiply"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(A[v_ax0, v_ax1, v_ax2, v_ax3], B[()])
            T.writes(T_multiply[v_ax0, v_ax1, v_ax2, v_ax3])
            T_multiply[v_ax0, v_ax1, v_ax2, v_ax3] = A[v_ax0, v_ax1, v_ax2, v_ax3] * B[()]

@T.prim_func
def power(A: T.Buffer((), "float32"), T_power: T.Buffer((), "float32")):
    T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    with T.block("T_power"):
        vi = T.axis.spatial(T.int64(1), T.int64(0))
        T.reads(A[()])
        T.writes(T_power[()])
        T_power[()] = T.pow(A[()], T.float32(2))

@T.prim_func
def power1(A: T.Buffer((), "float32"), T_power: T.Buffer((), "float32")):
    T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    with T.block("T_power"):
        vi = T.axis.spatial(T.int64(1), T.int64(0))
        T.reads(A[()])
        T.writes(T_power[()])
        T_power[()] = T.pow(A[()], T.float32(0.5))

@T.prim_func
def reshape(A: T.Buffer((T.int64(1), T.int64(77)), "int32"), T_reshape: T.Buffer((T.int64(1), T.int64(77)), "int32")):
    T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    for ax0, ax1 in T.grid(T.int64(1), T.int64(77)):
        with T.block("T_reshape"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(A[T.int64(0), v_ax1 % T.int64(77)])
            T.writes(T_reshape[v_ax0, v_ax1])
            T_reshape[v_ax0, v_ax1] = A[T.int64(0), v_ax1 % T.int64(77)]

@T.prim_func
def reshape19(A: T.Buffer((T.int64(2), T.int64(320)), "float32"), T_reshape: T.Buffer((T.int64(2), T.int64(320), T.int64(1), T.int64(1)), "float32")):
    T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(320), T.int64(1), T.int64(1)):
        with T.block("T_reshape"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(A[((v_ax1 + v_ax2 + v_ax3) // T.int64(320) + v_ax0) % T.int64(2), (v_ax1 + v_ax2 + v_ax3) % T.int64(320)])
            T.writes(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3])
            T_reshape[v_ax0, v_ax1, v_ax2, v_ax3] = A[((v_ax1 + v_ax2 + v_ax3) // T.int64(320) + v_ax0) % T.int64(2), (v_ax1 + v_ax2 + v_ax3) % T.int64(320)]

@T.prim_func
def reshape21(A: T.Buffer((T.int64(2), T.int64(640)), "float32"), T_reshape: T.Buffer((T.int64(2), T.int64(640), T.int64(1), T.int64(1)), "float32")):
    T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(640), T.int64(1), T.int64(1)):
        with T.block("T_reshape"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(A[((v_ax1 + v_ax2 + v_ax3) // T.int64(640) + v_ax0) % T.int64(2), (v_ax1 + v_ax2 + v_ax3) % T.int64(640)])
            T.writes(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3])
            T_reshape[v_ax0, v_ax1, v_ax2, v_ax3] = A[((v_ax1 + v_ax2 + v_ax3) // T.int64(640) + v_ax0) % T.int64(2), (v_ax1 + v_ax2 + v_ax3) % T.int64(640)]

@T.prim_func
def reshape28(A: T.Buffer((T.int64(2), T.int64(1280)), "float32"), T_reshape: T.Buffer((T.int64(2), T.int64(1280), T.int64(1), T.int64(1)), "float32")):
    T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(2), T.int64(1280), T.int64(1), T.int64(1)):
        with T.block("T_reshape"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(A[((v_ax1 + v_ax2 + v_ax3) // T.int64(1280) + v_ax0) % T.int64(2), (v_ax1 + v_ax2 + v_ax3) % T.int64(1280)])
            T.writes(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3])
            T_reshape[v_ax0, v_ax1, v_ax2, v_ax3] = A[((v_ax1 + v_ax2 + v_ax3) // T.int64(1280) + v_ax0) % T.int64(2), (v_ax1 + v_ax2 + v_ax3) % T.int64(1280)]

@T.prim_func
def resize2d2(A: T.Buffer((T.int64(1), T.int64(512), T.int64(128), T.int64(128)), "float32"), resize: T.Buffer((T.int64(1), T.int64(512), T.int64(256), T.int64(256)), "float32")):
    T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(512), T.int64(256), T.int64(256)):
        with T.block("resize"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(A[v_i0, v_i1, T.max(T.min(T.Div(v_i2, T.int64(2)), T.int64(127)), T.int64(0)), T.max(T.min(T.Div(v_i3, T.int64(2)), T.int64(127)), T.int64(0))])
            T.writes(resize[v_i0, v_i1, v_i2, v_i3])
            resize[v_i0, v_i1, v_i2, v_i3] = A[v_i0, v_i1, T.max(T.min(T.Div(v_i2, T.int64(2)), T.int64(127)), T.int64(0)), T.max(T.min(T.Div(v_i3, T.int64(2)), T.int64(127)), T.int64(0))]

@T.prim_func
def resize2d3(A: T.Buffer((T.int64(1), T.int64(512), T.int64(256), T.int64(256)), "float32"), resize: T.Buffer((T.int64(1), T.int64(512), T.int64(512), T.int64(512)), "float32")):
    T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(512), T.int64(512), T.int64(512)):
        with T.block("resize"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(A[v_i0, v_i1, T.max(T.min(T.Div(v_i2, T.int64(2)), T.int64(255)), T.int64(0)), T.max(T.min(T.Div(v_i3, T.int64(2)), T.int64(255)), T.int64(0))])
            T.writes(resize[v_i0, v_i1, v_i2, v_i3])
            resize[v_i0, v_i1, v_i2, v_i3] = A[v_i0, v_i1, T.max(T.min(T.Div(v_i2, T.int64(2)), T.int64(255)), T.int64(0)), T.max(T.min(T.Div(v_i3, T.int64(2)), T.int64(255)), T.int64(0))]

@T.prim_func
def resize2d4(A: T.Buffer((T.int64(1), T.int64(256), T.int64(512), T.int64(512)), "float32"), resize: T.Buffer((T.int64(1), T.int64(256), T.int64(1024), T.int64(1024)), "float32")):
    T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(256), T.int64(1024), T.int64(1024)):
        with T.block("resize"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(A[v_i0, v_i1, T.max(T.min(T.Div(v_i2, T.int64(2)), T.int64(511)), T.int64(0)), T.max(T.min(T.Div(v_i3, T.int64(2)), T.int64(511)), T.int64(0))])
            T.writes(resize[v_i0, v_i1, v_i2, v_i3])
            resize[v_i0, v_i1, v_i2, v_i3] = A[v_i0, v_i1, T.max(T.min(T.Div(v_i2, T.int64(2)), T.int64(511)), T.int64(0)), T.max(T.min(T.Div(v_i3, T.int64(2)), T.int64(511)), T.int64(0))]

@T.prim_func
def silu(A: T.Buffer((T.int64(2), T.int64(1280)), "float32"), T_multiply: T.Buffer((T.int64(2), T.int64(1280)), "float32")):
    T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    compute = T.alloc_buffer((T.int64(2), T.int64(1280)))
    for i0, i1 in T.grid(T.int64(2), T.int64(1280)):
        with T.block("compute"):
            v_i0, v_i1 = T.axis.remap("SS", [i0, i1])
            T.reads(A[v_i0, v_i1])
            T.writes(compute[v_i0, v_i1])
            compute[v_i0, v_i1] = T.sigmoid(A[v_i0, v_i1])
    for ax0, ax1 in T.grid(T.int64(2), T.int64(1280)):
        with T.block("T_multiply"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(A[v_ax0, v_ax1], compute[v_ax0, v_ax1])
            T.writes(T_multiply[v_ax0, v_ax1])
            T_multiply[v_ax0, v_ax1] = A[v_ax0, v_ax1] * compute[v_ax0, v_ax1]

@T.prim_func
def softmax(A: T.Buffer((T.int64(20), T.int64(77), T.int64(77)), "float32"), T_softmax_norm: T.Buffer((T.int64(20), T.int64(77), T.int64(77)), "float32")):
    T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    T_softmax_maxelem = T.alloc_buffer((T.int64(20), T.int64(77)))
    T_softmax_exp = T.alloc_buffer((T.int64(20), T.int64(77), T.int64(77)))
    T_softmax_expsum = T.alloc_buffer((T.int64(20), T.int64(77)))
    for i0, i1, k in T.grid(T.int64(20), T.int64(77), T.int64(77)):
        with T.block("T_softmax_maxelem"):
            v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])
            T.reads(A[v_i0, v_i1, v_k])
            T.writes(T_softmax_maxelem[v_i0, v_i1])
            with T.init():
                T_softmax_maxelem[v_i0, v_i1] = T.float32(-3.4028234663852886e+38)
            T_softmax_maxelem[v_i0, v_i1] = T.max(T_softmax_maxelem[v_i0, v_i1], A[v_i0, v_i1, v_k])
    for i0, i1, i2 in T.grid(T.int64(20), T.int64(77), T.int64(77)):
        with T.block("T_softmax_exp"):
            v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
            T.reads(A[v_i0, v_i1, v_i2], T_softmax_maxelem[v_i0, v_i1])
            T.writes(T_softmax_exp[v_i0, v_i1, v_i2])
            T_softmax_exp[v_i0, v_i1, v_i2] = T.exp(A[v_i0, v_i1, v_i2] - T_softmax_maxelem[v_i0, v_i1])
    for i0, i1, k in T.grid(T.int64(20), T.int64(77), T.int64(77)):
        with T.block("T_softmax_expsum"):
            v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])
            T.reads(T_softmax_exp[v_i0, v_i1, v_k])
            T.writes(T_softmax_expsum[v_i0, v_i1])
            with T.init():
                T_softmax_expsum[v_i0, v_i1] = T.float32(0)
            T_softmax_expsum[v_i0, v_i1] = T_softmax_expsum[v_i0, v_i1] + T_softmax_exp[v_i0, v_i1, v_k]
    for i0, i1, i2 in T.grid(T.int64(20), T.int64(77), T.int64(77)):
        with T.block("T_softmax_norm"):
            v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
            T.reads(T_softmax_exp[v_i0, v_i1, v_i2], T_softmax_expsum[v_i0, v_i1])
            T.writes(T_softmax_norm[v_i0, v_i1, v_i2])
            T.block_attr({"axis": 2})
            T_softmax_norm[v_i0, v_i1, v_i2] = T_softmax_exp[v_i0, v_i1, v_i2] / T_softmax_expsum[v_i0, v_i1]

@T.prim_func
def softmax1(A: T.Buffer((T.int64(2), T.int64(10), T.int64(4096), T.int64(4096)), "float32"), T_softmax_norm: T.Buffer((T.int64(2), T.int64(10), T.int64(4096), T.int64(4096)), "float32")):
    T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    T_softmax_maxelem = T.alloc_buffer((T.int64(2), T.int64(10), T.int64(4096)))
    T_softmax_exp = T.alloc_buffer((T.int64(2), T.int64(10), T.int64(4096), T.int64(4096)))
    T_softmax_expsum = T.alloc_buffer((T.int64(2), T.int64(10), T.int64(4096)))
    for i0, i1, i2, k in T.grid(T.int64(2), T.int64(10), T.int64(4096), T.int64(4096)):
        with T.block("T_softmax_maxelem"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(A[v_i0, v_i1, v_i2, v_k])
            T.writes(T_softmax_maxelem[v_i0, v_i1, v_i2])
            with T.init():
                T_softmax_maxelem[v_i0, v_i1, v_i2] = T.float32(-3.4028234663852886e+38)
            T_softmax_maxelem[v_i0, v_i1, v_i2] = T.max(T_softmax_maxelem[v_i0, v_i1, v_i2], A[v_i0, v_i1, v_i2, v_k])
    for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(10), T.int64(4096), T.int64(4096)):
        with T.block("T_softmax_exp"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(A[v_i0, v_i1, v_i2, v_i3], T_softmax_maxelem[v_i0, v_i1, v_i2])
            T.writes(T_softmax_exp[v_i0, v_i1, v_i2, v_i3])
            T_softmax_exp[v_i0, v_i1, v_i2, v_i3] = T.exp(A[v_i0, v_i1, v_i2, v_i3] - T_softmax_maxelem[v_i0, v_i1, v_i2])
    for i0, i1, i2, k in T.grid(T.int64(2), T.int64(10), T.int64(4096), T.int64(4096)):
        with T.block("T_softmax_expsum"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(T_softmax_exp[v_i0, v_i1, v_i2, v_k])
            T.writes(T_softmax_expsum[v_i0, v_i1, v_i2])
            with T.init():
                T_softmax_expsum[v_i0, v_i1, v_i2] = T.float32(0)
            T_softmax_expsum[v_i0, v_i1, v_i2] = T_softmax_expsum[v_i0, v_i1, v_i2] + T_softmax_exp[v_i0, v_i1, v_i2, v_k]
    for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(10), T.int64(4096), T.int64(4096)):
        with T.block("T_softmax_norm"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(T_softmax_exp[v_i0, v_i1, v_i2, v_i3], T_softmax_expsum[v_i0, v_i1, v_i2])
            T.writes(T_softmax_norm[v_i0, v_i1, v_i2, v_i3])
            T.block_attr({"axis": 3})
            T_softmax_norm[v_i0, v_i1, v_i2, v_i3] = T_softmax_exp[v_i0, v_i1, v_i2, v_i3] / T_softmax_expsum[v_i0, v_i1, v_i2]

@T.prim_func
def softmax2(A: T.Buffer((T.int64(2), T.int64(10), T.int64(4096), T.int64(77)), "float32"), T_softmax_norm: T.Buffer((T.int64(2), T.int64(10), T.int64(4096), T.int64(77)), "float32")):
    T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    T_softmax_maxelem = T.alloc_buffer((T.int64(2), T.int64(10), T.int64(4096)))
    T_softmax_exp = T.alloc_buffer((T.int64(2), T.int64(10), T.int64(4096), T.int64(77)))
    T_softmax_expsum = T.alloc_buffer((T.int64(2), T.int64(10), T.int64(4096)))
    for i0, i1, i2, k in T.grid(T.int64(2), T.int64(10), T.int64(4096), T.int64(77)):
        with T.block("T_softmax_maxelem"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(A[v_i0, v_i1, v_i2, v_k])
            T.writes(T_softmax_maxelem[v_i0, v_i1, v_i2])
            with T.init():
                T_softmax_maxelem[v_i0, v_i1, v_i2] = T.float32(-3.4028234663852886e+38)
            T_softmax_maxelem[v_i0, v_i1, v_i2] = T.max(T_softmax_maxelem[v_i0, v_i1, v_i2], A[v_i0, v_i1, v_i2, v_k])
    for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(10), T.int64(4096), T.int64(77)):
        with T.block("T_softmax_exp"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(A[v_i0, v_i1, v_i2, v_i3], T_softmax_maxelem[v_i0, v_i1, v_i2])
            T.writes(T_softmax_exp[v_i0, v_i1, v_i2, v_i3])
            T_softmax_exp[v_i0, v_i1, v_i2, v_i3] = T.exp(A[v_i0, v_i1, v_i2, v_i3] - T_softmax_maxelem[v_i0, v_i1, v_i2])
    for i0, i1, i2, k in T.grid(T.int64(2), T.int64(10), T.int64(4096), T.int64(77)):
        with T.block("T_softmax_expsum"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(T_softmax_exp[v_i0, v_i1, v_i2, v_k])
            T.writes(T_softmax_expsum[v_i0, v_i1, v_i2])
            with T.init():
                T_softmax_expsum[v_i0, v_i1, v_i2] = T.float32(0)
            T_softmax_expsum[v_i0, v_i1, v_i2] = T_softmax_expsum[v_i0, v_i1, v_i2] + T_softmax_exp[v_i0, v_i1, v_i2, v_k]
    for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(10), T.int64(4096), T.int64(77)):
        with T.block("T_softmax_norm"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(T_softmax_exp[v_i0, v_i1, v_i2, v_i3], T_softmax_expsum[v_i0, v_i1, v_i2])
            T.writes(T_softmax_norm[v_i0, v_i1, v_i2, v_i3])
            T.block_attr({"axis": 3})
            T_softmax_norm[v_i0, v_i1, v_i2, v_i3] = T_softmax_exp[v_i0, v_i1, v_i2, v_i3] / T_softmax_expsum[v_i0, v_i1, v_i2]

@T.prim_func
def softmax3(A: T.Buffer((T.int64(2), T.int64(20), T.int64(1024), T.int64(1024)), "float32"), T_softmax_norm: T.Buffer((T.int64(2), T.int64(20), T.int64(1024), T.int64(1024)), "float32")):
    T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    T_softmax_maxelem = T.alloc_buffer((T.int64(2), T.int64(20), T.int64(1024)))
    T_softmax_exp = T.alloc_buffer((T.int64(2), T.int64(20), T.int64(1024), T.int64(1024)))
    T_softmax_expsum = T.alloc_buffer((T.int64(2), T.int64(20), T.int64(1024)))
    for i0, i1, i2, k in T.grid(T.int64(2), T.int64(20), T.int64(1024), T.int64(1024)):
        with T.block("T_softmax_maxelem"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(A[v_i0, v_i1, v_i2, v_k])
            T.writes(T_softmax_maxelem[v_i0, v_i1, v_i2])
            with T.init():
                T_softmax_maxelem[v_i0, v_i1, v_i2] = T.float32(-3.4028234663852886e+38)
            T_softmax_maxelem[v_i0, v_i1, v_i2] = T.max(T_softmax_maxelem[v_i0, v_i1, v_i2], A[v_i0, v_i1, v_i2, v_k])
    for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(20), T.int64(1024), T.int64(1024)):
        with T.block("T_softmax_exp"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(A[v_i0, v_i1, v_i2, v_i3], T_softmax_maxelem[v_i0, v_i1, v_i2])
            T.writes(T_softmax_exp[v_i0, v_i1, v_i2, v_i3])
            T_softmax_exp[v_i0, v_i1, v_i2, v_i3] = T.exp(A[v_i0, v_i1, v_i2, v_i3] - T_softmax_maxelem[v_i0, v_i1, v_i2])
    for i0, i1, i2, k in T.grid(T.int64(2), T.int64(20), T.int64(1024), T.int64(1024)):
        with T.block("T_softmax_expsum"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(T_softmax_exp[v_i0, v_i1, v_i2, v_k])
            T.writes(T_softmax_expsum[v_i0, v_i1, v_i2])
            with T.init():
                T_softmax_expsum[v_i0, v_i1, v_i2] = T.float32(0)
            T_softmax_expsum[v_i0, v_i1, v_i2] = T_softmax_expsum[v_i0, v_i1, v_i2] + T_softmax_exp[v_i0, v_i1, v_i2, v_k]
    for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(20), T.int64(1024), T.int64(1024)):
        with T.block("T_softmax_norm"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(T_softmax_exp[v_i0, v_i1, v_i2, v_i3], T_softmax_expsum[v_i0, v_i1, v_i2])
            T.writes(T_softmax_norm[v_i0, v_i1, v_i2, v_i3])
            T.block_attr({"axis": 3})
            T_softmax_norm[v_i0, v_i1, v_i2, v_i3] = T_softmax_exp[v_i0, v_i1, v_i2, v_i3] / T_softmax_expsum[v_i0, v_i1, v_i2]

@T.prim_func
def softmax4(A: T.Buffer((T.int64(2), T.int64(20), T.int64(1024), T.int64(77)), "float32"), T_softmax_norm: T.Buffer((T.int64(2), T.int64(20), T.int64(1024), T.int64(77)), "float32")):
    T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    T_softmax_maxelem = T.alloc_buffer((T.int64(2), T.int64(20), T.int64(1024)))
    T_softmax_exp = T.alloc_buffer((T.int64(2), T.int64(20), T.int64(1024), T.int64(77)))
    T_softmax_expsum = T.alloc_buffer((T.int64(2), T.int64(20), T.int64(1024)))
    for i0, i1, i2, k in T.grid(T.int64(2), T.int64(20), T.int64(1024), T.int64(77)):
        with T.block("T_softmax_maxelem"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(A[v_i0, v_i1, v_i2, v_k])
            T.writes(T_softmax_maxelem[v_i0, v_i1, v_i2])
            with T.init():
                T_softmax_maxelem[v_i0, v_i1, v_i2] = T.float32(-3.4028234663852886e+38)
            T_softmax_maxelem[v_i0, v_i1, v_i2] = T.max(T_softmax_maxelem[v_i0, v_i1, v_i2], A[v_i0, v_i1, v_i2, v_k])
    for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(20), T.int64(1024), T.int64(77)):
        with T.block("T_softmax_exp"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(A[v_i0, v_i1, v_i2, v_i3], T_softmax_maxelem[v_i0, v_i1, v_i2])
            T.writes(T_softmax_exp[v_i0, v_i1, v_i2, v_i3])
            T_softmax_exp[v_i0, v_i1, v_i2, v_i3] = T.exp(A[v_i0, v_i1, v_i2, v_i3] - T_softmax_maxelem[v_i0, v_i1, v_i2])
    for i0, i1, i2, k in T.grid(T.int64(2), T.int64(20), T.int64(1024), T.int64(77)):
        with T.block("T_softmax_expsum"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(T_softmax_exp[v_i0, v_i1, v_i2, v_k])
            T.writes(T_softmax_expsum[v_i0, v_i1, v_i2])
            with T.init():
                T_softmax_expsum[v_i0, v_i1, v_i2] = T.float32(0)
            T_softmax_expsum[v_i0, v_i1, v_i2] = T_softmax_expsum[v_i0, v_i1, v_i2] + T_softmax_exp[v_i0, v_i1, v_i2, v_k]
    for i0, i1, i2, i3 in T.grid(T.int64(2), T.int64(20), T.int64(1024), T.int64(77)):
        with T.block("T_softmax_norm"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(T_softmax_exp[v_i0, v_i1, v_i2, v_i3], T_softmax_expsum[v_i0, v_i1, v_i2])
            T.writes(T_softmax_norm[v_i0, v_i1, v_i2, v_i3])
            T.block_attr({"axis": 3})
            T_softmax_norm[v_i0, v_i1, v_i2, v_i3] = T_softmax_exp[v_i0, v_i1, v_i2, v_i3] / T_softmax_expsum[v_i0, v_i1, v_i2]

@T.prim_func
def softmax5(A: T.Buffer((T.int64(1), T.int64(1), T.int64(16384), T.int64(16384)), "float32"), T_softmax_norm: T.Buffer((T.int64(1), T.int64(1), T.int64(16384), T.int64(16384)), "float32")):
    T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    T_softmax_maxelem = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(16384)))
    T_softmax_exp = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(16384), T.int64(16384)))
    T_softmax_expsum = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(16384)))
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(16384), T.int64(16384)):
        with T.block("T_softmax_maxelem"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(A[v_i0, v_i1, v_i2, v_k])
            T.writes(T_softmax_maxelem[v_i0, v_i1, v_i2])
            with T.init():
                T_softmax_maxelem[v_i0, v_i1, v_i2] = T.float32(-3.4028234663852886e+38)
            T_softmax_maxelem[v_i0, v_i1, v_i2] = T.max(T_softmax_maxelem[v_i0, v_i1, v_i2], A[v_i0, v_i1, v_i2, v_k])
    for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(1), T.int64(16384), T.int64(16384)):
        with T.block("T_softmax_exp"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(A[v_i0, v_i1, v_i2, v_i3], T_softmax_maxelem[v_i0, v_i1, v_i2])
            T.writes(T_softmax_exp[v_i0, v_i1, v_i2, v_i3])
            T_softmax_exp[v_i0, v_i1, v_i2, v_i3] = T.exp(A[v_i0, v_i1, v_i2, v_i3] - T_softmax_maxelem[v_i0, v_i1, v_i2])
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(16384), T.int64(16384)):
        with T.block("T_softmax_expsum"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(T_softmax_exp[v_i0, v_i1, v_i2, v_k])
            T.writes(T_softmax_expsum[v_i0, v_i1, v_i2])
            with T.init():
                T_softmax_expsum[v_i0, v_i1, v_i2] = T.float32(0)
            T_softmax_expsum[v_i0, v_i1, v_i2] = T_softmax_expsum[v_i0, v_i1, v_i2] + T_softmax_exp[v_i0, v_i1, v_i2, v_k]
    for i0, i1, i2, i3 in T.grid(T.int64(1), T.int64(1), T.int64(16384), T.int64(16384)):
        with T.block("T_softmax_norm"):
            v_i0, v_i1, v_i2, v_i3 = T.axis.remap("SSSS", [i0, i1, i2, i3])
            T.reads(T_softmax_exp[v_i0, v_i1, v_i2, v_i3], T_softmax_expsum[v_i0, v_i1, v_i2])
            T.writes(T_softmax_norm[v_i0, v_i1, v_i2, v_i3])
            T.block_attr({"axis": 3})
            T_softmax_norm[v_i0, v_i1, v_i2, v_i3] = T_softmax_exp[v_i0, v_i1, v_i2, v_i3] / T_softmax_expsum[v_i0, v_i1, v_i2]

@T.prim_func
def softmax6(A: T.Buffer((T.int64(12), T.int64(77), T.int64(77)), "float32"), T_softmax_norm: T.Buffer((T.int64(12), T.int64(77), T.int64(77)), "float32")):
    T.func_attr({"op_pattern": 4, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    T_softmax_maxelem = T.alloc_buffer((T.int64(12), T.int64(77)))
    T_softmax_exp = T.alloc_buffer((T.int64(12), T.int64(77), T.int64(77)))
    T_softmax_expsum = T.alloc_buffer((T.int64(12), T.int64(77)))
    for i0, i1, k in T.grid(T.int64(12), T.int64(77), T.int64(77)):
        with T.block("T_softmax_maxelem"):
            v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])
            T.reads(A[v_i0, v_i1, v_k])
            T.writes(T_softmax_maxelem[v_i0, v_i1])
            with T.init():
                T_softmax_maxelem[v_i0, v_i1] = T.float32(-3.4028234663852886e+38)
            T_softmax_maxelem[v_i0, v_i1] = T.max(T_softmax_maxelem[v_i0, v_i1], A[v_i0, v_i1, v_k])
    for i0, i1, i2 in T.grid(T.int64(12), T.int64(77), T.int64(77)):
        with T.block("T_softmax_exp"):
            v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
            T.reads(A[v_i0, v_i1, v_i2], T_softmax_maxelem[v_i0, v_i1])
            T.writes(T_softmax_exp[v_i0, v_i1, v_i2])
            T_softmax_exp[v_i0, v_i1, v_i2] = T.exp(A[v_i0, v_i1, v_i2] - T_softmax_maxelem[v_i0, v_i1])
    for i0, i1, k in T.grid(T.int64(12), T.int64(77), T.int64(77)):
        with T.block("T_softmax_expsum"):
            v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])
            T.reads(T_softmax_exp[v_i0, v_i1, v_k])
            T.writes(T_softmax_expsum[v_i0, v_i1])
            with T.init():
                T_softmax_expsum[v_i0, v_i1] = T.float32(0)
            T_softmax_expsum[v_i0, v_i1] = T_softmax_expsum[v_i0, v_i1] + T_softmax_exp[v_i0, v_i1, v_k]
    for i0, i1, i2 in T.grid(T.int64(12), T.int64(77), T.int64(77)):
        with T.block("T_softmax_norm"):
            v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
            T.reads(T_softmax_exp[v_i0, v_i1, v_i2], T_softmax_expsum[v_i0, v_i1])
            T.writes(T_softmax_norm[v_i0, v_i1, v_i2])
            T.block_attr({"axis": 2})
            T_softmax_norm[v_i0, v_i1, v_i2] = T_softmax_exp[v_i0, v_i1, v_i2] / T_softmax_expsum[v_i0, v_i1]

@T.prim_func
def squeeze(A: T.Buffer((T.int64(1), T.int64(77), T.int64(1280)), "float32"), T_squeeze: T.Buffer((T.int64(77), T.int64(1280)), "float32")):
    T.func_attr({"op_pattern": 1, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    for ax0, ax1 in T.grid(T.int64(77), T.int64(1280)):
        with T.block("T_squeeze"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(A[T.int64(0), v_ax0, v_ax1])
            T.writes(T_squeeze[v_ax0, v_ax1])
            T_squeeze[v_ax0, v_ax1] = A[T.int64(0), v_ax0, v_ax1]

@T.prim_func
def subtract1(A: T.Buffer((), "float32"), B: T.Buffer((), "float32"), T_subtract: T.Buffer((), "float32")):
    T.func_attr({"op_pattern": 0, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    with T.block("T_subtract"):
        vi = T.axis.spatial(1, T.int64(0))
        T.reads(A[()], B[()])
        T.writes(T_subtract[()])
        T_subtract[()] = A[()] - B[()]

@T.prim_func
def take(A: T.Buffer((T.int64(49408), T.int64(1280)), "float32"), B: T.Buffer((T.int64(77),), "int32"), T_take: T.Buffer((T.int64(77), T.int64(1280)), "float32")):
    T.func_attr({"op_pattern": 8, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    for ax0, ax1 in T.grid(T.int64(77), T.int64(1280)):
        with T.block("T_take"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(A[B[v_ax0], v_ax1], B[v_ax0])
            T.writes(T_take[v_ax0, v_ax1])
            T_take[v_ax0, v_ax1] = A[B[v_ax0], v_ax1]

@T.prim_func
def take2(A: T.Buffer((T.int64(77), T.int64(1280)), "float32"), B: T.Buffer((T.int64(1),), "int64"), T_take: T.Buffer((T.int64(1), T.int64(1280)), "float32")):
    T.func_attr({"op_pattern": 8, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    for ax0, ax1 in T.grid(T.int64(1), T.int64(1280)):
        with T.block("T_take"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(A[B[v_ax0], v_ax1], B[v_ax0])
            T.writes(T_take[v_ax0, v_ax1])
            T_take[v_ax0, v_ax1] = A[B[v_ax0], v_ax1]

@T.prim_func
def take3(A: T.Buffer((T.int64(49408), T.int64(768)), "float32"), B: T.Buffer((T.int64(77),), "int32"), T_take: T.Buffer((T.int64(77), T.int64(768)), "float32")):
    T.func_attr({"op_pattern": 8, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    for ax0, ax1 in T.grid(T.int64(77), T.int64(768)):
        with T.block("T_take"):
            v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
            T.reads(A[B[v_ax0], v_ax1], B[v_ax0])
            T.writes(T_take[v_ax0, v_ax1])
            T_take[v_ax0, v_ax1] = A[B[v_ax0], v_ax1]

@T.prim_func
def tir_image_to_rgba(A: T.Buffer((T.int64(1), T.int64(1024), T.int64(1024), T.int64(3)), "float32"), image_to_rgba: T.Buffer((T.int64(1024), T.int64(1024)), "uint32")):
    T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    for y, x in T.grid(T.int64(1024), T.int64(1024)):
        with T.block("image_to_rgba"):
            v_y, v_x = T.axis.remap("SS", [y, x])
            T.reads(A[T.int64(0), v_y, v_x, T.int64(0):T.int64(3)])
            T.writes(image_to_rgba[v_y, v_x])
            image_to_rgba[v_y, v_x] = T.bitwise_or(T.bitwise_or(T.bitwise_or(T.Cast("uint32", A[T.int64(0), v_y, v_x, T.int64(0)]), T.shift_left(T.Cast("uint32", A[T.int64(0), v_y, v_x, T.int64(1)]), T.uint32(8))), T.shift_left(T.Cast("uint32", A[T.int64(0), v_y, v_x, T.int64(2)]), T.uint32(16))), T.uint32(4278190080))

@T.prim_func
def transpose30(A: T.Buffer((T.int64(1), T.int64(512), T.int64(16384)), "float32"), T_transpose: T.Buffer((T.int64(1), T.int64(16384), T.int64(512)), "float32")):
    T.func_attr({"op_pattern": 2, "tir.noalias": T.bool(True)})
    # with T.block("root"):
    for ax0, ax1, ax2 in T.grid(T.int64(1), T.int64(16384), T.int64(512)):
        with T.block("T_transpose"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(A[v_ax0, v_ax2, v_ax1])
            T.writes(T_transpose[v_ax0, v_ax1, v_ax2])
            T_transpose[v_ax0, v_ax1, v_ax2] = A[v_ax0, v_ax2, v_ax1]

DlightBench.register_bench_workload(fused_conv2d14_add26, "stable_diffusion_xl", "fused_conv2d14_add26")
DlightBench.register_bench_workload(fused_conv2d21_add27, "stable_diffusion_xl", "fused_conv2d21_add27")
DlightBench.register_bench_workload(fused_matmul28_multiply13, "stable_diffusion_xl", "fused_matmul28_multiply13")
DlightBench.register_bench_workload(matmul29, "stable_diffusion_xl", "matmul29")
DlightBench.register_bench_workload(fused_conv2d30_add34, "stable_diffusion_xl", "fused_conv2d30_add34")
DlightBench.register_bench_workload(fused_conv2d30_add34_add35_divide7, "stable_diffusion_xl", "fused_conv2d30_add34_add35_divide7")
DlightBench.register_bench_workload(fused_conv2d31_add36, "stable_diffusion_xl", "fused_conv2d31_add36")
DlightBench.register_bench_workload(fused_conv2d32_add37, "stable_diffusion_xl", "fused_conv2d32_add37")
DlightBench.register_bench_workload(fused_conv2d33_add37_add38_divide8, "stable_diffusion_xl", "fused_conv2d33_add37_add38_divide8")
DlightBench.register_bench_workload(fused_conv2d33_add37, "stable_diffusion_xl", "fused_conv2d33_add37")
DlightBench.register_bench_workload(fused_conv2d35_add39, "stable_diffusion_xl", "fused_conv2d35_add39")
DlightBench.register_bench_workload(fused_conv2d36_add40, "stable_diffusion_xl", "fused_conv2d36_add40")
DlightBench.register_bench_workload(fused_conv2d37_add40_add41_divide9, "stable_diffusion_xl", "fused_conv2d37_add40_add41_divide9")
DlightBench.register_bench_workload(fused_conv2d37_add40, "stable_diffusion_xl", "fused_conv2d37_add40")

DlightBench.register_bench_workload(fused_split2_subtract_multiply11_add29, "stable_diffusion_xl", "fused_split2_subtract_multiply11_add29")
DlightBench.register_bench_workload(fused_conv2d26_add28, "stable_diffusion_xl", "fused_conv2d26_add28")
DlightBench.register_bench_workload(concatenate10, "stable_diffusion_xl", "concatenate10")
DlightBench.register_bench_workload(fused_group_norm11_silu10, "stable_diffusion_xl", "fused_group_norm11_silu10")
DlightBench.register_bench_workload(fused_group_norm10_silu9, "stable_diffusion_xl", "fused_group_norm10_silu9")
DlightBench.register_bench_workload(fused_reshape26_transpose20_add15_concatenate7, "stable_diffusion_xl", "fused_reshape26_transpose20_add15_concatenate7")
DlightBench.register_bench_workload(fused_conv2d16_add12, "stable_diffusion_xl", "fused_conv2d16_add12")
DlightBench.register_bench_workload(fused_conv2d15_add12_add14, "stable_diffusion_xl", "fused_conv2d15_add12_add14")
DlightBench.register_bench_workload(concatenate6, "stable_diffusion_xl", "concatenate6")
DlightBench.register_bench_workload(fused_conv2d13_add20, "stable_diffusion_xl", "fused_conv2d13_add20")
DlightBench.register_bench_workload(fused_conv2d23_add7, "stable_diffusion_xl", "fused_conv2d23_add7")
DlightBench.register_bench_workload(fused_group_norm8_silu7, "stable_diffusion_xl", "fused_group_norm8_silu7")
DlightBench.register_bench_workload(fused_reshape33_transpose29_add22_concatenate5, "stable_diffusion_xl", "fused_reshape33_transpose29_add22_concatenate5")
DlightBench.register_bench_workload(concatenate4, "stable_diffusion_xl", "concatenate4")
DlightBench.register_bench_workload(fused_conv2d8_add20_add21, "stable_diffusion_xl", "fused_conv2d8_add20_add21")
DlightBench.register_bench_workload(fused_matmul26_add23_add24, "stable_diffusion_xl", "fused_matmul26_add23_add24")
DlightBench.register_bench_workload(fused_split1_gelu2_multiply10, "stable_diffusion_xl", "fused_split1_gelu2_multiply10")
DlightBench.register_bench_workload(fused_conv2d17_add12_add14, "stable_diffusion_xl", "fused_conv2d17_add12_add14")
DlightBench.register_bench_workload(fused_reshape32_transpose26_transpose27, "stable_diffusion_xl", "fused_reshape32_transpose26_transpose27")
DlightBench.register_bench_workload(matmul22, "stable_diffusion_xl", "matmul22")
DlightBench.register_bench_workload(fused_conv2d19_add12_add14, "stable_diffusion_xl", "fused_conv2d19_add12_add14")
DlightBench.register_bench_workload(fused_transpose24_reshape31, "stable_diffusion_xl", "fused_transpose24_reshape31")
DlightBench.register_bench_workload(softmax3, "stable_diffusion_xl", "softmax3")
DlightBench.register_bench_workload(fused_group_norm13_silu12, "stable_diffusion_xl", "fused_group_norm13_silu12")
DlightBench.register_bench_workload(fused_reshape30_transpose22_transpose23, "stable_diffusion_xl", "fused_reshape30_transpose22_transpose23")
DlightBench.register_bench_workload(group_norm6, "stable_diffusion_xl", "group_norm6")
DlightBench.register_bench_workload(fused_conv2d8_add20_add22_divide4, "stable_diffusion_xl", "fused_conv2d8_add20_add22_divide4")
DlightBench.register_bench_workload(fused_group_norm5_silu5, "stable_diffusion_xl", "fused_group_norm5_silu5")
DlightBench.register_bench_workload(fused_matmul25_add25, "stable_diffusion_xl", "fused_matmul25_add25")
DlightBench.register_bench_workload(fused_conv2d7_add20_add21, "stable_diffusion_xl", "fused_conv2d7_add20_add21")
DlightBench.register_bench_workload(fused_transpose21_reshape29, "stable_diffusion_xl", "fused_transpose21_reshape29")
DlightBench.register_bench_workload(fused_group_norm4_silu4, "stable_diffusion_xl", "fused_group_norm4_silu4")
DlightBench.register_bench_workload(fused_reshape26_transpose20_add15_concatenate8, "stable_diffusion_xl", "fused_reshape26_transpose20_add15_concatenate8")
DlightBench.register_bench_workload(fused_conv2d6_add19, "stable_diffusion_xl", "fused_conv2d6_add19")
DlightBench.register_bench_workload(fused_split_gelu1_multiply7, "stable_diffusion_xl", "fused_split_gelu1_multiply7")
DlightBench.register_bench_workload(matmul16, "stable_diffusion_xl", "matmul16")
DlightBench.register_bench_workload(softmax2, "stable_diffusion_xl", "softmax2")
DlightBench.register_bench_workload(matmul19, "stable_diffusion_xl", "matmul19")
DlightBench.register_bench_workload(fused_reshape25_transpose16, "stable_diffusion_xl", "fused_reshape25_transpose16")
DlightBench.register_bench_workload(fused_reshape25_transpose16_transpose17, "stable_diffusion_xl", "fused_reshape25_transpose16_transpose17")
DlightBench.register_bench_workload(fused_transpose14_reshape24, "stable_diffusion_xl", "fused_transpose14_reshape24")
DlightBench.register_bench_workload(matmul13, "stable_diffusion_xl", "matmul13")
DlightBench.register_bench_workload(matmul21, "stable_diffusion_xl", "matmul21")
DlightBench.register_bench_workload(fused_matmul15_multiply6, "stable_diffusion_xl", "fused_matmul15_multiply6")
DlightBench.register_bench_workload(softmax1, "stable_diffusion_xl", "softmax1")
DlightBench.register_bench_workload(fused_group_norm7_silu6, "stable_diffusion_xl", "fused_group_norm7_silu6")
DlightBench.register_bench_workload(fused_reshape23_transpose12_transpose13, "stable_diffusion_xl", "fused_reshape23_transpose12_transpose13")
DlightBench.register_bench_workload(fused_reshape23_transpose12, "stable_diffusion_xl", "fused_reshape23_transpose12")
DlightBench.register_bench_workload(fused_matmul11_add16, "stable_diffusion_xl", "fused_matmul11_add16")
DlightBench.register_bench_workload(group_norm3, "stable_diffusion_xl", "group_norm3")
DlightBench.register_bench_workload(concatenate9, "stable_diffusion_xl", "concatenate9")
DlightBench.register_bench_workload(fused_reshape30_transpose22, "stable_diffusion_xl", "fused_reshape30_transpose22")
DlightBench.register_bench_workload(fused_conv2d4_add12_add15_divide1, "stable_diffusion_xl", "fused_conv2d4_add12_add15_divide1")
DlightBench.register_bench_workload(fused_group_norm2_silu3, "stable_diffusion_xl", "fused_group_norm2_silu3")
DlightBench.register_bench_workload(fused_matmul10_add13_strided_slice7, "stable_diffusion_xl", "fused_matmul10_add13_strided_slice7")
DlightBench.register_bench_workload(fused_conv2d20_add12, "stable_diffusion_xl", "fused_conv2d20_add12")
DlightBench.register_bench_workload(fused_group_norm1_silu2, "stable_diffusion_xl", "fused_group_norm1_silu2")
DlightBench.register_bench_workload(fused_transpose10_reshape22, "stable_diffusion_xl", "fused_transpose10_reshape22")
DlightBench.register_bench_workload(reshape19, "stable_diffusion_xl", "reshape19")
DlightBench.register_bench_workload(fused_matmul7_add5_add6, "stable_diffusion_xl", "fused_matmul7_add5_add6")
DlightBench.register_bench_workload(fused_group_norm12_silu11, "stable_diffusion_xl", "fused_group_norm12_silu11")
DlightBench.register_bench_workload(fused_matmul7_add5, "stable_diffusion_xl", "fused_matmul7_add5")
DlightBench.register_bench_workload(fused_conv2d10_add20_add21, "stable_diffusion_xl", "fused_conv2d10_add20_add21")
DlightBench.register_bench_workload(fused_matmul17_add18, "stable_diffusion_xl", "fused_matmul17_add18")
DlightBench.register_bench_workload(fused_conv2d1_add7_add10_divide, "stable_diffusion_xl", "fused_conv2d1_add7_add10_divide")
DlightBench.register_bench_workload(fused_matmul8_add5_silu, "stable_diffusion_xl", "fused_matmul8_add5_silu")
DlightBench.register_bench_workload(fused_reshape14_strided_slice4_reshape15_cast5_multiply3_multiply4_tir_sin1_tir_cos1_concatenate2_strided_slice5_reshape16_strided_slice6_reshape16_concatenate2_reshape17_concatenate3, "stable_diffusion_xl", "fused_reshape14_strided_slice4_reshape15_cast5_multiply3_multiply4_tir_sin1_tir_cos1_concatenate2_strided_slice5_reshape16_strided_slice6_reshape16_concatenate2_reshape17_concatenate3")
DlightBench.register_bench_workload(fused_reshape2_reshape2_add, "stable_diffusion_xl", "fused_reshape2_reshape2_add")
DlightBench.register_bench_workload(fused_matmul9_add8_cast4, "stable_diffusion_xl", "fused_matmul9_add8_cast4")
DlightBench.register_bench_workload(fused_matmul3_add4_gelu, "stable_diffusion_xl", "fused_matmul3_add4_gelu")
DlightBench.register_bench_workload(fused_broadcast_to1_strided_slice1_reshape12_cast3_multiply1_multiply2_tir_sin_tir_cos_concatenate1_strided_slice2_reshape13_strided_slice3_reshape13_concatenate1_cast4, "stable_diffusion_xl", "fused_broadcast_to1_strided_slice1_reshape12_cast3_multiply1_multiply2_tir_sin_tir_cos_concatenate1_strided_slice2_reshape13_strided_slice3_reshape13_concatenate1_cast4")
DlightBench.register_bench_workload(fused_group_norm9_silu8, "stable_diffusion_xl", "fused_group_norm9_silu8")
DlightBench.register_bench_workload(power1, "stable_diffusion_xl", "power1")
DlightBench.register_bench_workload(fused_conv2d18_add12, "stable_diffusion_xl", "fused_conv2d18_add12")
DlightBench.register_bench_workload(layer_norm3, "stable_diffusion_xl", "layer_norm3")
DlightBench.register_bench_workload(add48, "stable_diffusion_xl", "add48")
DlightBench.register_bench_workload(argmax, "stable_diffusion_xl", "argmax")
DlightBench.register_bench_workload(concatenate13, "stable_diffusion_xl", "concatenate13")
DlightBench.register_bench_workload(add29, "stable_diffusion_xl", "add29")
DlightBench.register_bench_workload(fused_matmul4_add2_add, "stable_diffusion_xl", "fused_matmul4_add2_add")
DlightBench.register_bench_workload(multiply18, "stable_diffusion_xl", "multiply18")
DlightBench.register_bench_workload(concatenate12, "stable_diffusion_xl", "concatenate12")
DlightBench.register_bench_workload(fused_matmul23_multiply9, "stable_diffusion_xl", "fused_matmul23_multiply9")
DlightBench.register_bench_workload(fused_conv2d3_add12_add14, "stable_diffusion_xl", "fused_conv2d3_add12_add14")
DlightBench.register_bench_workload(subtract1, "stable_diffusion_xl", "subtract1")
DlightBench.register_bench_workload(fused_strided_slice_reshape11, "stable_diffusion_xl", "fused_strided_slice_reshape11")
DlightBench.register_bench_workload(fused_matmul_add2_add, "stable_diffusion_xl", "fused_matmul_add2_add")
DlightBench.register_bench_workload(fused_reshape9_transpose3_reshape10, "stable_diffusion_xl", "fused_reshape9_transpose3_reshape10")
DlightBench.register_bench_workload(fused_reshape44_transpose38_reshape45_transpose39, "stable_diffusion_xl", "fused_reshape44_transpose38_reshape45_transpose39")
DlightBench.register_bench_workload(fused_conv2d39_add42_divide10_add43_tir_clip, "stable_diffusion_xl", "fused_conv2d39_add42_divide10_add43_tir_clip")
DlightBench.register_bench_workload(softmax6, "stable_diffusion_xl", "softmax6")
DlightBench.register_bench_workload(fused_conv2d_add7, "stable_diffusion_xl", "fused_conv2d_add7")
DlightBench.register_bench_workload(fused_reshape33_transpose29_add22, "stable_diffusion_xl", "fused_reshape33_transpose29_add22")
DlightBench.register_bench_workload(matmul2, "stable_diffusion_xl", "matmul2")
DlightBench.register_bench_workload(fused_matmul19_add23_divide5_add24, "stable_diffusion_xl", "fused_matmul19_add23_divide5_add24")
DlightBench.register_bench_workload(fused_matmul30_add45, "stable_diffusion_xl", "fused_matmul30_add45")
DlightBench.register_bench_workload(fused_conv2d2_add11, "stable_diffusion_xl", "fused_conv2d2_add11")
DlightBench.register_bench_workload(tir_image_to_rgba, "stable_diffusion_xl", "tir_image_to_rgba")
DlightBench.register_bench_workload(fused_matmul18_add16_add17, "stable_diffusion_xl", "fused_matmul18_add16_add17")
DlightBench.register_bench_workload(fused_reshape7_reshape8, "stable_diffusion_xl", "fused_reshape7_reshape8")
DlightBench.register_bench_workload(fused_conv2d24_add7_add9, "stable_diffusion_xl", "fused_conv2d24_add7_add9")
DlightBench.register_bench_workload(fused_conv2d22_add7_add9, "stable_diffusion_xl", "fused_conv2d22_add7_add9")
DlightBench.register_bench_workload(softmax, "stable_diffusion_xl", "softmax")
DlightBench.register_bench_workload(fused_reshape33_transpose29_add22_concatenate4, "stable_diffusion_xl", "fused_reshape33_transpose29_add22_concatenate4")
DlightBench.register_bench_workload(cast, "stable_diffusion_xl", "cast")
DlightBench.register_bench_workload(matmul14, "stable_diffusion_xl", "matmul14")
DlightBench.register_bench_workload(squeeze, "stable_diffusion_xl", "squeeze")
DlightBench.register_bench_workload(fused_reshape7_add3_reshape8, "stable_diffusion_xl", "fused_reshape7_add3_reshape8")
DlightBench.register_bench_workload(fused_conv2d12_add20_add21, "stable_diffusion_xl", "fused_conv2d12_add20_add21")
DlightBench.register_bench_workload(fused_conv2d5_add12, "stable_diffusion_xl", "fused_conv2d5_add12")
DlightBench.register_bench_workload(matmul1, "stable_diffusion_xl", "matmul1")
DlightBench.register_bench_workload(fused_reshape5_transpose1_reshape6, "stable_diffusion_xl", "fused_reshape5_transpose1_reshape6")
DlightBench.register_bench_workload(fused_conv2d11_add20, "stable_diffusion_xl", "fused_conv2d11_add20")
DlightBench.register_bench_workload(matmul31, "stable_diffusion_xl", "matmul31")
DlightBench.register_bench_workload(fused_reshape5_transpose1_reshape6_transpose2, "stable_diffusion_xl", "fused_reshape5_transpose1_reshape6_transpose2")
DlightBench.register_bench_workload(fused_matmul7_add5_strided_slice8, "stable_diffusion_xl", "fused_matmul7_add5_strided_slice8")
DlightBench.register_bench_workload(layer_norm1, "stable_diffusion_xl", "layer_norm1")
DlightBench.register_bench_workload(fused_matmul_add2, "stable_diffusion_xl", "fused_matmul_add2")
DlightBench.register_bench_workload(fused_matmul_add2_multiply, "stable_diffusion_xl", "fused_matmul_add2_multiply")
DlightBench.register_bench_workload(resize2d3, "stable_diffusion_xl", "resize2d3")
DlightBench.register_bench_workload(fused_conv2d28_add31, "stable_diffusion_xl", "fused_conv2d28_add31")
DlightBench.register_bench_workload(take, "stable_diffusion_xl", "take")
DlightBench.register_bench_workload(matmul5, "stable_diffusion_xl", "matmul5")
DlightBench.register_bench_workload(fused_reshape33_transpose29_add22_resize2d, "stable_diffusion_xl", "fused_reshape33_transpose29_add22_resize2d")
DlightBench.register_bench_workload(fused_matmul34_add45_add44, "stable_diffusion_xl", "fused_matmul34_add45_add44")
DlightBench.register_bench_workload(fused_conv2d4_add12_add14, "stable_diffusion_xl", "fused_conv2d4_add12_add14")
DlightBench.register_bench_workload(fused_cast_reshape1, "stable_diffusion_xl", "fused_cast_reshape1")
DlightBench.register_bench_workload(fused_matmul12_multiply5, "stable_diffusion_xl", "fused_matmul12_multiply5")
DlightBench.register_bench_workload(fused_reshape48_transpose40_reshape49, "stable_diffusion_xl", "fused_reshape48_transpose40_reshape49")
DlightBench.register_bench_workload(fused_matmul33_add47_multiply16_tir_sigmoid_multiply17, "stable_diffusion_xl", "fused_matmul33_add47_multiply16_tir_sigmoid_multiply17")
DlightBench.register_bench_workload(fused_matmul11_add16_divide3_add17, "stable_diffusion_xl", "fused_matmul11_add16_divide3_add17")
DlightBench.register_bench_workload(layer_norm, "stable_diffusion_xl", "layer_norm")
DlightBench.register_bench_workload(fused_matmul30_add45_add44, "stable_diffusion_xl", "fused_matmul30_add45_add44")
DlightBench.register_bench_workload(fused_group_norm16_silu14, "stable_diffusion_xl", "fused_group_norm16_silu14")
DlightBench.register_bench_workload(resize2d2, "stable_diffusion_xl", "resize2d2")
DlightBench.register_bench_workload(concatenate11, "stable_diffusion_xl", "concatenate11")
DlightBench.register_bench_workload(fused_reshape44_transpose38_reshape45, "stable_diffusion_xl", "fused_reshape44_transpose38_reshape45")
DlightBench.register_bench_workload(fused_matmul30_add45_multiply15, "stable_diffusion_xl", "fused_matmul30_add45_multiply15")
DlightBench.register_bench_workload(fused_conv2d1_add7_add9, "stable_diffusion_xl", "fused_conv2d1_add7_add9")
DlightBench.register_bench_workload(fused_group_norm17_silu15, "stable_diffusion_xl", "fused_group_norm17_silu15")
DlightBench.register_bench_workload(fused_group_norm18_silu16, "stable_diffusion_xl", "fused_group_norm18_silu16")
DlightBench.register_bench_workload(fused_reshape43_reshape43_add44, "stable_diffusion_xl", "fused_reshape43_reshape43_add44")
DlightBench.register_bench_workload(reshape21, "stable_diffusion_xl", "reshape21")
DlightBench.register_bench_workload(take3, "stable_diffusion_xl", "take3")
DlightBench.register_bench_workload(fused_conv2d9_add20, "stable_diffusion_xl", "fused_conv2d9_add20")
DlightBench.register_bench_workload(fused_reshape26_transpose20_add15, "stable_diffusion_xl", "fused_reshape26_transpose20_add15")
DlightBench.register_bench_workload(fused_group_norm_silu1, "stable_diffusion_xl", "fused_group_norm_silu1")
DlightBench.register_bench_workload(fused_reshape_cast_reshape1, "stable_diffusion_xl", "fused_reshape_cast_reshape1")
DlightBench.register_bench_workload(resize2d4, "stable_diffusion_xl", "resize2d4")
DlightBench.register_bench_workload(fused_reshape46_add46_reshape47, "stable_diffusion_xl", "fused_reshape46_add46_reshape47")
DlightBench.register_bench_workload(matmul32, "stable_diffusion_xl", "matmul32")
DlightBench.register_bench_workload(fused_conv2d27_add30, "stable_diffusion_xl", "fused_conv2d27_add30")
DlightBench.register_bench_workload(fused_matmul19_add23, "stable_diffusion_xl", "fused_matmul19_add23")
DlightBench.register_bench_workload(fused_conv2d29_add31_add32_divide6_divide6, "stable_diffusion_xl", "fused_conv2d29_add31_add32_divide6_divide6")
DlightBench.register_bench_workload(fused_conv2d25_add7, "stable_diffusion_xl", "fused_conv2d25_add7")
DlightBench.register_bench_workload(reshape, "stable_diffusion_xl", "reshape")
DlightBench.register_bench_workload(softmax5, "stable_diffusion_xl", "softmax5")
DlightBench.register_bench_workload(fused_reshape26_transpose20_add15_resize2d1, "stable_diffusion_xl", "fused_reshape26_transpose20_add15_resize2d1")
DlightBench.register_bench_workload(fused_group_norm19_silu17, "stable_diffusion_xl", "fused_group_norm19_silu17")
DlightBench.register_bench_workload(matmul24, "stable_diffusion_xl", "matmul24")
DlightBench.register_bench_workload(softmax4, "stable_diffusion_xl", "softmax4")
DlightBench.register_bench_workload(fused_reshape37_transpose33_transpose34, "stable_diffusion_xl", "fused_reshape37_transpose33_transpose34")
DlightBench.register_bench_workload(fused_matmul6_add5_silu, "stable_diffusion_xl", "fused_matmul6_add5_silu")
DlightBench.register_bench_workload(fused_conv2d38_add40, "stable_diffusion_xl", "fused_conv2d38_add40")
DlightBench.register_bench_workload(divide11, "stable_diffusion_xl", "divide11")
DlightBench.register_bench_workload(matmul11, "stable_diffusion_xl", "matmul11")
DlightBench.register_bench_workload(fused_conv2d29_add31, "stable_diffusion_xl", "fused_conv2d29_add31")
DlightBench.register_bench_workload(fused_reshape37_transpose33, "stable_diffusion_xl", "fused_reshape37_transpose33")
DlightBench.register_bench_workload(fused_matmul27_add33, "stable_diffusion_xl", "fused_matmul27_add33")
DlightBench.register_bench_workload(group_norm15, "stable_diffusion_xl", "group_norm15")
DlightBench.register_bench_workload(silu, "stable_diffusion_xl", "silu")
DlightBench.register_bench_workload(power, "stable_diffusion_xl", "power")
DlightBench.register_bench_workload(fused_conv2d29_add31_add32_divide6, "stable_diffusion_xl", "fused_conv2d29_add31_add32_divide6")
DlightBench.register_bench_workload(fused_reshape36_transpose30_transpose31, "stable_diffusion_xl", "fused_reshape36_transpose30_transpose31")
DlightBench.register_bench_workload(fused_transpose31_reshape39_add32_divide6, "stable_diffusion_xl", "fused_transpose31_reshape39_add32_divide6")
DlightBench.register_bench_workload(reshape28, "stable_diffusion_xl", "reshape28")
DlightBench.register_bench_workload(fused_transpose36_multiply14_tir_round, "stable_diffusion_xl", "fused_transpose36_multiply14_tir_round")
DlightBench.register_bench_workload(concatenate, "stable_diffusion_xl", "concatenate")
DlightBench.register_bench_workload(layer_norm2, "stable_diffusion_xl", "layer_norm2")
DlightBench.register_bench_workload(fused_transpose35_reshape38, "stable_diffusion_xl", "fused_transpose35_reshape38")
DlightBench.register_bench_workload(fused_matmul20_multiply8, "stable_diffusion_xl", "fused_matmul20_multiply8")
DlightBench.register_bench_workload(take2, "stable_diffusion_xl", "take2")
DlightBench.register_bench_workload(multiply12, "stable_diffusion_xl", "multiply12")
DlightBench.register_bench_workload(fused_group_norm20_silu18, "stable_diffusion_xl", "fused_group_norm20_silu18")
DlightBench.register_bench_workload(fused_conv2d34_add37, "stable_diffusion_xl", "fused_conv2d34_add37")
DlightBench.register_bench_workload(fused_reshape32_transpose26, "stable_diffusion_xl", "fused_reshape32_transpose26")
DlightBench.register_bench_workload(transpose30, "stable_diffusion_xl", "transpose30")
DlightBench.register_bench_workload(fused_group_norm14_silu13, "stable_diffusion_xl", "fused_group_norm14_silu13")
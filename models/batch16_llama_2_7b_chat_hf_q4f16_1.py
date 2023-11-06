from dlight_bench import DlightBench
from tvm.script import tir as T

@T.prim_func
def divide(var_A: T.handle, var_B: T.handle, var_T_divide: T.handle):
    T.func_attr({"tir.noalias": T.bool(True)})
    A = T.match_buffer(var_A, (T.int64(16), T.int64(1), T.int64(32000)))
    B = T.match_buffer(var_B, (T.int64(16), T.int64(1), T.int64(1)))
    T_divide = T.match_buffer(var_T_divide, (T.int64(16), T.int64(1), T.int64(32000)))
    # with T.block("root"):
    for ax0, ax1, ax2 in T.grid(T.int64(16), T.int64(1), T.int64(32000)):
        with T.block("T_divide"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(A[v_ax0, v_ax1, v_ax2], B[v_ax0, v_ax1, T.int64(0)])
            T.writes(T_divide[v_ax0, v_ax1, v_ax2])
            T_divide[v_ax0, v_ax1, v_ax2] = A[v_ax0, v_ax1, v_ax2] / B[v_ax0, v_ax1, T.int64(0)]

@T.prim_func
def fused_fused_decode1_fused_NT_matmul1_add(lv3: T.Buffer((T.int64(4096), T.int64(512)), "uint32"), lv4: T.Buffer((T.int64(4096), T.int64(128)), "float16"), p_lv15: T.handle, p_inputs_embeds: T.handle, p_output0: T.handle):
    T.func_attr({"tir.noalias": T.bool(True)})
    n = T.int64()
    lv15 = T.match_buffer(p_lv15, (T.int64(1), n, T.int64(4096)), "float16")
    inputs_embeds = T.match_buffer(p_inputs_embeds, (T.int64(1), n, T.int64(4096)), "float16")
    p_output0_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(4096)), "float16")
    # with T.block("root"):
    p_output0_intermediate_1 = T.alloc_buffer((T.int64(4096), T.int64(4096)), "float16")
    var_NT_matmul_intermediate = T.alloc_buffer((T.int64(1), n, T.int64(4096)), "float16")
    for i, j in T.grid(T.int64(4096), T.int64(4096)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv3[v_i, v_j // T.int64(8)], lv4[v_i, v_j // T.int64(32)])
            T.writes(p_output0_intermediate_1[v_i, v_j])
            p_output0_intermediate_1[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv3[v_i, v_j // T.int64(8)], T.Cast("uint32", v_j % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv4[v_i, v_j // T.int64(32)]
    for i0, i1, i2, k in T.grid(T.int64(1), n, T.int64(4096), T.int64(4096)):
        with T.block("NT_matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv15[v_i0, v_i1, v_k], p_output0_intermediate_1[v_i2, v_k])
            T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2] + lv15[v_i0, v_i1, v_k] * p_output0_intermediate_1[v_i2, v_k]
    for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(4096)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(inputs_embeds[v_ax0, v_ax1, v_ax2], var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2])
            T.writes(p_output0_intermediate[v_ax0, v_ax1, v_ax2])
            p_output0_intermediate[v_ax0, v_ax1, v_ax2] = inputs_embeds[v_ax0, v_ax1, v_ax2] + var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2]

@T.prim_func
def fused_fused_decode1_fused_NT_matmul6_add1(lv490: T.Buffer((T.int64(4096), T.int64(512)), "uint32"), lv491: T.Buffer((T.int64(4096), T.int64(128)), "float16"), p_lv884: T.handle, p_inputs_embeds1: T.handle, p_output0: T.handle):
    T.func_attr({"tir.noalias": T.bool(True)})
    lv884 = T.match_buffer(p_lv884, (T.int64(16), T.int64(1), T.int64(4096)), "float16")
    inputs_embeds1 = T.match_buffer(p_inputs_embeds1, (T.int64(16), T.int64(1), T.int64(4096)), "float16")
    p_output0_intermediate = T.match_buffer(p_output0, (T.int64(16), T.int64(1), T.int64(4096)), "float16")
    # with T.block("root"):
    p_output0_intermediate_1 = T.alloc_buffer((T.int64(4096), T.int64(4096)), "float16")
    var_NT_matmul_intermediate = T.alloc_buffer((T.int64(16), T.int64(1), T.int64(4096)), "float16")
    for i, j in T.grid(T.int64(4096), T.int64(4096)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv490[v_i, v_j // T.int64(8)], lv491[v_i, v_j // T.int64(32)])
            T.writes(p_output0_intermediate_1[v_i, v_j])
            p_output0_intermediate_1[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv490[v_i, v_j // T.int64(8)], T.Cast("uint32", v_j % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv491[v_i, v_j // T.int64(32)]
    for i0, i1, i2, k in T.grid(T.int64(16), T.int64(1), T.int64(4096), T.int64(4096)):
        with T.block("NT_matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv884[v_i0, v_i1, v_k], p_output0_intermediate_1[v_i2, v_k])
            T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2] + lv884[v_i0, v_i1, v_k] * p_output0_intermediate_1[v_i2, v_k]
    for ax0, ax1, ax2 in T.grid(T.int64(16), T.int64(1), T.int64(4096)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(inputs_embeds1[v_ax0, v_ax1, v_ax2], var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2])
            T.writes(p_output0_intermediate[v_ax0, v_ax1, v_ax2])
            p_output0_intermediate[v_ax0, v_ax1, v_ax2] = inputs_embeds1[v_ax0, v_ax1, v_ax2] + var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2]

@T.prim_func
def fused_fused_decode2_NT_matmul2(lv7: T.Buffer((T.int64(22016), T.int64(512)), "uint32"), lv8: T.Buffer((T.int64(22016), T.int64(128)), "float16"), p_lv19: T.handle, p_output0: T.handle):
    T.func_attr({"tir.noalias": T.bool(True)})
    n = T.int64()
    lv19 = T.match_buffer(p_lv19, (T.int64(1), n, T.int64(4096)), "float16")
    var_NT_matmul_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(22016)), "float16")
    # with T.block("root"):
    p_output0_intermediate = T.alloc_buffer((T.int64(22016), T.int64(4096)), "float16")
    for i, j in T.grid(T.int64(22016), T.int64(4096)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv7[v_i, v_j // T.int64(8)], lv8[v_i, v_j // T.int64(32)])
            T.writes(p_output0_intermediate[v_i, v_j])
            p_output0_intermediate[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv7[v_i, v_j // T.int64(8)], T.Cast("uint32", v_j % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv8[v_i, v_j // T.int64(32)]
    for i0, i1, i2, k in T.grid(T.int64(1), n, T.int64(22016), T.int64(4096)):
        with T.block("NT_matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv19[v_i0, v_i1, v_k], p_output0_intermediate[v_i2, v_k])
            T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2] + lv19[v_i0, v_i1, v_k] * p_output0_intermediate[v_i2, v_k]

@T.prim_func
def fused_fused_decode2_NT_matmul7(lv494: T.Buffer((T.int64(22016), T.int64(512)), "uint32"), lv495: T.Buffer((T.int64(22016), T.int64(128)), "float16"), p_lv888: T.handle, p_output0: T.handle):
    T.func_attr({"tir.noalias": T.bool(True)})
    lv888 = T.match_buffer(p_lv888, (T.int64(16), T.int64(1), T.int64(4096)), "float16")
    var_NT_matmul_intermediate = T.match_buffer(p_output0, (T.int64(16), T.int64(1), T.int64(22016)), "float16")
    # with T.block("root"):
    p_output0_intermediate = T.alloc_buffer((T.int64(22016), T.int64(4096)), "float16")
    for i, j in T.grid(T.int64(22016), T.int64(4096)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv494[v_i, v_j // T.int64(8)], lv495[v_i, v_j // T.int64(32)])
            T.writes(p_output0_intermediate[v_i, v_j])
            p_output0_intermediate[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv494[v_i, v_j // T.int64(8)], T.Cast("uint32", v_j % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv495[v_i, v_j // T.int64(32)]
    for i0, i1, i2, k in T.grid(T.int64(16), T.int64(1), T.int64(22016), T.int64(4096)):
        with T.block("NT_matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv888[v_i0, v_i1, v_k], p_output0_intermediate[v_i2, v_k])
            T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2] + lv888[v_i0, v_i1, v_k] * p_output0_intermediate[v_i2, v_k]

@T.prim_func
def fused_fused_decode3_fused_NT_matmul3_add(lv11: T.Buffer((T.int64(4096), T.int64(1376)), "uint32"), lv12: T.Buffer((T.int64(4096), T.int64(344)), "float16"), p_lv10: T.handle, p_lv6: T.handle, p_output0: T.handle):
    T.func_attr({"tir.noalias": T.bool(True)})
    n = T.int64()
    lv10 = T.match_buffer(p_lv10, (T.int64(1), n, T.int64(11008)), "float16")
    lv6 = T.match_buffer(p_lv6, (T.int64(1), n, T.int64(4096)), "float16")
    p_output0_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(4096)), "float16")
    # with T.block("root"):
    p_output0_intermediate_1 = T.alloc_buffer((T.int64(4096), T.int64(11008)), "float16")
    var_NT_matmul_intermediate = T.alloc_buffer((T.int64(1), n, T.int64(4096)), "float16")
    for i, j in T.grid(T.int64(4096), T.int64(11008)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv11[v_i, v_j // T.int64(8)], lv12[v_i, v_j // T.int64(32)])
            T.writes(p_output0_intermediate_1[v_i, v_j])
            p_output0_intermediate_1[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv11[v_i, v_j // T.int64(8)], T.Cast("uint32", v_j % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv12[v_i, v_j // T.int64(32)]
    for i0, i1, i2, k in T.grid(T.int64(1), n, T.int64(4096), T.int64(11008)):
        with T.block("NT_matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv10[v_i0, v_i1, v_k], p_output0_intermediate_1[v_i2, v_k])
            T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2] + lv10[v_i0, v_i1, v_k] * p_output0_intermediate_1[v_i2, v_k]
    for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(4096)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(lv6[v_ax0, v_ax1, v_ax2], var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2])
            T.writes(p_output0_intermediate[v_ax0, v_ax1, v_ax2])
            p_output0_intermediate[v_ax0, v_ax1, v_ax2] = lv6[v_ax0, v_ax1, v_ax2] + var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2]

@T.prim_func
def fused_fused_decode3_fused_NT_matmul8_add1(lv498: T.Buffer((T.int64(4096), T.int64(1376)), "uint32"), lv499: T.Buffer((T.int64(4096), T.int64(344)), "float16"), p_lv497: T.handle, p_lv493: T.handle, p_output0: T.handle):
    T.func_attr({"tir.noalias": T.bool(True)})
    lv497 = T.match_buffer(p_lv497, (T.int64(16), T.int64(1), T.int64(11008)), "float16")
    lv493 = T.match_buffer(p_lv493, (T.int64(16), T.int64(1), T.int64(4096)), "float16")
    p_output0_intermediate = T.match_buffer(p_output0, (T.int64(16), T.int64(1), T.int64(4096)), "float16")
    # with T.block("root"):
    p_output0_intermediate_1 = T.alloc_buffer((T.int64(4096), T.int64(11008)), "float16")
    var_NT_matmul_intermediate = T.alloc_buffer((T.int64(16), T.int64(1), T.int64(4096)), "float16")
    for i, j in T.grid(T.int64(4096), T.int64(11008)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv498[v_i, v_j // T.int64(8)], lv499[v_i, v_j // T.int64(32)])
            T.writes(p_output0_intermediate_1[v_i, v_j])
            p_output0_intermediate_1[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv498[v_i, v_j // T.int64(8)], T.Cast("uint32", v_j % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv499[v_i, v_j // T.int64(32)]
    for i0, i1, i2, k in T.grid(T.int64(16), T.int64(1), T.int64(4096), T.int64(11008)):
        with T.block("NT_matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv497[v_i0, v_i1, v_k], p_output0_intermediate_1[v_i2, v_k])
            T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2] + lv497[v_i0, v_i1, v_k] * p_output0_intermediate_1[v_i2, v_k]
    for ax0, ax1, ax2 in T.grid(T.int64(16), T.int64(1), T.int64(4096)):
        with T.block("T_add"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(lv493[v_ax0, v_ax1, v_ax2], var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2])
            T.writes(p_output0_intermediate[v_ax0, v_ax1, v_ax2])
            p_output0_intermediate[v_ax0, v_ax1, v_ax2] = lv493[v_ax0, v_ax1, v_ax2] + var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2]

@T.prim_func
def fused_fused_decode4_fused_NT_matmul4_cast(p_lv480: T.handle, p_lv481: T.handle, lv868: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16"), p_output0: T.handle):
    T.func_attr({"tir.noalias": T.bool(True)})
    lv480 = T.match_buffer(p_lv480, (T.int64(32000), T.int64(512)), "uint32")
    lv481 = T.match_buffer(p_lv481, (T.int64(32000), T.int64(128)), "float16")
    p_output0_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(1), T.int64(32000)))
    # with T.block("root"):
    p_output0_intermediate_1 = T.alloc_buffer((T.int64(32000), T.int64(4096)), "float16")
    var_NT_matmul_intermediate = T.alloc_buffer((T.int64(1), T.int64(1), T.int64(32000)), "float16")
    for i, j in T.grid(T.int64(32000), T.int64(4096)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv480[v_i, v_j // T.int64(8)], lv481[v_i, v_j // T.int64(32)])
            T.writes(p_output0_intermediate_1[v_i, v_j])
            p_output0_intermediate_1[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv480[v_i, v_j // T.int64(8)], T.Cast("uint32", v_j % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv481[v_i, v_j // T.int64(32)]
    for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), T.int64(32000), T.int64(4096)):
        with T.block("NT_matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv868[v_i0, v_i1, v_k], p_output0_intermediate_1[v_i2, v_k])
            T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2] + lv868[v_i0, v_i1, v_k] * p_output0_intermediate_1[v_i2, v_k]
    for i0, i1, i2 in T.grid(T.int64(1), T.int64(1), T.int64(32000)):
        with T.block("compute"):
            v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
            T.reads(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
            T.writes(p_output0_intermediate[v_i0, v_i1, v_i2])
            p_output0_intermediate[v_i0, v_i1, v_i2] = T.Cast("float32", var_NT_matmul_intermediate[v_i0, v_i1, v_i2])

@T.prim_func
def fused_fused_decode4_fused_NT_matmul9_cast1(p_lv967: T.handle, p_lv968: T.handle, p_lv1737: T.handle, p_output0: T.handle):
    T.func_attr({"tir.noalias": T.bool(True)})
    lv967 = T.match_buffer(p_lv967, (T.int64(32000), T.int64(512)), "uint32")
    lv968 = T.match_buffer(p_lv968, (T.int64(32000), T.int64(128)), "float16")
    lv1737 = T.match_buffer(p_lv1737, (T.int64(16), T.int64(1), T.int64(4096)), "float16")
    p_output0_intermediate = T.match_buffer(p_output0, (T.int64(16), T.int64(1), T.int64(32000)))
    # with T.block("root"):
    p_output0_intermediate_1 = T.alloc_buffer((T.int64(32000), T.int64(4096)), "float16")
    var_NT_matmul_intermediate = T.alloc_buffer((T.int64(16), T.int64(1), T.int64(32000)), "float16")
    for i, j in T.grid(T.int64(32000), T.int64(4096)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv967[v_i, v_j // T.int64(8)], lv968[v_i, v_j // T.int64(32)])
            T.writes(p_output0_intermediate_1[v_i, v_j])
            p_output0_intermediate_1[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv967[v_i, v_j // T.int64(8)], T.Cast("uint32", v_j % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv968[v_i, v_j // T.int64(32)]
    for i0, i1, i2, k in T.grid(T.int64(16), T.int64(1), T.int64(32000), T.int64(4096)):
        with T.block("NT_matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv1737[v_i0, v_i1, v_k], p_output0_intermediate_1[v_i2, v_k])
            T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2] + lv1737[v_i0, v_i1, v_k] * p_output0_intermediate_1[v_i2, v_k]
    for i0, i1, i2 in T.grid(T.int64(16), T.int64(1), T.int64(32000)):
        with T.block("compute"):
            v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
            T.reads(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
            T.writes(p_output0_intermediate[v_i0, v_i1, v_i2])
            p_output0_intermediate[v_i0, v_i1, v_i2] = T.Cast("float32", var_NT_matmul_intermediate[v_i0, v_i1, v_i2])

# @T.prim_func
# def fused_fused_decode4_take(p_lv484: T.handle, p_lv485: T.handle, p_lv: T.handle, p_output0: T.handle, n: T.int32, nseq: T.int32):
#     T.func_attr({"tir.noalias": T.bool(True)})
#     lv484 = T.match_buffer(p_lv484, (T.int32(32000), 512), "uint32")
#     lv485 = T.match_buffer(p_lv485, (T.int32(32000), 128), "float16")
#     lv = T.match_buffer(p_lv, (T.int32(16) * n,), "int32")
#     var_T_take_intermediate = T.match_buffer(p_output0, (T.int32(16) * n, 4096), "float16")
#     # with T.block("root"):
#     for ax0, ax1 in T.grid(T.int32(16) * n, 4096):
#         with T.block("T_take"):
#             v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
#             T.reads(lv484[lv[v_ax0], v_ax1 // 8], lv[v_ax0], lv485[lv[v_ax0], v_ax1 // 32])
#             T.writes(var_T_take_intermediate[v_ax0, v_ax1])
#             var_T_take_intermediate[v_ax0, v_ax1] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv484[lv[v_ax0], v_ax1 // 8], T.Cast("uint32", v_ax1 % 8) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv485[lv[v_ax0], v_ax1 // 32]

@T.prim_func
def fused_fused_decode_NT_matmul(lv: T.Buffer((T.int64(12288), T.int64(512)), "uint32"), lv1: T.Buffer((T.int64(12288), T.int64(128)), "float16"), p_lv3: T.handle, p_output0: T.handle):
    T.func_attr({"tir.noalias": T.bool(True)})
    n = T.int64()
    lv3 = T.match_buffer(p_lv3, (T.int64(1), n, T.int64(4096)), "float16")
    var_NT_matmul_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(12288)), "float16")
    # with T.block("root"):
    p_output0_intermediate = T.alloc_buffer((T.int64(12288), T.int64(4096)), "float16")
    for i, j in T.grid(T.int64(12288), T.int64(4096)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv[v_i, v_j // T.int64(8)], lv1[v_i, v_j // T.int64(32)])
            T.writes(p_output0_intermediate[v_i, v_j])
            p_output0_intermediate[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv[v_i, v_j // T.int64(8)], T.Cast("uint32", v_j % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv1[v_i, v_j // T.int64(32)]
    for i0, i1, i2, k in T.grid(T.int64(1), n, T.int64(12288), T.int64(4096)):
        with T.block("NT_matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv3[v_i0, v_i1, v_k], p_output0_intermediate[v_i2, v_k])
            T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2] + lv3[v_i0, v_i1, v_k] * p_output0_intermediate[v_i2, v_k]

@T.prim_func
def fused_fused_decode_NT_matmul5(lv487: T.Buffer((T.int64(12288), T.int64(512)), "uint32"), lv488: T.Buffer((T.int64(12288), T.int64(128)), "float16"), p_lv872: T.handle, p_output0: T.handle):
    T.func_attr({"tir.noalias": T.bool(True)})
    lv872 = T.match_buffer(p_lv872, (T.int64(16), T.int64(1), T.int64(4096)), "float16")
    var_NT_matmul_intermediate = T.match_buffer(p_output0, (T.int64(16), T.int64(1), T.int64(12288)), "float16")
    # with T.block("root"):
    p_output0_intermediate = T.alloc_buffer((T.int64(12288), T.int64(4096)), "float16")
    for i, j in T.grid(T.int64(12288), T.int64(4096)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(lv487[v_i, v_j // T.int64(8)], lv488[v_i, v_j // T.int64(32)])
            T.writes(p_output0_intermediate[v_i, v_j])
            p_output0_intermediate[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv487[v_i, v_j // T.int64(8)], T.Cast("uint32", v_j % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv488[v_i, v_j // T.int64(32)]
    for i0, i1, i2, k in T.grid(T.int64(16), T.int64(1), T.int64(12288), T.int64(4096)):
        with T.block("NT_matmul"):
            v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
            T.reads(lv872[v_i0, v_i1, v_k], p_output0_intermediate[v_i2, v_k])
            T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
            with T.init():
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
            var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2] + lv872[v_i0, v_i1, v_k] * p_output0_intermediate[v_i2, v_k]

@T.prim_func
def fused_split1_silu_multiply(p_lv2: T.handle, p_output0: T.handle):
    T.func_attr({"tir.noalias": T.bool(True)})
    n = T.int64()
    lv2 = T.match_buffer(p_lv2, (T.int64(1), n, T.int64(22016)), "float16")
    var_T_multiply_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(11008)), "float16")
    # with T.block("root"):
    var_T_split_sections_intermediate = T.alloc_buffer((T.int64(1), n, T.int64(11008)), "float16")
    var_T_split_sections_intermediate_1 = T.alloc_buffer((T.int64(1), n, T.int64(11008)), "float16")
    compute = T.alloc_buffer((T.int64(1), n, T.int64(11008)), "float16")
    var_T_multiply_intermediate_1 = T.alloc_buffer((T.int64(1), n, T.int64(11008)), "float16")
    for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(11008)):
        with T.block("T_split_sections"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(lv2[v_ax0, v_ax1, v_ax2])
            T.writes(var_T_split_sections_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_split_sections_intermediate[v_ax0, v_ax1, v_ax2] = lv2[v_ax0, v_ax1, v_ax2]
    for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(11008)):
        with T.block("T_split_sections_1"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(lv2[v_ax0, v_ax1, v_ax2 + T.int64(11008)])
            T.writes(var_T_split_sections_intermediate_1[v_ax0, v_ax1, v_ax2])
            var_T_split_sections_intermediate_1[v_ax0, v_ax1, v_ax2] = lv2[v_ax0, v_ax1, v_ax2 + T.int64(11008)]
    for i0, i1, i2 in T.grid(T.int64(1), n, T.int64(11008)):
        with T.block("compute"):
            v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
            T.reads(var_T_split_sections_intermediate[v_i0, v_i1, v_i2])
            T.writes(compute[v_i0, v_i1, v_i2])
            compute[v_i0, v_i1, v_i2] = T.sigmoid(var_T_split_sections_intermediate[v_i0, v_i1, v_i2])
    for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(11008)):
        with T.block("T_multiply"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_T_split_sections_intermediate[v_ax0, v_ax1, v_ax2], compute[v_ax0, v_ax1, v_ax2])
            T.writes(var_T_multiply_intermediate_1[v_ax0, v_ax1, v_ax2])
            var_T_multiply_intermediate_1[v_ax0, v_ax1, v_ax2] = var_T_split_sections_intermediate[v_ax0, v_ax1, v_ax2] * compute[v_ax0, v_ax1, v_ax2]
    for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(11008)):
        with T.block("T_multiply_1"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_T_multiply_intermediate_1[v_ax0, v_ax1, v_ax2], var_T_split_sections_intermediate_1[v_ax0, v_ax1, v_ax2])
            T.writes(var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2] = var_T_multiply_intermediate_1[v_ax0, v_ax1, v_ax2] * var_T_split_sections_intermediate_1[v_ax0, v_ax1, v_ax2]

@T.prim_func
def fused_split3_silu1_multiply1(p_lv131: T.handle, p_output0: T.handle):
    T.func_attr({"tir.noalias": T.bool(True)})
    lv131 = T.match_buffer(p_lv131, (T.int64(16), T.int64(1), T.int64(22016)), "float16")
    var_T_multiply_intermediate = T.match_buffer(p_output0, (T.int64(16), T.int64(1), T.int64(11008)), "float16")
    # with T.block("root"):
    var_T_split_sections_intermediate = T.alloc_buffer((T.int64(16), T.int64(1), T.int64(11008)), "float16")
    var_T_split_sections_intermediate_1 = T.alloc_buffer((T.int64(16), T.int64(1), T.int64(11008)), "float16")
    compute = T.alloc_buffer((T.int64(16), T.int64(1), T.int64(11008)), "float16")
    var_T_multiply_intermediate_1 = T.alloc_buffer((T.int64(16), T.int64(1), T.int64(11008)), "float16")
    for ax0, ax1, ax2 in T.grid(T.int64(16), T.int64(1), T.int64(11008)):
        with T.block("T_split_sections"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(lv131[v_ax0, v_ax1, v_ax2])
            T.writes(var_T_split_sections_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_split_sections_intermediate[v_ax0, v_ax1, v_ax2] = lv131[v_ax0, v_ax1, v_ax2]
    for ax0, ax1, ax2 in T.grid(T.int64(16), T.int64(1), T.int64(11008)):
        with T.block("T_split_sections_1"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(lv131[v_ax0, v_ax1, v_ax2 + T.int64(11008)])
            T.writes(var_T_split_sections_intermediate_1[v_ax0, v_ax1, v_ax2])
            var_T_split_sections_intermediate_1[v_ax0, v_ax1, v_ax2] = lv131[v_ax0, v_ax1, v_ax2 + T.int64(11008)]
    for i0, i1, i2 in T.grid(T.int64(16), T.int64(1), T.int64(11008)):
        with T.block("compute"):
            v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
            T.reads(var_T_split_sections_intermediate[v_i0, v_i1, v_i2])
            T.writes(compute[v_i0, v_i1, v_i2])
            compute[v_i0, v_i1, v_i2] = T.sigmoid(var_T_split_sections_intermediate[v_i0, v_i1, v_i2])
    for ax0, ax1, ax2 in T.grid(T.int64(16), T.int64(1), T.int64(11008)):
        with T.block("T_multiply"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_T_split_sections_intermediate[v_ax0, v_ax1, v_ax2], compute[v_ax0, v_ax1, v_ax2])
            T.writes(var_T_multiply_intermediate_1[v_ax0, v_ax1, v_ax2])
            var_T_multiply_intermediate_1[v_ax0, v_ax1, v_ax2] = var_T_split_sections_intermediate[v_ax0, v_ax1, v_ax2] * compute[v_ax0, v_ax1, v_ax2]
    for ax0, ax1, ax2 in T.grid(T.int64(16), T.int64(1), T.int64(11008)):
        with T.block("T_multiply_1"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(var_T_multiply_intermediate_1[v_ax0, v_ax1, v_ax2], var_T_split_sections_intermediate_1[v_ax0, v_ax1, v_ax2])
            T.writes(var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2])
            var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2] = var_T_multiply_intermediate_1[v_ax0, v_ax1, v_ax2] * var_T_split_sections_intermediate_1[v_ax0, v_ax1, v_ax2]

@T.prim_func
def kv_cache_transpose_append(var_pages: T.handle, var_k_data: T.handle, var_v_data: T.handle, var_page_table_indptr: T.handle, var_page_table_values: T.handle, var_last_page_offset: T.handle, var_append_length_indptr: T.handle, var_pos2seqidx: T.handle, layer_id: T.int32):
    num_pages, nlayer, nhead, page_size, nfeat = T.int32(), T.int32(), T.int32(), T.int32(), T.int32()
    pages = T.match_buffer(var_pages, (num_pages, nlayer, 2, nhead, page_size, nfeat), "float16")
    ntoken = T.int32()
    k_data = T.match_buffer(var_k_data, (ntoken, nhead, nfeat), "float16")
    v_data = T.match_buffer(var_v_data, (ntoken, nhead, nfeat), "float16")
    page_table_indptr = T.match_buffer(var_page_table_indptr, (T.int32(16) + 1,), "int32")
    npage = T.int32()
    page_table_values = T.match_buffer(var_page_table_values, (npage,), "int32")
    last_page_offset = T.match_buffer(var_last_page_offset, (T.int32(16),), "int32")
    append_length_indptr = T.match_buffer(var_append_length_indptr, (T.int32(16) + 1,), "int32")
    pos2seqidx = T.match_buffer(var_pos2seqidx, (ntoken,), "int32")
    # with T.block("root"):
    for global_pos, h, f in T.grid(ntoken, nhead, nfeat):
        with T.block("k_transpose_append"):
            vgpos, vh, vf = T.axis.remap("SSS", [global_pos, h, f])
            seq_idx = T.int32()
            seqlen = T.int32()
            T.reads(pos2seqidx[vgpos], page_table_indptr[seq_idx:seq_idx + 2], last_page_offset[seq_idx], k_data[vgpos, vh, vf], page_table_values[page_table_indptr[seq_idx] + (seqlen - (append_length_indptr[seq_idx + 1] - vgpos)) // page_size], append_length_indptr[seq_idx + 1])
            T.writes(pages[page_table_values[page_table_indptr[seq_idx] + (seqlen - (append_length_indptr[seq_idx + 1] - vgpos)) // page_size], layer_id, 0, vh, (seqlen - (append_length_indptr[seq_idx + 1] - vgpos)) % page_size, vf])
            with T.LetStmt(pos2seqidx[vgpos], var=seq_idx):
                with T.LetStmt((page_table_indptr[seq_idx + 1] - page_table_indptr[seq_idx] - 1) * page_size + last_page_offset[seq_idx], var=seqlen):
                    pages[page_table_values[page_table_indptr[seq_idx] + (seqlen - (append_length_indptr[seq_idx + 1] - vgpos)) // page_size], layer_id, 0, vh, (seqlen - (append_length_indptr[seq_idx + 1] - vgpos)) % page_size, vf] = k_data[vgpos, vh, vf]
        with T.block("v_transpose_append"):
            vgpos, vh, vf = T.axis.remap("SSS", [global_pos, h, f])
            seq_idx = T.int32()
            seqlen = T.int32()
            T.reads(pos2seqidx[vgpos], page_table_indptr[seq_idx:seq_idx + 2], last_page_offset[seq_idx], v_data[vgpos, vh, vf], page_table_values[page_table_indptr[seq_idx] + (seqlen - (append_length_indptr[seq_idx + 1] - vgpos)) // page_size], append_length_indptr[seq_idx + 1])
            T.writes(pages[page_table_values[page_table_indptr[seq_idx] + (seqlen - (append_length_indptr[seq_idx + 1] - vgpos)) // page_size], layer_id, 1, vh, (seqlen - (append_length_indptr[seq_idx + 1] - vgpos)) % page_size, vf])
            with T.LetStmt(pos2seqidx[vgpos], var=seq_idx):
                with T.LetStmt((page_table_indptr[seq_idx + 1] - page_table_indptr[seq_idx] - 1) * page_size + last_page_offset[seq_idx], var=seqlen):
                    pages[page_table_values[page_table_indptr[seq_idx] + (seqlen - (append_length_indptr[seq_idx + 1] - vgpos)) // page_size], layer_id, 1, vh, (seqlen - (append_length_indptr[seq_idx + 1] - vgpos)) % page_size, vf] = v_data[vgpos, vh, vf]

@T.prim_func
def reshape(var_A: T.handle, var_T_reshape: T.handle):
    T.func_attr({"tir.noalias": T.bool(True)})
    A = T.match_buffer(var_A, (T.int64(16),))
    T_reshape = T.match_buffer(var_T_reshape, (T.int64(16), T.int64(1), T.int64(1)))
    # with T.block("root"):
    for ax0, ax1, ax2 in T.grid(T.int64(16), T.int64(1), T.int64(1)):
        with T.block("T_reshape"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(A[(v_ax0 + v_ax1 + v_ax2) % T.int64(16)])
            T.writes(T_reshape[v_ax0, v_ax1, v_ax2])
            T_reshape[v_ax0, v_ax1, v_ax2] = A[(v_ax0 + v_ax1 + v_ax2) % T.int64(16)]

@T.prim_func
def reshape1(var_A: T.handle, var_T_reshape: T.handle):
    T.func_attr({"tir.noalias": T.bool(True)})
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
def reshape2(var_A: T.handle, var_T_reshape: T.handle):
    T.func_attr({"tir.noalias": T.bool(True)})
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

# @T.prim_func
# def reshape3(var_A: T.handle, var_T_reshape: T.handle):
#     T.func_attr({"tir.noalias": T.bool(True)})
#     n = T.int64()
#     A = T.match_buffer(var_A, (T.int64(16), n), "int32")
#     T_reshape = T.match_buffer(var_T_reshape, (T.int64(16) * n,), "int32")
#     # with T.block("root"):
#     for ax0 in range(T.int64(16) * n):
#         with T.block("T_reshape"):
#             v_ax0 = T.axis.spatial(T.int64(16) * n, ax0)
#             T.reads(A[v_ax0 // n % T.int64(16), v_ax0 % n])
#             T.writes(T_reshape[v_ax0])
#             T_reshape[v_ax0] = A[v_ax0 // n % T.int64(16), v_ax0 % n]

# @T.prim_func
# def reshape4(var_A: T.handle, var_T_reshape: T.handle):
#     T.func_attr({"tir.noalias": T.bool(True)})
#     n = T.int64()
#     A = T.match_buffer(var_A, (T.int64(16) * n, T.int64(4096)), "float16")
#     T_reshape = T.match_buffer(var_T_reshape, (T.int64(16), n, T.int64(4096)), "float16")
#     # with T.block("root"):
#     for ax0, ax1, ax2 in T.grid(T.int64(16), n, T.int64(4096)):
#         with T.block("T_reshape"):
#             v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
#             T.reads(A[(v_ax2 // T.int64(4096) + v_ax0 * n + v_ax1) % (T.int64(16) * n), v_ax2 % T.int64(4096)])
#             T.writes(T_reshape[v_ax0, v_ax1, v_ax2])
#             T_reshape[v_ax0, v_ax1, v_ax2] = A[(v_ax2 // T.int64(4096) + v_ax0 * n + v_ax1) % (T.int64(16) * n), v_ax2 % T.int64(4096)]

@T.prim_func
def reshape5(var_A: T.handle, var_T_reshape: T.handle):
    T.func_attr({"tir.noalias": T.bool(True)})
    A = T.match_buffer(var_A, (T.int64(16), T.int64(1), T.int64(4096)), "float16")
    T_reshape = T.match_buffer(var_T_reshape, (T.int64(16), T.int64(1), T.int64(32), T.int64(128)), "float16")
    # with T.block("root"):
    for ax0, ax1, ax2, ax3 in T.grid(T.int64(16), T.int64(1), T.int64(32), T.int64(128)):
        with T.block("T_reshape"):
            v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
            T.reads(A[((v_ax2 * T.int64(128) + v_ax3) // T.int64(4096) + v_ax0 + v_ax1) % T.int64(16), T.int64(0), (v_ax2 * T.int64(128) + v_ax3) % T.int64(4096)])
            T.writes(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3])
            T_reshape[v_ax0, v_ax1, v_ax2, v_ax3] = A[((v_ax2 * T.int64(128) + v_ax3) // T.int64(4096) + v_ax0 + v_ax1) % T.int64(16), T.int64(0), (v_ax2 * T.int64(128) + v_ax3) % T.int64(4096)]

@T.prim_func
def reshape6(var_A: T.handle, var_T_reshape: T.handle):
    T.func_attr({"tir.noalias": T.bool(True)})
    A = T.match_buffer(var_A, (T.int64(16), T.int64(1), T.int64(32), T.int64(128)), "float16")
    T_reshape = T.match_buffer(var_T_reshape, (T.int64(16), T.int64(1), T.int64(4096)), "float16")
    # with T.block("root"):
    for ax0, ax1, ax2 in T.grid(T.int64(16), T.int64(1), T.int64(4096)):
        with T.block("T_reshape"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(A[(v_ax2 // T.int64(4096) + v_ax0 + v_ax1) % T.int64(16), T.int64(0), v_ax2 % T.int64(4096) // T.int64(128), v_ax2 % T.int64(128)])
            T.writes(T_reshape[v_ax0, v_ax1, v_ax2])
            T_reshape[v_ax0, v_ax1, v_ax2] = A[(v_ax2 // T.int64(4096) + v_ax0 + v_ax1) % T.int64(16), T.int64(0), v_ax2 % T.int64(4096) // T.int64(128), v_ax2 % T.int64(128)]

@T.prim_func
def rms_norm(var_A: T.handle, B: T.Buffer((T.int64(4096),), "float16"), var_rms_norm: T.handle):
    T.func_attr({"tir.noalias": T.bool(True)})
    n = T.int64()
    A = T.match_buffer(var_A, (T.int64(1), n, T.int64(4096)), "float16")
    rms_norm = T.match_buffer(var_rms_norm, (T.int64(1), n, T.int64(4096)), "float16")
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
            T.writes(rms_norm[v_bsz, v_i, v_k])
            rms_norm[v_bsz, v_i, v_k] = T.Cast("float16", T.Cast("float32", B[v_k]) * (T.Cast("float32", A[v_bsz, v_i, v_k]) / T.sqrt(Ared_temp[v_bsz, v_i] * T.float32(0.000244140625) + T.float32(1.0000000000000001e-05))))

@T.prim_func
def rms_norm1(var_A: T.handle, B: T.Buffer((T.int64(4096),), "float16"), var_rms_norm: T.handle):
    T.func_attr({"tir.noalias": T.bool(True)})
    A = T.match_buffer(var_A, (T.int64(16), T.int64(1), T.int64(4096)), "float16")
    rms_norm = T.match_buffer(var_rms_norm, (T.int64(16), T.int64(1), T.int64(4096)), "float16")
    # with T.block("root"):
    Ared_temp = T.alloc_buffer((T.int64(16), T.int64(1)))
    for bsz, i, k in T.grid(T.int64(16), T.int64(1), T.int64(4096)):
        with T.block("Ared_temp"):
            v_bsz, v_i, v_k = T.axis.remap("SSR", [bsz, i, k])
            T.reads(A[v_bsz, v_i, v_k])
            T.writes(Ared_temp[v_bsz, v_i])
            with T.init():
                Ared_temp[v_bsz, v_i] = T.float32(0)
            Ared_temp[v_bsz, v_i] = Ared_temp[v_bsz, v_i] + T.Cast("float32", A[v_bsz, v_i, v_k]) * T.Cast("float32", A[v_bsz, v_i, v_k])
    for bsz, i, k in T.grid(T.int64(16), T.int64(1), T.int64(4096)):
        with T.block("rms_norm"):
            v_bsz, v_i, v_k = T.axis.remap("SSS", [bsz, i, k])
            T.reads(B[v_k], A[v_bsz, v_i, v_k], Ared_temp[v_bsz, v_i])
            T.writes(rms_norm[v_bsz, v_i, v_k])
            rms_norm[v_bsz, v_i, v_k] = T.Cast("float16", T.Cast("float32", B[v_k]) * (T.Cast("float32", A[v_bsz, v_i, v_k]) / T.sqrt(Ared_temp[v_bsz, v_i] * T.float32(0.000244140625) + T.float32(1.0000000000000001e-05))))

@T.prim_func
def slice(var_A: T.handle, slice: T.Buffer((T.int64(1), T.int64(1), T.int64(4096)), "float16")):
    T.func_attr({"tir.noalias": T.bool(True)})
    n = T.int64()
    A = T.match_buffer(var_A, (T.int64(1), n, T.int64(4096)), "float16")
    # with T.block("root"):
    for i, j, k in T.grid(T.int64(1), T.int64(1), T.int64(4096)):
        with T.block("slice"):
            v_i, v_j, v_k = T.axis.remap("SSS", [i, j, k])
            T.reads(A[v_i, n - T.int64(1), v_k])
            T.writes(slice[v_i, v_j, v_k])
            slice[v_i, v_j, v_k] = A[v_i, n - T.int64(1), v_k]

@T.prim_func
def slice1(var_A: T.handle, var_slice: T.handle):
    T.func_attr({"tir.noalias": T.bool(True)})
    A = T.match_buffer(var_A, (T.int64(16), T.int64(1), T.int64(4096)), "float16")
    slice = T.match_buffer(var_slice, (T.int64(16), T.int64(1), T.int64(4096)), "float16")
    # with T.block("root"):
    for i, j, k in T.grid(T.int64(16), T.int64(1), T.int64(4096)):
        with T.block("slice"):
            v_i, v_j, v_k = T.axis.remap("SSS", [i, j, k])
            T.reads(A[v_i, T.int64(0), v_k])
            T.writes(slice[v_i, v_j, v_k])
            slice[v_i, v_j, v_k] = A[v_i, T.int64(0), v_k]

@T.prim_func
def softmax(var_A: T.handle, var_T_softmax_norm: T.handle):
    T.func_attr({"tir.noalias": T.bool(True)})
    A = T.match_buffer(var_A, (T.int64(16), T.int64(1), T.int64(32000)))
    T_softmax_norm = T.match_buffer(var_T_softmax_norm, (T.int64(16), T.int64(1), T.int64(32000)))
    # with T.block("root"):
    T_softmax_maxelem = T.alloc_buffer((T.int64(16), T.int64(1)))
    T_softmax_exp = T.alloc_buffer((T.int64(16), T.int64(1), T.int64(32000)))
    T_softmax_expsum = T.alloc_buffer((T.int64(16), T.int64(1)))
    for i0, i1, k in T.grid(T.int64(16), T.int64(1), T.int64(32000)):
        with T.block("T_softmax_maxelem"):
            v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])
            T.reads(A[v_i0, v_i1, v_k])
            T.writes(T_softmax_maxelem[v_i0, v_i1])
            with T.init():
                T_softmax_maxelem[v_i0, v_i1] = T.float32(-3.4028234663852886e+38)
            T_softmax_maxelem[v_i0, v_i1] = T.max(T_softmax_maxelem[v_i0, v_i1], A[v_i0, v_i1, v_k])
    for i0, i1, i2 in T.grid(T.int64(16), T.int64(1), T.int64(32000)):
        with T.block("T_softmax_exp"):
            v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
            T.reads(A[v_i0, v_i1, v_i2], T_softmax_maxelem[v_i0, v_i1])
            T.writes(T_softmax_exp[v_i0, v_i1, v_i2])
            T_softmax_exp[v_i0, v_i1, v_i2] = T.exp(A[v_i0, v_i1, v_i2] - T_softmax_maxelem[v_i0, v_i1])
    for i0, i1, k in T.grid(T.int64(16), T.int64(1), T.int64(32000)):
        with T.block("T_softmax_expsum"):
            v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])
            T.reads(T_softmax_exp[v_i0, v_i1, v_k])
            T.writes(T_softmax_expsum[v_i0, v_i1])
            with T.init():
                T_softmax_expsum[v_i0, v_i1] = T.float32(0)
            T_softmax_expsum[v_i0, v_i1] = T_softmax_expsum[v_i0, v_i1] + T_softmax_exp[v_i0, v_i1, v_k]
    for i0, i1, i2 in T.grid(T.int64(16), T.int64(1), T.int64(32000)):
        with T.block("T_softmax_norm"):
            v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
            T.reads(T_softmax_exp[v_i0, v_i1, v_i2], T_softmax_expsum[v_i0, v_i1])
            T.writes(T_softmax_norm[v_i0, v_i1, v_i2])
            T.block_attr({"axis": 2})
            T_softmax_norm[v_i0, v_i1, v_i2] = T_softmax_exp[v_i0, v_i1, v_i2] / T_softmax_expsum[v_i0, v_i1]

@T.prim_func
def split(var_A: T.handle, var_T_split: T.handle, var_T_split_1: T.handle, var_T_split_2: T.handle):
    T.func_attr({"tir.noalias": T.bool(True)})
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
def split2(var_A: T.handle, var_T_split: T.handle, var_T_split_1: T.handle, var_T_split_2: T.handle):
    T.func_attr({"tir.noalias": T.bool(True)})
    A = T.match_buffer(var_A, (T.int64(16), T.int64(1), T.int64(12288)), "float16")
    T_split = T.match_buffer(var_T_split, (T.int64(16), T.int64(1), T.int64(4096)), "float16")
    T_split_1 = T.match_buffer(var_T_split_1, (T.int64(16), T.int64(1), T.int64(4096)), "float16")
    T_split_2 = T.match_buffer(var_T_split_2, (T.int64(16), T.int64(1), T.int64(4096)), "float16")
    # with T.block("root"):
    for ax0, ax1, ax2 in T.grid(T.int64(16), T.int64(1), T.int64(4096)):
        with T.block("T_split"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(A[v_ax0, v_ax1, v_ax2])
            T.writes(T_split[v_ax0, v_ax1, v_ax2])
            T_split[v_ax0, v_ax1, v_ax2] = A[v_ax0, v_ax1, v_ax2]
    for ax0, ax1, ax2 in T.grid(T.int64(16), T.int64(1), T.int64(4096)):
        with T.block("T_split_1"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(A[v_ax0, v_ax1, v_ax2 + T.int64(4096)])
            T.writes(T_split_1[v_ax0, v_ax1, v_ax2])
            T_split_1[v_ax0, v_ax1, v_ax2] = A[v_ax0, v_ax1, v_ax2 + T.int64(4096)]
    for ax0, ax1, ax2 in T.grid(T.int64(16), T.int64(1), T.int64(4096)):
        with T.block("T_split_2"):
            v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
            T.reads(A[v_ax0, v_ax1, v_ax2 + T.int64(8192)])
            T.writes(T_split_2[v_ax0, v_ax1, v_ax2])
            T_split_2[v_ax0, v_ax1, v_ax2] = A[v_ax0, v_ax1, v_ax2 + T.int64(8192)]

DlightBench.register_bench_workload(divide, "llama_2_batch", "divide")
DlightBench.register_bench_workload(fused_fused_decode1_fused_NT_matmul1_add, "llama_2_batch", "fused_fused_decode1_fused_NT_matmul1_add")
DlightBench.register_bench_workload(fused_fused_decode1_fused_NT_matmul6_add1, "llama_2_batch", "fused_fused_decode1_fused_NT_matmul6_add1")
DlightBench.register_bench_workload(fused_fused_decode2_NT_matmul2, "llama_2_batch", "fused_fused_decode2_NT_matmul2")
DlightBench.register_bench_workload(fused_fused_decode2_NT_matmul7, "llama_2_batch", "fused_fused_decode2_NT_matmul7")
DlightBench.register_bench_workload(fused_fused_decode3_fused_NT_matmul3_add, "llama_2_batch", "fused_fused_decode3_fused_NT_matmul3_add")
DlightBench.register_bench_workload(fused_fused_decode3_fused_NT_matmul8_add1, "llama_2_batch", "fused_fused_decode3_fused_NT_matmul8_add1")
DlightBench.register_bench_workload(fused_fused_decode4_fused_NT_matmul4_cast, "llama_2_batch", "fused_fused_decode4_fused_NT_matmul4_cast")
DlightBench.register_bench_workload(fused_fused_decode4_fused_NT_matmul9_cast1, "llama_2_batch", "fused_fused_decode4_fused_NT_matmul9_cast1")
# DlightBench.register_bench_workload(fused_fused_decode4_take, "llama_2_batch", "fused_fused_decode4_take")
DlightBench.register_bench_workload(fused_fused_decode_NT_matmul, "llama_2_batch", "fused_fused_decode_NT_matmul")
DlightBench.register_bench_workload(fused_fused_decode_NT_matmul5, "llama_2_batch", "fused_fused_decode_NT_matmul5")
DlightBench.register_bench_workload(fused_split1_silu_multiply, "llama_2_batch", "fused_split1_silu_multiply")
DlightBench.register_bench_workload(fused_split3_silu1_multiply1, "llama_2_batch", "fused_split3_silu1_multiply1")
DlightBench.register_bench_workload(kv_cache_transpose_append, "llama_2_batch", "kv_cache_transpose_append")
DlightBench.register_bench_workload(reshape, "llama_2_batch", "reshape")
DlightBench.register_bench_workload(reshape1, "llama_2_batch", "reshape1")
DlightBench.register_bench_workload(reshape2, "llama_2_batch", "reshape2")
# DlightBench.register_bench_workload(reshape3, "llama_2_batch", "reshape3")
# DlightBench.register_bench_workload(reshape4, "llama_2_batch", "reshape4")
DlightBench.register_bench_workload(reshape5, "llama_2_batch", "reshape5")
DlightBench.register_bench_workload(reshape6, "llama_2_batch", "reshape6")
DlightBench.register_bench_workload(rms_norm, "llama_2_batch", "rms_norm")
DlightBench.register_bench_workload(rms_norm1, "llama_2_batch", "rms_norm1")
DlightBench.register_bench_workload(slice, "llama_2_batch", "slice")
DlightBench.register_bench_workload(slice1, "llama_2_batch", "slice1")
DlightBench.register_bench_workload(softmax, "llama_2_batch", "softmax")
DlightBench.register_bench_workload(split, "llama_2_batch", "split")
DlightBench.register_bench_workload(split2, "llama_2_batch", "split2")

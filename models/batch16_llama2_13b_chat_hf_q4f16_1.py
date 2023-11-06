# from tvm.script import ir as I
# from tvm.script import tir as T
# from tvm.script import relax as R

@I.ir_module
class Module:
    @T.prim_func(private=True)
    def divide(var_A: T.handle, var_B: T.handle, var_T_divide: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        nseq, vocab_size = T.int64(), T.int64()
        A = T.match_buffer(var_A, (nseq, T.int64(1), vocab_size))
        B = T.match_buffer(var_B, (nseq, T.int64(1), T.int64(1)))
        T_divide = T.match_buffer(var_T_divide, (nseq, T.int64(1), vocab_size))
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(nseq, T.int64(1), vocab_size):
            with T.block("T_divide"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(A[v_ax0, v_ax1, v_ax2], B[v_ax0, v_ax1, T.int64(0)])
                T.writes(T_divide[v_ax0, v_ax1, v_ax2])
                T_divide[v_ax0, v_ax1, v_ax2] = A[v_ax0, v_ax1, v_ax2] / B[v_ax0, v_ax1, T.int64(0)]

    @T.prim_func(private=True)
    def fused_fused_decode1_fused_NT_matmul1_add(lv3: T.Buffer((T.int64(5120), T.int64(640)), "uint32"), lv4: T.Buffer((T.int64(5120), T.int64(160)), "float16"), p_lv1100: T.handle, p_inputs_embeds1: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        nseq = T.int64()
        lv1100 = T.match_buffer(p_lv1100, (nseq, T.int64(1), T.int64(5120)), "float16")
        inputs_embeds1 = T.match_buffer(p_inputs_embeds1, (nseq, T.int64(1), T.int64(5120)), "float16")
        p_output0_intermediate = T.match_buffer(p_output0, (nseq, T.int64(1), T.int64(5120)), "float16")
        # with T.block("root"):
        p_output0_intermediate_1 = T.alloc_buffer((T.int64(5120), T.int64(5120)), "float16")
        var_NT_matmul_intermediate = T.alloc_buffer((nseq, T.int64(1), T.int64(5120)), "float16")
        for i, j in T.grid(T.int64(5120), T.int64(5120)):
            with T.block("decode"):
                v_i, v_j = T.axis.remap("SS", [i, j])
                T.reads(lv3[v_i, v_j // T.int64(8)], lv4[v_i, v_j // T.int64(32)])
                T.writes(p_output0_intermediate_1[v_i, v_j])
                p_output0_intermediate_1[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv3[v_i, v_j // T.int64(8)], T.Cast("uint32", v_j % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv4[v_i, v_j // T.int64(32)]
        for i0, i1, i2, k in T.grid(nseq, T.int64(1), T.int64(5120), T.int64(5120)):
            with T.block("NT_matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(lv1100[v_i0, v_i1, v_k], p_output0_intermediate_1[v_i2, v_k])
                T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
                with T.init():
                    var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2] + lv1100[v_i0, v_i1, v_k] * p_output0_intermediate_1[v_i2, v_k]
        for ax0, ax1, ax2 in T.grid(nseq, T.int64(1), T.int64(5120)):
            with T.block("T_add"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(inputs_embeds1[v_ax0, v_ax1, v_ax2], var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2])
                T.writes(p_output0_intermediate[v_ax0, v_ax1, v_ax2])
                p_output0_intermediate[v_ax0, v_ax1, v_ax2] = inputs_embeds1[v_ax0, v_ax1, v_ax2] + var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2]

    @T.prim_func(private=True)
    def fused_fused_decode1_fused_NT_matmul6_add1(lv610: T.Buffer((T.int64(5120), T.int64(640)), "uint32"), lv611: T.Buffer((T.int64(5120), T.int64(160)), "float16"), p_lv15: T.handle, p_inputs_embeds: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        lv15 = T.match_buffer(p_lv15, (T.int64(1), n, T.int64(5120)), "float16")
        inputs_embeds = T.match_buffer(p_inputs_embeds, (T.int64(1), n, T.int64(5120)), "float16")
        p_output0_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(5120)), "float16")
        # with T.block("root"):
        p_output0_intermediate_1 = T.alloc_buffer((T.int64(5120), T.int64(5120)), "float16")
        var_NT_matmul_intermediate = T.alloc_buffer((T.int64(1), n, T.int64(5120)), "float16")
        for i, j in T.grid(T.int64(5120), T.int64(5120)):
            with T.block("decode"):
                v_i, v_j = T.axis.remap("SS", [i, j])
                T.reads(lv610[v_i, v_j // T.int64(8)], lv611[v_i, v_j // T.int64(32)])
                T.writes(p_output0_intermediate_1[v_i, v_j])
                p_output0_intermediate_1[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv610[v_i, v_j // T.int64(8)], T.Cast("uint32", v_j % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv611[v_i, v_j // T.int64(32)]
        for i0, i1, i2, k in T.grid(T.int64(1), n, T.int64(5120), T.int64(5120)):
            with T.block("NT_matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(lv15[v_i0, v_i1, v_k], p_output0_intermediate_1[v_i2, v_k])
                T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
                with T.init():
                    var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2] + lv15[v_i0, v_i1, v_k] * p_output0_intermediate_1[v_i2, v_k]
        for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(5120)):
            with T.block("T_add"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(inputs_embeds[v_ax0, v_ax1, v_ax2], var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2])
                T.writes(p_output0_intermediate[v_ax0, v_ax1, v_ax2])
                p_output0_intermediate[v_ax0, v_ax1, v_ax2] = inputs_embeds[v_ax0, v_ax1, v_ax2] + var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2]

    @T.prim_func(private=True)
    def fused_fused_decode2_NT_matmul2(lv7: T.Buffer((T.int64(27648), T.int64(640)), "uint32"), lv8: T.Buffer((T.int64(27648), T.int64(160)), "float16"), p_lv1104: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        nseq = T.int64()
        lv1104 = T.match_buffer(p_lv1104, (nseq, T.int64(1), T.int64(5120)), "float16")
        var_NT_matmul_intermediate = T.match_buffer(p_output0, (nseq, T.int64(1), T.int64(27648)), "float16")
        # with T.block("root"):
        p_output0_intermediate = T.alloc_buffer((T.int64(27648), T.int64(5120)), "float16")
        for i, j in T.grid(T.int64(27648), T.int64(5120)):
            with T.block("decode"):
                v_i, v_j = T.axis.remap("SS", [i, j])
                T.reads(lv7[v_i, v_j // T.int64(8)], lv8[v_i, v_j // T.int64(32)])
                T.writes(p_output0_intermediate[v_i, v_j])
                p_output0_intermediate[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv7[v_i, v_j // T.int64(8)], T.Cast("uint32", v_j % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv8[v_i, v_j // T.int64(32)]
        for i0, i1, i2, k in T.grid(nseq, T.int64(1), T.int64(27648), T.int64(5120)):
            with T.block("NT_matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(lv1104[v_i0, v_i1, v_k], p_output0_intermediate[v_i2, v_k])
                T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
                with T.init():
                    var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2] + lv1104[v_i0, v_i1, v_k] * p_output0_intermediate[v_i2, v_k]

    @T.prim_func(private=True)
    def fused_fused_decode2_NT_matmul7(lv614: T.Buffer((T.int64(27648), T.int64(640)), "uint32"), lv615: T.Buffer((T.int64(27648), T.int64(160)), "float16"), p_lv19: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        lv19 = T.match_buffer(p_lv19, (T.int64(1), n, T.int64(5120)), "float16")
        var_NT_matmul_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(27648)), "float16")
        # with T.block("root"):
        p_output0_intermediate = T.alloc_buffer((T.int64(27648), T.int64(5120)), "float16")
        for i, j in T.grid(T.int64(27648), T.int64(5120)):
            with T.block("decode"):
                v_i, v_j = T.axis.remap("SS", [i, j])
                T.reads(lv614[v_i, v_j // T.int64(8)], lv615[v_i, v_j // T.int64(32)])
                T.writes(p_output0_intermediate[v_i, v_j])
                p_output0_intermediate[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv614[v_i, v_j // T.int64(8)], T.Cast("uint32", v_j % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv615[v_i, v_j // T.int64(32)]
        for i0, i1, i2, k in T.grid(T.int64(1), n, T.int64(27648), T.int64(5120)):
            with T.block("NT_matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(lv19[v_i0, v_i1, v_k], p_output0_intermediate[v_i2, v_k])
                T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
                with T.init():
                    var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2] + lv19[v_i0, v_i1, v_k] * p_output0_intermediate[v_i2, v_k]

    @T.prim_func(private=True)
    def fused_fused_decode3_fused_NT_matmul3_add(lv11: T.Buffer((T.int64(5120), T.int64(1728)), "uint32"), lv12: T.Buffer((T.int64(5120), T.int64(432)), "float16"), p_lv10: T.handle, p_lv6: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        nseq = T.int64()
        lv10 = T.match_buffer(p_lv10, (nseq, T.int64(1), T.int64(13824)), "float16")
        lv6 = T.match_buffer(p_lv6, (nseq, T.int64(1), T.int64(5120)), "float16")
        p_output0_intermediate = T.match_buffer(p_output0, (nseq, T.int64(1), T.int64(5120)), "float16")
        # with T.block("root"):
        p_output0_intermediate_1 = T.alloc_buffer((T.int64(5120), T.int64(13824)), "float16")
        var_NT_matmul_intermediate = T.alloc_buffer((nseq, T.int64(1), T.int64(5120)), "float16")
        for i, j in T.grid(T.int64(5120), T.int64(13824)):
            with T.block("decode"):
                v_i, v_j = T.axis.remap("SS", [i, j])
                T.reads(lv11[v_i, v_j // T.int64(8)], lv12[v_i, v_j // T.int64(32)])
                T.writes(p_output0_intermediate_1[v_i, v_j])
                p_output0_intermediate_1[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv11[v_i, v_j // T.int64(8)], T.Cast("uint32", v_j % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv12[v_i, v_j // T.int64(32)]
        for i0, i1, i2, k in T.grid(nseq, T.int64(1), T.int64(5120), T.int64(13824)):
            with T.block("NT_matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(lv10[v_i0, v_i1, v_k], p_output0_intermediate_1[v_i2, v_k])
                T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
                with T.init():
                    var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2] + lv10[v_i0, v_i1, v_k] * p_output0_intermediate_1[v_i2, v_k]
        for ax0, ax1, ax2 in T.grid(nseq, T.int64(1), T.int64(5120)):
            with T.block("T_add"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(lv6[v_ax0, v_ax1, v_ax2], var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2])
                T.writes(p_output0_intermediate[v_ax0, v_ax1, v_ax2])
                p_output0_intermediate[v_ax0, v_ax1, v_ax2] = lv6[v_ax0, v_ax1, v_ax2] + var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2]

    @T.prim_func(private=True)
    def fused_fused_decode3_fused_NT_matmul8_add1(lv618: T.Buffer((T.int64(5120), T.int64(1728)), "uint32"), lv619: T.Buffer((T.int64(5120), T.int64(432)), "float16"), p_lv617: T.handle, p_lv613: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        lv617 = T.match_buffer(p_lv617, (T.int64(1), n, T.int64(13824)), "float16")
        lv613 = T.match_buffer(p_lv613, (T.int64(1), n, T.int64(5120)), "float16")
        p_output0_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(5120)), "float16")
        # with T.block("root"):
        p_output0_intermediate_1 = T.alloc_buffer((T.int64(5120), T.int64(13824)), "float16")
        var_NT_matmul_intermediate = T.alloc_buffer((T.int64(1), n, T.int64(5120)), "float16")
        for i, j in T.grid(T.int64(5120), T.int64(13824)):
            with T.block("decode"):
                v_i, v_j = T.axis.remap("SS", [i, j])
                T.reads(lv618[v_i, v_j // T.int64(8)], lv619[v_i, v_j // T.int64(32)])
                T.writes(p_output0_intermediate_1[v_i, v_j])
                p_output0_intermediate_1[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv618[v_i, v_j // T.int64(8)], T.Cast("uint32", v_j % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv619[v_i, v_j // T.int64(32)]
        for i0, i1, i2, k in T.grid(T.int64(1), n, T.int64(5120), T.int64(13824)):
            with T.block("NT_matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(lv617[v_i0, v_i1, v_k], p_output0_intermediate_1[v_i2, v_k])
                T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
                with T.init():
                    var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2] + lv617[v_i0, v_i1, v_k] * p_output0_intermediate_1[v_i2, v_k]
        for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(5120)):
            with T.block("T_add"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(lv613[v_ax0, v_ax1, v_ax2], var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2])
                T.writes(p_output0_intermediate[v_ax0, v_ax1, v_ax2])
                p_output0_intermediate[v_ax0, v_ax1, v_ax2] = lv613[v_ax0, v_ax1, v_ax2] + var_NT_matmul_intermediate[v_ax0, v_ax1, v_ax2]

    @T.prim_func(private=True)
    def fused_fused_decode4_fused_NT_matmul4_cast(p_lv600: T.handle, p_lv601: T.handle, p_lv2169: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        vocab_size = T.int64()
        lv600 = T.match_buffer(p_lv600, (vocab_size, T.int64(640)), "uint32")
        lv601 = T.match_buffer(p_lv601, (vocab_size, T.int64(160)), "float16")
        nseq = T.int64()
        lv2169 = T.match_buffer(p_lv2169, (nseq, T.int64(1), T.int64(5120)), "float16")
        p_output0_intermediate = T.match_buffer(p_output0, (nseq, T.int64(1), vocab_size))
        # with T.block("root"):
        p_output0_intermediate_1 = T.alloc_buffer((vocab_size, T.int64(5120)), "float16")
        var_NT_matmul_intermediate = T.alloc_buffer((nseq, T.int64(1), vocab_size), "float16")
        for i, j in T.grid(vocab_size, T.int64(5120)):
            with T.block("decode"):
                v_i, v_j = T.axis.remap("SS", [i, j])
                T.reads(lv600[v_i, v_j // T.int64(8)], lv601[v_i, v_j // T.int64(32)])
                T.writes(p_output0_intermediate_1[v_i, v_j])
                p_output0_intermediate_1[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv600[v_i, v_j // T.int64(8)], T.Cast("uint32", v_j % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv601[v_i, v_j // T.int64(32)]
        for i0, i1, i2, k in T.grid(nseq, T.int64(1), vocab_size, T.int64(5120)):
            with T.block("NT_matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(lv2169[v_i0, v_i1, v_k], p_output0_intermediate_1[v_i2, v_k])
                T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
                with T.init():
                    var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2] + lv2169[v_i0, v_i1, v_k] * p_output0_intermediate_1[v_i2, v_k]
        for i0, i1, i2 in T.grid(nseq, T.int64(1), vocab_size):
            with T.block("compute"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                T.reads(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
                T.writes(p_output0_intermediate[v_i0, v_i1, v_i2])
                p_output0_intermediate[v_i0, v_i1, v_i2] = T.Cast("float32", var_NT_matmul_intermediate[v_i0, v_i1, v_i2])

    @T.prim_func(private=True)
    def fused_fused_decode4_fused_NT_matmul9_cast1(p_lv1207: T.handle, p_lv1208: T.handle, lv1084: T.Buffer((T.int64(1), T.int64(1), T.int64(5120)), "float16"), p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        vocab_size = T.int64()
        lv1207 = T.match_buffer(p_lv1207, (vocab_size, T.int64(640)), "uint32")
        lv1208 = T.match_buffer(p_lv1208, (vocab_size, T.int64(160)), "float16")
        p_output0_intermediate = T.match_buffer(p_output0, (T.int64(1), T.int64(1), vocab_size))
        # with T.block("root"):
        p_output0_intermediate_1 = T.alloc_buffer((vocab_size, T.int64(5120)), "float16")
        var_NT_matmul_intermediate = T.alloc_buffer((T.int64(1), T.int64(1), vocab_size), "float16")
        for i, j in T.grid(vocab_size, T.int64(5120)):
            with T.block("decode"):
                v_i, v_j = T.axis.remap("SS", [i, j])
                T.reads(lv1207[v_i, v_j // T.int64(8)], lv1208[v_i, v_j // T.int64(32)])
                T.writes(p_output0_intermediate_1[v_i, v_j])
                p_output0_intermediate_1[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv1207[v_i, v_j // T.int64(8)], T.Cast("uint32", v_j % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv1208[v_i, v_j // T.int64(32)]
        for i0, i1, i2, k in T.grid(T.int64(1), T.int64(1), vocab_size, T.int64(5120)):
            with T.block("NT_matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(lv1084[v_i0, v_i1, v_k], p_output0_intermediate_1[v_i2, v_k])
                T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
                with T.init():
                    var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2] + lv1084[v_i0, v_i1, v_k] * p_output0_intermediate_1[v_i2, v_k]
        for i0, i1, i2 in T.grid(T.int64(1), T.int64(1), vocab_size):
            with T.block("compute"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                T.reads(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
                T.writes(p_output0_intermediate[v_i0, v_i1, v_i2])
                p_output0_intermediate[v_i0, v_i1, v_i2] = T.Cast("float32", var_NT_matmul_intermediate[v_i0, v_i1, v_i2])

    @T.prim_func(private=True)
    def fused_fused_decode4_take(p_lv604: T.handle, p_lv605: T.handle, p_lv: T.handle, p_output0: T.handle, n: T.int32, nseq: T.int32):
        T.func_attr({"tir.noalias": T.bool(True)})
        vocab_size = T.int32()
        lv604 = T.match_buffer(p_lv604, (vocab_size, 640), "uint32")
        lv605 = T.match_buffer(p_lv605, (vocab_size, 160), "float16")
        lv = T.match_buffer(p_lv, (nseq * n,), "int32")
        var_T_take_intermediate = T.match_buffer(p_output0, (nseq * n, 5120), "float16")
        # with T.block("root"):
        for ax0, ax1 in T.grid(nseq * n, 5120):
            with T.block("T_take"):
                v_ax0, v_ax1 = T.axis.remap("SS", [ax0, ax1])
                T.reads(lv604[lv[v_ax0], v_ax1 // 8], lv[v_ax0], lv605[lv[v_ax0], v_ax1 // 32])
                T.writes(var_T_take_intermediate[v_ax0, v_ax1])
                var_T_take_intermediate[v_ax0, v_ax1] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv604[lv[v_ax0], v_ax1 // 8], T.Cast("uint32", v_ax1 % 8) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv605[lv[v_ax0], v_ax1 // 32]

    @T.prim_func(private=True)
    def fused_fused_decode_NT_matmul(lv: T.Buffer((T.int64(15360), T.int64(640)), "uint32"), lv1: T.Buffer((T.int64(15360), T.int64(160)), "float16"), p_lv1088: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        nseq = T.int64()
        lv1088 = T.match_buffer(p_lv1088, (nseq, T.int64(1), T.int64(5120)), "float16")
        var_NT_matmul_intermediate = T.match_buffer(p_output0, (nseq, T.int64(1), T.int64(15360)), "float16")
        # with T.block("root"):
        p_output0_intermediate = T.alloc_buffer((T.int64(15360), T.int64(5120)), "float16")
        for i, j in T.grid(T.int64(15360), T.int64(5120)):
            with T.block("decode"):
                v_i, v_j = T.axis.remap("SS", [i, j])
                T.reads(lv[v_i, v_j // T.int64(8)], lv1[v_i, v_j // T.int64(32)])
                T.writes(p_output0_intermediate[v_i, v_j])
                p_output0_intermediate[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv[v_i, v_j // T.int64(8)], T.Cast("uint32", v_j % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv1[v_i, v_j // T.int64(32)]
        for i0, i1, i2, k in T.grid(nseq, T.int64(1), T.int64(15360), T.int64(5120)):
            with T.block("NT_matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(lv1088[v_i0, v_i1, v_k], p_output0_intermediate[v_i2, v_k])
                T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
                with T.init():
                    var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2] + lv1088[v_i0, v_i1, v_k] * p_output0_intermediate[v_i2, v_k]

    @T.prim_func(private=True)
    def fused_fused_decode_NT_matmul5(lv607: T.Buffer((T.int64(15360), T.int64(640)), "uint32"), lv608: T.Buffer((T.int64(15360), T.int64(160)), "float16"), p_lv3: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        lv3 = T.match_buffer(p_lv3, (T.int64(1), n, T.int64(5120)), "float16")
        var_NT_matmul_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(15360)), "float16")
        # with T.block("root"):
        p_output0_intermediate = T.alloc_buffer((T.int64(15360), T.int64(5120)), "float16")
        for i, j in T.grid(T.int64(15360), T.int64(5120)):
            with T.block("decode"):
                v_i, v_j = T.axis.remap("SS", [i, j])
                T.reads(lv607[v_i, v_j // T.int64(8)], lv608[v_i, v_j // T.int64(32)])
                T.writes(p_output0_intermediate[v_i, v_j])
                p_output0_intermediate[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(lv607[v_i, v_j // T.int64(8)], T.Cast("uint32", v_j % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * lv608[v_i, v_j // T.int64(32)]
        for i0, i1, i2, k in T.grid(T.int64(1), n, T.int64(15360), T.int64(5120)):
            with T.block("NT_matmul"):
                v_i0, v_i1, v_i2, v_k = T.axis.remap("SSSR", [i0, i1, i2, k])
                T.reads(lv3[v_i0, v_i1, v_k], p_output0_intermediate[v_i2, v_k])
                T.writes(var_NT_matmul_intermediate[v_i0, v_i1, v_i2])
                with T.init():
                    var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = T.float16(0)
                var_NT_matmul_intermediate[v_i0, v_i1, v_i2] = var_NT_matmul_intermediate[v_i0, v_i1, v_i2] + lv3[v_i0, v_i1, v_k] * p_output0_intermediate[v_i2, v_k]

    @T.prim_func(private=True)
    def fused_split1_silu_multiply(p_lv2: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        nseq = T.int64()
        lv2 = T.match_buffer(p_lv2, (nseq, T.int64(1), T.int64(27648)), "float16")
        var_T_multiply_intermediate = T.match_buffer(p_output0, (nseq, T.int64(1), T.int64(13824)), "float16")
        # with T.block("root"):
        var_T_split_sections_intermediate = T.alloc_buffer((nseq, T.int64(1), T.int64(13824)), "float16")
        var_T_split_sections_intermediate_1 = T.alloc_buffer((nseq, T.int64(1), T.int64(13824)), "float16")
        compute = T.alloc_buffer((nseq, T.int64(1), T.int64(13824)), "float16")
        var_T_multiply_intermediate_1 = T.alloc_buffer((nseq, T.int64(1), T.int64(13824)), "float16")
        for ax0, ax1, ax2 in T.grid(nseq, T.int64(1), T.int64(13824)):
            with T.block("T_split_sections"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(lv2[v_ax0, v_ax1, v_ax2])
                T.writes(var_T_split_sections_intermediate[v_ax0, v_ax1, v_ax2])
                var_T_split_sections_intermediate[v_ax0, v_ax1, v_ax2] = lv2[v_ax0, v_ax1, v_ax2]
        for ax0, ax1, ax2 in T.grid(nseq, T.int64(1), T.int64(13824)):
            with T.block("T_split_sections_1"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(lv2[v_ax0, v_ax1, v_ax2 + T.int64(13824)])
                T.writes(var_T_split_sections_intermediate_1[v_ax0, v_ax1, v_ax2])
                var_T_split_sections_intermediate_1[v_ax0, v_ax1, v_ax2] = lv2[v_ax0, v_ax1, v_ax2 + T.int64(13824)]
        for i0, i1, i2 in T.grid(nseq, T.int64(1), T.int64(13824)):
            with T.block("compute"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                T.reads(var_T_split_sections_intermediate[v_i0, v_i1, v_i2])
                T.writes(compute[v_i0, v_i1, v_i2])
                compute[v_i0, v_i1, v_i2] = T.sigmoid(var_T_split_sections_intermediate[v_i0, v_i1, v_i2])
        for ax0, ax1, ax2 in T.grid(nseq, T.int64(1), T.int64(13824)):
            with T.block("T_multiply"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(var_T_split_sections_intermediate[v_ax0, v_ax1, v_ax2], compute[v_ax0, v_ax1, v_ax2])
                T.writes(var_T_multiply_intermediate_1[v_ax0, v_ax1, v_ax2])
                var_T_multiply_intermediate_1[v_ax0, v_ax1, v_ax2] = var_T_split_sections_intermediate[v_ax0, v_ax1, v_ax2] * compute[v_ax0, v_ax1, v_ax2]
        for ax0, ax1, ax2 in T.grid(nseq, T.int64(1), T.int64(13824)):
            with T.block("T_multiply_1"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(var_T_multiply_intermediate_1[v_ax0, v_ax1, v_ax2], var_T_split_sections_intermediate_1[v_ax0, v_ax1, v_ax2])
                T.writes(var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2])
                var_T_multiply_intermediate[v_ax0, v_ax1, v_ax2] = var_T_multiply_intermediate_1[v_ax0, v_ax1, v_ax2] * var_T_split_sections_intermediate_1[v_ax0, v_ax1, v_ax2]

    @T.prim_func(private=True)
    def fused_split3_silu1_multiply1(p_lv163: T.handle, p_output0: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        lv163 = T.match_buffer(p_lv163, (T.int64(1), n, T.int64(27648)), "float16")
        var_T_multiply_intermediate = T.match_buffer(p_output0, (T.int64(1), n, T.int64(13824)), "float16")
        # with T.block("root"):
        var_T_split_sections_intermediate = T.alloc_buffer((T.int64(1), n, T.int64(13824)), "float16")
        var_T_split_sections_intermediate_1 = T.alloc_buffer((T.int64(1), n, T.int64(13824)), "float16")
        compute = T.alloc_buffer((T.int64(1), n, T.int64(13824)), "float16")
        var_T_multiply_intermediate_1 = T.alloc_buffer((T.int64(1), n, T.int64(13824)), "float16")
        for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(13824)):
            with T.block("T_split_sections"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(lv163[v_ax0, v_ax1, v_ax2])
                T.writes(var_T_split_sections_intermediate[v_ax0, v_ax1, v_ax2])
                var_T_split_sections_intermediate[v_ax0, v_ax1, v_ax2] = lv163[v_ax0, v_ax1, v_ax2]
        for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(13824)):
            with T.block("T_split_sections_1"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(lv163[v_ax0, v_ax1, v_ax2 + T.int64(13824)])
                T.writes(var_T_split_sections_intermediate_1[v_ax0, v_ax1, v_ax2])
                var_T_split_sections_intermediate_1[v_ax0, v_ax1, v_ax2] = lv163[v_ax0, v_ax1, v_ax2 + T.int64(13824)]
        for i0, i1, i2 in T.grid(T.int64(1), n, T.int64(13824)):
            with T.block("compute"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                T.reads(var_T_split_sections_intermediate[v_i0, v_i1, v_i2])
                T.writes(compute[v_i0, v_i1, v_i2])
                compute[v_i0, v_i1, v_i2] = T.sigmoid(var_T_split_sections_intermediate[v_i0, v_i1, v_i2])
        for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(13824)):
            with T.block("T_multiply"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(var_T_split_sections_intermediate[v_ax0, v_ax1, v_ax2], compute[v_ax0, v_ax1, v_ax2])
                T.writes(var_T_multiply_intermediate_1[v_ax0, v_ax1, v_ax2])
                var_T_multiply_intermediate_1[v_ax0, v_ax1, v_ax2] = var_T_split_sections_intermediate[v_ax0, v_ax1, v_ax2] * compute[v_ax0, v_ax1, v_ax2]
        for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(13824)):
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
        nseq = T.int32()
        page_table_indptr = T.match_buffer(var_page_table_indptr, (nseq + 1,), "int32")
        npage = T.int32()
        page_table_values = T.match_buffer(var_page_table_values, (npage,), "int32")
        last_page_offset = T.match_buffer(var_last_page_offset, (nseq,), "int32")
        append_length_indptr = T.match_buffer(var_append_length_indptr, (nseq + 1,), "int32")
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

    @T.prim_func(private=True)
    def reshape(var_A: T.handle, var_T_reshape: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        nseq = T.int64()
        A = T.match_buffer(var_A, (nseq, T.int64(1), T.int64(5120)), "float16")
        T_reshape = T.match_buffer(var_T_reshape, (nseq, T.int64(1), T.int64(40), T.int64(128)), "float16")
        # with T.block("root"):
        for ax0, ax1, ax2, ax3 in T.grid(nseq, T.int64(1), T.int64(40), T.int64(128)):
            with T.block("T_reshape"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(A[((v_ax2 * T.int64(128) + v_ax3) // T.int64(5120) + v_ax0 + v_ax1) % nseq, T.int64(0), (v_ax2 * T.int64(128) + v_ax3) % T.int64(5120)])
                T.writes(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3])
                T_reshape[v_ax0, v_ax1, v_ax2, v_ax3] = A[((v_ax2 * T.int64(128) + v_ax3) // T.int64(5120) + v_ax0 + v_ax1) % nseq, T.int64(0), (v_ax2 * T.int64(128) + v_ax3) % T.int64(5120)]

    @T.prim_func(private=True)
    def reshape1(var_A: T.handle, var_T_reshape: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        nseq = T.int64()
        A = T.match_buffer(var_A, (nseq, T.int64(1), T.int64(40), T.int64(128)), "float16")
        T_reshape = T.match_buffer(var_T_reshape, (nseq, T.int64(1), T.int64(5120)), "float16")
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(nseq, T.int64(1), T.int64(5120)):
            with T.block("T_reshape"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(A[(v_ax2 // T.int64(5120) + v_ax0 + v_ax1) % nseq, T.int64(0), v_ax2 % T.int64(5120) // T.int64(128), v_ax2 % T.int64(128)])
                T.writes(T_reshape[v_ax0, v_ax1, v_ax2])
                T_reshape[v_ax0, v_ax1, v_ax2] = A[(v_ax2 // T.int64(5120) + v_ax0 + v_ax1) % nseq, T.int64(0), v_ax2 % T.int64(5120) // T.int64(128), v_ax2 % T.int64(128)]

    @T.prim_func(private=True)
    def reshape2(var_A: T.handle, var_T_reshape: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        nseq = T.int64()
        A = T.match_buffer(var_A, (nseq,))
        T_reshape = T.match_buffer(var_T_reshape, (nseq, T.int64(1), T.int64(1)))
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(nseq, T.int64(1), T.int64(1)):
            with T.block("T_reshape"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(A[(v_ax0 + v_ax1 + v_ax2) % nseq])
                T.writes(T_reshape[v_ax0, v_ax1, v_ax2])
                T_reshape[v_ax0, v_ax1, v_ax2] = A[(v_ax0 + v_ax1 + v_ax2) % nseq]

    @T.prim_func(private=True)
    def reshape3(var_A: T.handle, var_T_reshape: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        nseq, n = T.int64(), T.int64()
        A = T.match_buffer(var_A, (nseq, n), "int32")
        T_reshape = T.match_buffer(var_T_reshape, (nseq * n,), "int32")
        # with T.block("root"):
        for ax0 in range(nseq * n):
            with T.block("T_reshape"):
                v_ax0 = T.axis.spatial(nseq * n, ax0)
                T.reads(A[v_ax0 // n % nseq, v_ax0 % n])
                T.writes(T_reshape[v_ax0])
                T_reshape[v_ax0] = A[v_ax0 // n % nseq, v_ax0 % n]

    @T.prim_func(private=True)
    def reshape4(var_A: T.handle, var_T_reshape: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        nseq = T.int64()
        n = T.int64()
        A = T.match_buffer(var_A, (nseq * n, T.int64(5120)), "float16")
        T_reshape = T.match_buffer(var_T_reshape, (nseq, n, T.int64(5120)), "float16")
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(nseq, n, T.int64(5120)):
            with T.block("T_reshape"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(A[(v_ax2 // T.int64(5120) + v_ax0 * n + v_ax1) % (nseq * n), v_ax2 % T.int64(5120)])
                T.writes(T_reshape[v_ax0, v_ax1, v_ax2])
                T_reshape[v_ax0, v_ax1, v_ax2] = A[(v_ax2 // T.int64(5120) + v_ax0 * n + v_ax1) % (nseq * n), v_ax2 % T.int64(5120)]

    @T.prim_func(private=True)
    def reshape5(var_A: T.handle, var_T_reshape: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(5120)), "float16")
        T_reshape = T.match_buffer(var_T_reshape, (T.int64(1), n, T.int64(40), T.int64(128)), "float16")
        # with T.block("root"):
        for ax0, ax1, ax2, ax3 in T.grid(T.int64(1), n, T.int64(40), T.int64(128)):
            with T.block("T_reshape"):
                v_ax0, v_ax1, v_ax2, v_ax3 = T.axis.remap("SSSS", [ax0, ax1, ax2, ax3])
                T.reads(A[T.int64(0), ((v_ax2 * T.int64(128) + v_ax3) // T.int64(5120) + v_ax0 * n + v_ax1) % n, (v_ax2 * T.int64(128) + v_ax3) % T.int64(5120)])
                T.writes(T_reshape[v_ax0, v_ax1, v_ax2, v_ax3])
                T_reshape[v_ax0, v_ax1, v_ax2, v_ax3] = A[T.int64(0), ((v_ax2 * T.int64(128) + v_ax3) // T.int64(5120) + v_ax0 * n + v_ax1) % n, (v_ax2 * T.int64(128) + v_ax3) % T.int64(5120)]

    @T.prim_func(private=True)
    def reshape6(var_A: T.handle, var_T_reshape: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(40), T.int64(128)), "float16")
        T_reshape = T.match_buffer(var_T_reshape, (T.int64(1), n, T.int64(5120)), "float16")
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(5120)):
            with T.block("T_reshape"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(A[T.int64(0), (v_ax2 // T.int64(5120) + v_ax0 * n + v_ax1) % n, v_ax2 % T.int64(5120) // T.int64(128), v_ax2 % T.int64(128)])
                T.writes(T_reshape[v_ax0, v_ax1, v_ax2])
                T_reshape[v_ax0, v_ax1, v_ax2] = A[T.int64(0), (v_ax2 // T.int64(5120) + v_ax0 * n + v_ax1) % n, v_ax2 % T.int64(5120) // T.int64(128), v_ax2 % T.int64(128)]

    @T.prim_func(private=True)
    def rms_norm(var_A: T.handle, B: T.Buffer((T.int64(5120),), "float16"), var_rms_norm: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(5120)), "float16")
        rms_norm_1 = T.match_buffer(var_rms_norm, (T.int64(1), n, T.int64(5120)), "float16")
        # with T.block("root"):
        Ared_temp = T.alloc_buffer((T.int64(1), n))
        for bsz, i, k in T.grid(T.int64(1), n, T.int64(5120)):
            with T.block("Ared_temp"):
                v_bsz, v_i, v_k = T.axis.remap("SSR", [bsz, i, k])
                T.reads(A[v_bsz, v_i, v_k])
                T.writes(Ared_temp[v_bsz, v_i])
                with T.init():
                    Ared_temp[v_bsz, v_i] = T.float32(0)
                Ared_temp[v_bsz, v_i] = Ared_temp[v_bsz, v_i] + T.Cast("float32", A[v_bsz, v_i, v_k]) * T.Cast("float32", A[v_bsz, v_i, v_k])
        for bsz, i, k in T.grid(T.int64(1), n, T.int64(5120)):
            with T.block("rms_norm"):
                v_bsz, v_i, v_k = T.axis.remap("SSS", [bsz, i, k])
                T.reads(B[v_k], A[v_bsz, v_i, v_k], Ared_temp[v_bsz, v_i])
                T.writes(rms_norm_1[v_bsz, v_i, v_k])
                rms_norm_1[v_bsz, v_i, v_k] = T.Cast("float16", T.Cast("float32", B[v_k]) * (T.Cast("float32", A[v_bsz, v_i, v_k]) / T.sqrt(Ared_temp[v_bsz, v_i] * T.float32(0.00019531250000000001) + T.float32(1.0000000000000001e-05))))

    @T.prim_func(private=True)
    def rms_norm1(var_A: T.handle, B: T.Buffer((T.int64(5120),), "float16"), var_rms_norm: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        nseq = T.int64()
        A = T.match_buffer(var_A, (nseq, T.int64(1), T.int64(5120)), "float16")
        rms_norm = T.match_buffer(var_rms_norm, (nseq, T.int64(1), T.int64(5120)), "float16")
        # with T.block("root"):
        Ared_temp = T.alloc_buffer((nseq, T.int64(1)))
        for bsz, i, k in T.grid(nseq, T.int64(1), T.int64(5120)):
            with T.block("Ared_temp"):
                v_bsz, v_i, v_k = T.axis.remap("SSR", [bsz, i, k])
                T.reads(A[v_bsz, v_i, v_k])
                T.writes(Ared_temp[v_bsz, v_i])
                with T.init():
                    Ared_temp[v_bsz, v_i] = T.float32(0)
                Ared_temp[v_bsz, v_i] = Ared_temp[v_bsz, v_i] + T.Cast("float32", A[v_bsz, v_i, v_k]) * T.Cast("float32", A[v_bsz, v_i, v_k])
        for bsz, i, k in T.grid(nseq, T.int64(1), T.int64(5120)):
            with T.block("rms_norm"):
                v_bsz, v_i, v_k = T.axis.remap("SSS", [bsz, i, k])
                T.reads(B[v_k], A[v_bsz, v_i, v_k], Ared_temp[v_bsz, v_i])
                T.writes(rms_norm[v_bsz, v_i, v_k])
                rms_norm[v_bsz, v_i, v_k] = T.Cast("float16", T.Cast("float32", B[v_k]) * (T.Cast("float32", A[v_bsz, v_i, v_k]) / T.sqrt(Ared_temp[v_bsz, v_i] * T.float32(0.00019531250000000001) + T.float32(1.0000000000000001e-05))))

    @T.prim_func(private=True)
    def slice(var_A: T.handle, slice_1: T.Buffer((T.int64(1), T.int64(1), T.int64(5120)), "float16")):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(5120)), "float16")
        # with T.block("root"):
        for i, j, k in T.grid(T.int64(1), T.int64(1), T.int64(5120)):
            with T.block("slice"):
                v_i, v_j, v_k = T.axis.remap("SSS", [i, j, k])
                T.reads(A[v_i, n - T.int64(1), v_k])
                T.writes(slice_1[v_i, v_j, v_k])
                slice_1[v_i, v_j, v_k] = A[v_i, n - T.int64(1), v_k]

    @T.prim_func(private=True)
    def slice1(var_A: T.handle, var_slice: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        nseq = T.int64()
        A = T.match_buffer(var_A, (nseq, T.int64(1), T.int64(5120)), "float16")
        slice = T.match_buffer(var_slice, (nseq, T.int64(1), T.int64(5120)), "float16")
        # with T.block("root"):
        for i, j, k in T.grid(nseq, T.int64(1), T.int64(5120)):
            with T.block("slice"):
                v_i, v_j, v_k = T.axis.remap("SSS", [i, j, k])
                T.reads(A[v_i, T.int64(0), v_k])
                T.writes(slice[v_i, v_j, v_k])
                slice[v_i, v_j, v_k] = A[v_i, T.int64(0), v_k]

    @T.prim_func(private=True)
    def softmax(var_A: T.handle, var_T_softmax_norm: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        nseq, vocab_size = T.int64(), T.int64()
        A = T.match_buffer(var_A, (nseq, T.int64(1), vocab_size))
        T_softmax_norm = T.match_buffer(var_T_softmax_norm, (nseq, T.int64(1), vocab_size))
        # with T.block("root"):
        T_softmax_maxelem = T.alloc_buffer((nseq, T.int64(1)))
        T_softmax_exp = T.alloc_buffer((nseq, T.int64(1), vocab_size))
        T_softmax_expsum = T.alloc_buffer((nseq, T.int64(1)))
        for i0, i1, k in T.grid(nseq, T.int64(1), vocab_size):
            with T.block("T_softmax_maxelem"):
                v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])
                T.reads(A[v_i0, v_i1, v_k])
                T.writes(T_softmax_maxelem[v_i0, v_i1])
                with T.init():
                    T_softmax_maxelem[v_i0, v_i1] = T.float32(-3.4028234663852886e+38)
                T_softmax_maxelem[v_i0, v_i1] = T.max(T_softmax_maxelem[v_i0, v_i1], A[v_i0, v_i1, v_k])
        for i0, i1, i2 in T.grid(nseq, T.int64(1), vocab_size):
            with T.block("T_softmax_exp"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                T.reads(A[v_i0, v_i1, v_i2], T_softmax_maxelem[v_i0, v_i1])
                T.writes(T_softmax_exp[v_i0, v_i1, v_i2])
                T_softmax_exp[v_i0, v_i1, v_i2] = T.exp(A[v_i0, v_i1, v_i2] - T_softmax_maxelem[v_i0, v_i1])
        for i0, i1, k in T.grid(nseq, T.int64(1), vocab_size):
            with T.block("T_softmax_expsum"):
                v_i0, v_i1, v_k = T.axis.remap("SSR", [i0, i1, k])
                T.reads(T_softmax_exp[v_i0, v_i1, v_k])
                T.writes(T_softmax_expsum[v_i0, v_i1])
                with T.init():
                    T_softmax_expsum[v_i0, v_i1] = T.float32(0)
                T_softmax_expsum[v_i0, v_i1] = T_softmax_expsum[v_i0, v_i1] + T_softmax_exp[v_i0, v_i1, v_k]
        for i0, i1, i2 in T.grid(nseq, T.int64(1), vocab_size):
            with T.block("T_softmax_norm"):
                v_i0, v_i1, v_i2 = T.axis.remap("SSS", [i0, i1, i2])
                T.reads(T_softmax_exp[v_i0, v_i1, v_i2], T_softmax_expsum[v_i0, v_i1])
                T.writes(T_softmax_norm[v_i0, v_i1, v_i2])
                T.block_attr({"axis": 2})
                T_softmax_norm[v_i0, v_i1, v_i2] = T_softmax_exp[v_i0, v_i1, v_i2] / T_softmax_expsum[v_i0, v_i1]

    @T.prim_func(private=True)
    def split(var_A: T.handle, var_T_split: T.handle, var_T_split_1: T.handle, var_T_split_2: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        nseq = T.int64()
        A = T.match_buffer(var_A, (nseq, T.int64(1), T.int64(15360)), "float16")
        T_split = T.match_buffer(var_T_split, (nseq, T.int64(1), T.int64(5120)), "float16")
        T_split_1 = T.match_buffer(var_T_split_1, (nseq, T.int64(1), T.int64(5120)), "float16")
        T_split_2 = T.match_buffer(var_T_split_2, (nseq, T.int64(1), T.int64(5120)), "float16")
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(nseq, T.int64(1), T.int64(5120)):
            with T.block("T_split"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(A[v_ax0, v_ax1, v_ax2])
                T.writes(T_split[v_ax0, v_ax1, v_ax2])
                T_split[v_ax0, v_ax1, v_ax2] = A[v_ax0, v_ax1, v_ax2]
        for ax0, ax1, ax2 in T.grid(nseq, T.int64(1), T.int64(5120)):
            with T.block("T_split_1"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(A[v_ax0, v_ax1, v_ax2 + T.int64(5120)])
                T.writes(T_split_1[v_ax0, v_ax1, v_ax2])
                T_split_1[v_ax0, v_ax1, v_ax2] = A[v_ax0, v_ax1, v_ax2 + T.int64(5120)]
        for ax0, ax1, ax2 in T.grid(nseq, T.int64(1), T.int64(5120)):
            with T.block("T_split_2"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(A[v_ax0, v_ax1, v_ax2 + T.int64(10240)])
                T.writes(T_split_2[v_ax0, v_ax1, v_ax2])
                T_split_2[v_ax0, v_ax1, v_ax2] = A[v_ax0, v_ax1, v_ax2 + T.int64(10240)]

    @T.prim_func(private=True)
    def split2(var_A: T.handle, var_T_split: T.handle, var_T_split_1: T.handle, var_T_split_2: T.handle):
        T.func_attr({"tir.noalias": T.bool(True)})
        n = T.int64()
        A = T.match_buffer(var_A, (T.int64(1), n, T.int64(15360)), "float16")
        T_split = T.match_buffer(var_T_split, (T.int64(1), n, T.int64(5120)), "float16")
        T_split_1 = T.match_buffer(var_T_split_1, (T.int64(1), n, T.int64(5120)), "float16")
        T_split_2 = T.match_buffer(var_T_split_2, (T.int64(1), n, T.int64(5120)), "float16")
        # with T.block("root"):
        for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(5120)):
            with T.block("T_split"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(A[v_ax0, v_ax1, v_ax2])
                T.writes(T_split[v_ax0, v_ax1, v_ax2])
                T_split[v_ax0, v_ax1, v_ax2] = A[v_ax0, v_ax1, v_ax2]
        for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(5120)):
            with T.block("T_split_1"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(A[v_ax0, v_ax1, v_ax2 + T.int64(5120)])
                T.writes(T_split_1[v_ax0, v_ax1, v_ax2])
                T_split_1[v_ax0, v_ax1, v_ax2] = A[v_ax0, v_ax1, v_ax2 + T.int64(5120)]
        for ax0, ax1, ax2 in T.grid(T.int64(1), n, T.int64(5120)):
            with T.block("T_split_2"):
                v_ax0, v_ax1, v_ax2 = T.axis.remap("SSS", [ax0, ax1, ax2])
                T.reads(A[v_ax0, v_ax1, v_ax2 + T.int64(10240)])
                T.writes(T_split_2[v_ax0, v_ax1, v_ax2])
                T_split_2[v_ax0, v_ax1, v_ax2] = A[v_ax0, v_ax1, v_ax2 + T.int64(10240)]

    attention = R.ExternFunc("attention_func")
    @R.function
    def create_kv_cache(cache_config: R.Shape(["reserved_nseq", "total_seq_len", "page_size"])) -> R.Object:
        reserved_nseq = T.int64()
        total_seq_len = T.int64()
        page_size = T.int64()
        R.func_attr({"tir_var_upper_bound": {"m": 4096, "n": 4096}})
        with R.dataflow():
            gv3: R.Object = R.call_packed("vm.builtin.paged_attention_kv_cache_create", cache_config, R.prim_value(40), R.prim_value(40), R.prim_value(128), R.const(0, "float16"), sinfo_args=(R.Object,))
            R.output(gv3)
        return gv3

    @R.function
    def decode_with_embed(inputs_embeds1: R.Tensor(("nseq", 1, 5120), dtype="float16"), kv_cache: R.Object, model_params: R.Tuple(R.Tensor(("vocab_size_1", 640), dtype="uint32"), R.Tensor(("vocab_size_1", 160), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor(("vocab_size", 640), dtype="uint32"), R.Tensor(("vocab_size", 160), dtype="float16"), R.Tensor(("cache_len", 128), dtype="float16"), R.Tensor(("cache_len", 128), dtype="float16"))) -> R.Tuple(R.Tensor(("nseq", 1, "vocab_size"), dtype="float32"), R.Object):
        nseq = T.int64()
        vocab_size = T.int64()
        vocab_size_1 = T.int64()
        cache_len = T.int64()
        R.func_attr({"num_input": 2, "tir_var_upper_bound": {"m": 4096, "n": 4096}})
        cls = Module
        with R.dataflow():
            lv: R.Tensor((5120,), dtype="float16") = model_params[10]
            lv1088 = R.call_tir(cls.rms_norm1, (inputs_embeds1, lv), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv_1: R.Tensor((15360, 640), dtype="uint32") = model_params[2]
            lv1: R.Tensor((15360, 160), dtype="float16") = model_params[3]
            lv_2 = R.call_tir(cls.fused_fused_decode_NT_matmul, (lv_1, lv1, lv1088), out_sinfo=R.Tensor((nseq, 1, 15360), dtype="float16"))
            lv1091 = R.call_tir(cls.split, (lv_2,), out_sinfo=[R.Tensor((nseq, 1, 5120), dtype="float16"), R.Tensor((nseq, 1, 5120), dtype="float16"), R.Tensor((nseq, 1, 5120), dtype="float16")])
            lv1092: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1091[0]
            lv1093 = R.call_tir(cls.reshape, (lv1092,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1094: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1091[1]
            lv1095 = R.call_tir(cls.reshape, (lv1094,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1096: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1091[2]
            lv1097 = R.call_tir(cls.reshape, (lv1096,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1098: R.Object = R.call_pure_packed("vm.builtin.paged_attention_kv_cache_append", kv_cache, cls.kv_cache_transpose_append, lv1095, lv1097, R.prim_value(0), sinfo_args=(R.Object,))
            lv1099 = R.call_dps_packed("vm.builtin.paged_attention_kv_cache_attention", (lv1098, cls.attention, lv1093, R.prim_value(0), R.prim_value(1), R.prim_value(T.float32(1)), R.prim_value(10000)), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1100 = R.call_tir(cls.reshape1, (lv1099,), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv3: R.Tensor((5120, 640), dtype="uint32") = model_params[4]
            lv4: R.Tensor((5120, 160), dtype="float16") = model_params[5]
            lv80 = R.call_tir(cls.fused_fused_decode1_fused_NT_matmul1_add, (lv3, lv4, lv1100, inputs_embeds1), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv5: R.Tensor((5120,), dtype="float16") = model_params[11]
            lv1104 = R.call_tir(cls.rms_norm1, (lv80, lv5), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv7: R.Tensor((27648, 640), dtype="uint32") = model_params[6]
            lv8: R.Tensor((27648, 160), dtype="float16") = model_params[7]
            lv1_1 = R.call_tir(cls.fused_fused_decode2_NT_matmul2, (lv7, lv8, lv1104), out_sinfo=R.Tensor((nseq, 1, 27648), dtype="float16"))
            lv10 = R.call_tir(cls.fused_split1_silu_multiply, (lv1_1,), out_sinfo=R.Tensor((nseq, 1, 13824), dtype="float16"))
            lv11: R.Tensor((5120, 1728), dtype="uint32") = model_params[8]
            lv12: R.Tensor((5120, 432), dtype="float16") = model_params[9]
            lv81 = R.call_tir(cls.fused_fused_decode3_fused_NT_matmul3_add, (lv11, lv12, lv10, lv80), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv10_1: R.Tensor((5120,), dtype="float16") = model_params[20]
            lv1115 = R.call_tir(cls.rms_norm1, (lv81, lv10_1), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv15: R.Tensor((15360, 640), dtype="uint32") = model_params[12]
            lv16: R.Tensor((15360, 160), dtype="float16") = model_params[13]
            lv2 = R.call_tir(cls.fused_fused_decode_NT_matmul, (lv15, lv16, lv1115), out_sinfo=R.Tensor((nseq, 1, 15360), dtype="float16"))
            lv1118 = R.call_tir(cls.split, (lv2,), out_sinfo=[R.Tensor((nseq, 1, 5120), dtype="float16"), R.Tensor((nseq, 1, 5120), dtype="float16"), R.Tensor((nseq, 1, 5120), dtype="float16")])
            lv1119: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1118[0]
            lv1120 = R.call_tir(cls.reshape, (lv1119,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1121: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1118[1]
            lv1122 = R.call_tir(cls.reshape, (lv1121,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1123: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1118[2]
            lv1124 = R.call_tir(cls.reshape, (lv1123,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1125: R.Object = R.call_pure_packed("vm.builtin.paged_attention_kv_cache_append", lv1098, cls.kv_cache_transpose_append, lv1122, lv1124, R.prim_value(1), sinfo_args=(R.Object,))
            lv1126 = R.call_dps_packed("vm.builtin.paged_attention_kv_cache_attention", (lv1125, cls.attention, lv1120, R.prim_value(1), R.prim_value(1), R.prim_value(T.float32(1)), R.prim_value(10000)), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1127 = R.call_tir(cls.reshape1, (lv1126,), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv18: R.Tensor((5120, 640), dtype="uint32") = model_params[14]
            lv19: R.Tensor((5120, 160), dtype="float16") = model_params[15]
            lv82 = R.call_tir(cls.fused_fused_decode1_fused_NT_matmul1_add, (lv18, lv19, lv1127, lv81), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv15_1: R.Tensor((5120,), dtype="float16") = model_params[21]
            lv1131 = R.call_tir(cls.rms_norm1, (lv82, lv15_1), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv22: R.Tensor((27648, 640), dtype="uint32") = model_params[16]
            lv23: R.Tensor((27648, 160), dtype="float16") = model_params[17]
            lv3_1 = R.call_tir(cls.fused_fused_decode2_NT_matmul2, (lv22, lv23, lv1131), out_sinfo=R.Tensor((nseq, 1, 27648), dtype="float16"))
            lv25 = R.call_tir(cls.fused_split1_silu_multiply, (lv3_1,), out_sinfo=R.Tensor((nseq, 1, 13824), dtype="float16"))
            lv26: R.Tensor((5120, 1728), dtype="uint32") = model_params[18]
            lv27: R.Tensor((5120, 432), dtype="float16") = model_params[19]
            lv83 = R.call_tir(cls.fused_fused_decode3_fused_NT_matmul3_add, (lv26, lv27, lv25, lv82), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv20: R.Tensor((5120,), dtype="float16") = model_params[30]
            lv1142 = R.call_tir(cls.rms_norm1, (lv83, lv20), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv30: R.Tensor((15360, 640), dtype="uint32") = model_params[22]
            lv31: R.Tensor((15360, 160), dtype="float16") = model_params[23]
            lv4_1 = R.call_tir(cls.fused_fused_decode_NT_matmul, (lv30, lv31, lv1142), out_sinfo=R.Tensor((nseq, 1, 15360), dtype="float16"))
            lv1145 = R.call_tir(cls.split, (lv4_1,), out_sinfo=[R.Tensor((nseq, 1, 5120), dtype="float16"), R.Tensor((nseq, 1, 5120), dtype="float16"), R.Tensor((nseq, 1, 5120), dtype="float16")])
            lv1146: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1145[0]
            lv1147 = R.call_tir(cls.reshape, (lv1146,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1148: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1145[1]
            lv1149 = R.call_tir(cls.reshape, (lv1148,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1150: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1145[2]
            lv1151 = R.call_tir(cls.reshape, (lv1150,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1152: R.Object = R.call_pure_packed("vm.builtin.paged_attention_kv_cache_append", lv1125, cls.kv_cache_transpose_append, lv1149, lv1151, R.prim_value(2), sinfo_args=(R.Object,))
            lv1153 = R.call_dps_packed("vm.builtin.paged_attention_kv_cache_attention", (lv1152, cls.attention, lv1147, R.prim_value(2), R.prim_value(1), R.prim_value(T.float32(1)), R.prim_value(10000)), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1154 = R.call_tir(cls.reshape1, (lv1153,), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv33: R.Tensor((5120, 640), dtype="uint32") = model_params[24]
            lv34: R.Tensor((5120, 160), dtype="float16") = model_params[25]
            lv84 = R.call_tir(cls.fused_fused_decode1_fused_NT_matmul1_add, (lv33, lv34, lv1154, lv83), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv25_1: R.Tensor((5120,), dtype="float16") = model_params[31]
            lv1158 = R.call_tir(cls.rms_norm1, (lv84, lv25_1), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv37: R.Tensor((27648, 640), dtype="uint32") = model_params[26]
            lv38: R.Tensor((27648, 160), dtype="float16") = model_params[27]
            lv5_1 = R.call_tir(cls.fused_fused_decode2_NT_matmul2, (lv37, lv38, lv1158), out_sinfo=R.Tensor((nseq, 1, 27648), dtype="float16"))
            lv40 = R.call_tir(cls.fused_split1_silu_multiply, (lv5_1,), out_sinfo=R.Tensor((nseq, 1, 13824), dtype="float16"))
            lv41: R.Tensor((5120, 1728), dtype="uint32") = model_params[28]
            lv42: R.Tensor((5120, 432), dtype="float16") = model_params[29]
            lv85 = R.call_tir(cls.fused_fused_decode3_fused_NT_matmul3_add, (lv41, lv42, lv40, lv84), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv30_1: R.Tensor((5120,), dtype="float16") = model_params[40]
            lv1169 = R.call_tir(cls.rms_norm1, (lv85, lv30_1), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv45: R.Tensor((15360, 640), dtype="uint32") = model_params[32]
            lv46: R.Tensor((15360, 160), dtype="float16") = model_params[33]
            lv6 = R.call_tir(cls.fused_fused_decode_NT_matmul, (lv45, lv46, lv1169), out_sinfo=R.Tensor((nseq, 1, 15360), dtype="float16"))
            lv1172 = R.call_tir(cls.split, (lv6,), out_sinfo=[R.Tensor((nseq, 1, 5120), dtype="float16"), R.Tensor((nseq, 1, 5120), dtype="float16"), R.Tensor((nseq, 1, 5120), dtype="float16")])
            lv1173: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1172[0]
            lv1174 = R.call_tir(cls.reshape, (lv1173,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1175: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1172[1]
            lv1176 = R.call_tir(cls.reshape, (lv1175,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1177: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1172[2]
            lv1178 = R.call_tir(cls.reshape, (lv1177,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1179: R.Object = R.call_pure_packed("vm.builtin.paged_attention_kv_cache_append", lv1152, cls.kv_cache_transpose_append, lv1176, lv1178, R.prim_value(3), sinfo_args=(R.Object,))
            lv1180 = R.call_dps_packed("vm.builtin.paged_attention_kv_cache_attention", (lv1179, cls.attention, lv1174, R.prim_value(3), R.prim_value(1), R.prim_value(T.float32(1)), R.prim_value(10000)), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1181 = R.call_tir(cls.reshape1, (lv1180,), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv48: R.Tensor((5120, 640), dtype="uint32") = model_params[34]
            lv49: R.Tensor((5120, 160), dtype="float16") = model_params[35]
            lv86 = R.call_tir(cls.fused_fused_decode1_fused_NT_matmul1_add, (lv48, lv49, lv1181, lv85), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv35: R.Tensor((5120,), dtype="float16") = model_params[41]
            lv1185 = R.call_tir(cls.rms_norm1, (lv86, lv35), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv52: R.Tensor((27648, 640), dtype="uint32") = model_params[36]
            lv53: R.Tensor((27648, 160), dtype="float16") = model_params[37]
            lv7_1 = R.call_tir(cls.fused_fused_decode2_NT_matmul2, (lv52, lv53, lv1185), out_sinfo=R.Tensor((nseq, 1, 27648), dtype="float16"))
            lv55 = R.call_tir(cls.fused_split1_silu_multiply, (lv7_1,), out_sinfo=R.Tensor((nseq, 1, 13824), dtype="float16"))
            lv56: R.Tensor((5120, 1728), dtype="uint32") = model_params[38]
            lv57: R.Tensor((5120, 432), dtype="float16") = model_params[39]
            lv87 = R.call_tir(cls.fused_fused_decode3_fused_NT_matmul3_add, (lv56, lv57, lv55, lv86), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv40_1: R.Tensor((5120,), dtype="float16") = model_params[50]
            lv1196 = R.call_tir(cls.rms_norm1, (lv87, lv40_1), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv60: R.Tensor((15360, 640), dtype="uint32") = model_params[42]
            lv61: R.Tensor((15360, 160), dtype="float16") = model_params[43]
            lv8_1 = R.call_tir(cls.fused_fused_decode_NT_matmul, (lv60, lv61, lv1196), out_sinfo=R.Tensor((nseq, 1, 15360), dtype="float16"))
            lv1199 = R.call_tir(cls.split, (lv8_1,), out_sinfo=[R.Tensor((nseq, 1, 5120), dtype="float16"), R.Tensor((nseq, 1, 5120), dtype="float16"), R.Tensor((nseq, 1, 5120), dtype="float16")])
            lv1200: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1199[0]
            lv1201 = R.call_tir(cls.reshape, (lv1200,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1202: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1199[1]
            lv1203 = R.call_tir(cls.reshape, (lv1202,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1204: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1199[2]
            lv1205 = R.call_tir(cls.reshape, (lv1204,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1206: R.Object = R.call_pure_packed("vm.builtin.paged_attention_kv_cache_append", lv1179, cls.kv_cache_transpose_append, lv1203, lv1205, R.prim_value(4), sinfo_args=(R.Object,))
            lv1207 = R.call_dps_packed("vm.builtin.paged_attention_kv_cache_attention", (lv1206, cls.attention, lv1201, R.prim_value(4), R.prim_value(1), R.prim_value(T.float32(1)), R.prim_value(10000)), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1208 = R.call_tir(cls.reshape1, (lv1207,), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv63: R.Tensor((5120, 640), dtype="uint32") = model_params[44]
            lv64: R.Tensor((5120, 160), dtype="float16") = model_params[45]
            lv88 = R.call_tir(cls.fused_fused_decode1_fused_NT_matmul1_add, (lv63, lv64, lv1208, lv87), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv45_1: R.Tensor((5120,), dtype="float16") = model_params[51]
            lv1212 = R.call_tir(cls.rms_norm1, (lv88, lv45_1), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv67: R.Tensor((27648, 640), dtype="uint32") = model_params[46]
            lv68: R.Tensor((27648, 160), dtype="float16") = model_params[47]
            lv9 = R.call_tir(cls.fused_fused_decode2_NT_matmul2, (lv67, lv68, lv1212), out_sinfo=R.Tensor((nseq, 1, 27648), dtype="float16"))
            lv70 = R.call_tir(cls.fused_split1_silu_multiply, (lv9,), out_sinfo=R.Tensor((nseq, 1, 13824), dtype="float16"))
            lv71: R.Tensor((5120, 1728), dtype="uint32") = model_params[48]
            lv72: R.Tensor((5120, 432), dtype="float16") = model_params[49]
            lv89 = R.call_tir(cls.fused_fused_decode3_fused_NT_matmul3_add, (lv71, lv72, lv70, lv88), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv50: R.Tensor((5120,), dtype="float16") = model_params[60]
            lv1223 = R.call_tir(cls.rms_norm1, (lv89, lv50), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv75: R.Tensor((15360, 640), dtype="uint32") = model_params[52]
            lv76: R.Tensor((15360, 160), dtype="float16") = model_params[53]
            lv10_2 = R.call_tir(cls.fused_fused_decode_NT_matmul, (lv75, lv76, lv1223), out_sinfo=R.Tensor((nseq, 1, 15360), dtype="float16"))
            lv1226 = R.call_tir(cls.split, (lv10_2,), out_sinfo=[R.Tensor((nseq, 1, 5120), dtype="float16"), R.Tensor((nseq, 1, 5120), dtype="float16"), R.Tensor((nseq, 1, 5120), dtype="float16")])
            lv1227: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1226[0]
            lv1228 = R.call_tir(cls.reshape, (lv1227,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1229: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1226[1]
            lv1230 = R.call_tir(cls.reshape, (lv1229,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1231: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1226[2]
            lv1232 = R.call_tir(cls.reshape, (lv1231,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1233: R.Object = R.call_pure_packed("vm.builtin.paged_attention_kv_cache_append", lv1206, cls.kv_cache_transpose_append, lv1230, lv1232, R.prim_value(5), sinfo_args=(R.Object,))
            lv1234 = R.call_dps_packed("vm.builtin.paged_attention_kv_cache_attention", (lv1233, cls.attention, lv1228, R.prim_value(5), R.prim_value(1), R.prim_value(T.float32(1)), R.prim_value(10000)), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1235 = R.call_tir(cls.reshape1, (lv1234,), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv78: R.Tensor((5120, 640), dtype="uint32") = model_params[54]
            lv79: R.Tensor((5120, 160), dtype="float16") = model_params[55]
            lv90 = R.call_tir(cls.fused_fused_decode1_fused_NT_matmul1_add, (lv78, lv79, lv1235, lv89), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv55_1: R.Tensor((5120,), dtype="float16") = model_params[61]
            lv1239 = R.call_tir(cls.rms_norm1, (lv90, lv55_1), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv82_1: R.Tensor((27648, 640), dtype="uint32") = model_params[56]
            lv83_1: R.Tensor((27648, 160), dtype="float16") = model_params[57]
            lv11_1 = R.call_tir(cls.fused_fused_decode2_NT_matmul2, (lv82_1, lv83_1, lv1239), out_sinfo=R.Tensor((nseq, 1, 27648), dtype="float16"))
            lv85_1 = R.call_tir(cls.fused_split1_silu_multiply, (lv11_1,), out_sinfo=R.Tensor((nseq, 1, 13824), dtype="float16"))
            lv86_1: R.Tensor((5120, 1728), dtype="uint32") = model_params[58]
            lv87_1: R.Tensor((5120, 432), dtype="float16") = model_params[59]
            lv91 = R.call_tir(cls.fused_fused_decode3_fused_NT_matmul3_add, (lv86_1, lv87_1, lv85_1, lv90), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv60_1: R.Tensor((5120,), dtype="float16") = model_params[70]
            lv1250 = R.call_tir(cls.rms_norm1, (lv91, lv60_1), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv90_1: R.Tensor((15360, 640), dtype="uint32") = model_params[62]
            lv91_1: R.Tensor((15360, 160), dtype="float16") = model_params[63]
            lv12_1 = R.call_tir(cls.fused_fused_decode_NT_matmul, (lv90_1, lv91_1, lv1250), out_sinfo=R.Tensor((nseq, 1, 15360), dtype="float16"))
            lv1253 = R.call_tir(cls.split, (lv12_1,), out_sinfo=[R.Tensor((nseq, 1, 5120), dtype="float16"), R.Tensor((nseq, 1, 5120), dtype="float16"), R.Tensor((nseq, 1, 5120), dtype="float16")])
            lv1254: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1253[0]
            lv1255 = R.call_tir(cls.reshape, (lv1254,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1256: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1253[1]
            lv1257 = R.call_tir(cls.reshape, (lv1256,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1258: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1253[2]
            lv1259 = R.call_tir(cls.reshape, (lv1258,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1260: R.Object = R.call_pure_packed("vm.builtin.paged_attention_kv_cache_append", lv1233, cls.kv_cache_transpose_append, lv1257, lv1259, R.prim_value(6), sinfo_args=(R.Object,))
            lv1261 = R.call_dps_packed("vm.builtin.paged_attention_kv_cache_attention", (lv1260, cls.attention, lv1255, R.prim_value(6), R.prim_value(1), R.prim_value(T.float32(1)), R.prim_value(10000)), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1262 = R.call_tir(cls.reshape1, (lv1261,), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv93: R.Tensor((5120, 640), dtype="uint32") = model_params[64]
            lv94: R.Tensor((5120, 160), dtype="float16") = model_params[65]
            lv92 = R.call_tir(cls.fused_fused_decode1_fused_NT_matmul1_add, (lv93, lv94, lv1262, lv91), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv65: R.Tensor((5120,), dtype="float16") = model_params[71]
            lv1266 = R.call_tir(cls.rms_norm1, (lv92, lv65), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv97: R.Tensor((27648, 640), dtype="uint32") = model_params[66]
            lv98: R.Tensor((27648, 160), dtype="float16") = model_params[67]
            lv13 = R.call_tir(cls.fused_fused_decode2_NT_matmul2, (lv97, lv98, lv1266), out_sinfo=R.Tensor((nseq, 1, 27648), dtype="float16"))
            lv100 = R.call_tir(cls.fused_split1_silu_multiply, (lv13,), out_sinfo=R.Tensor((nseq, 1, 13824), dtype="float16"))
            lv101: R.Tensor((5120, 1728), dtype="uint32") = model_params[68]
            lv102: R.Tensor((5120, 432), dtype="float16") = model_params[69]
            lv93_1 = R.call_tir(cls.fused_fused_decode3_fused_NT_matmul3_add, (lv101, lv102, lv100, lv92), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv70_1: R.Tensor((5120,), dtype="float16") = model_params[80]
            lv1277 = R.call_tir(cls.rms_norm1, (lv93_1, lv70_1), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv105: R.Tensor((15360, 640), dtype="uint32") = model_params[72]
            lv106: R.Tensor((15360, 160), dtype="float16") = model_params[73]
            lv14 = R.call_tir(cls.fused_fused_decode_NT_matmul, (lv105, lv106, lv1277), out_sinfo=R.Tensor((nseq, 1, 15360), dtype="float16"))
            lv1280 = R.call_tir(cls.split, (lv14,), out_sinfo=[R.Tensor((nseq, 1, 5120), dtype="float16"), R.Tensor((nseq, 1, 5120), dtype="float16"), R.Tensor((nseq, 1, 5120), dtype="float16")])
            lv1281: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1280[0]
            lv1282 = R.call_tir(cls.reshape, (lv1281,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1283: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1280[1]
            lv1284 = R.call_tir(cls.reshape, (lv1283,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1285: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1280[2]
            lv1286 = R.call_tir(cls.reshape, (lv1285,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1287: R.Object = R.call_pure_packed("vm.builtin.paged_attention_kv_cache_append", lv1260, cls.kv_cache_transpose_append, lv1284, lv1286, R.prim_value(7), sinfo_args=(R.Object,))
            lv1288 = R.call_dps_packed("vm.builtin.paged_attention_kv_cache_attention", (lv1287, cls.attention, lv1282, R.prim_value(7), R.prim_value(1), R.prim_value(T.float32(1)), R.prim_value(10000)), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1289 = R.call_tir(cls.reshape1, (lv1288,), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv108: R.Tensor((5120, 640), dtype="uint32") = model_params[74]
            lv109: R.Tensor((5120, 160), dtype="float16") = model_params[75]
            lv94_1 = R.call_tir(cls.fused_fused_decode1_fused_NT_matmul1_add, (lv108, lv109, lv1289, lv93_1), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv75_1: R.Tensor((5120,), dtype="float16") = model_params[81]
            lv1293 = R.call_tir(cls.rms_norm1, (lv94_1, lv75_1), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv112: R.Tensor((27648, 640), dtype="uint32") = model_params[76]
            lv113: R.Tensor((27648, 160), dtype="float16") = model_params[77]
            lv15_2 = R.call_tir(cls.fused_fused_decode2_NT_matmul2, (lv112, lv113, lv1293), out_sinfo=R.Tensor((nseq, 1, 27648), dtype="float16"))
            lv115 = R.call_tir(cls.fused_split1_silu_multiply, (lv15_2,), out_sinfo=R.Tensor((nseq, 1, 13824), dtype="float16"))
            lv116: R.Tensor((5120, 1728), dtype="uint32") = model_params[78]
            lv117: R.Tensor((5120, 432), dtype="float16") = model_params[79]
            lv95 = R.call_tir(cls.fused_fused_decode3_fused_NT_matmul3_add, (lv116, lv117, lv115, lv94_1), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv80_1: R.Tensor((5120,), dtype="float16") = model_params[90]
            lv1304 = R.call_tir(cls.rms_norm1, (lv95, lv80_1), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv120: R.Tensor((15360, 640), dtype="uint32") = model_params[82]
            lv121: R.Tensor((15360, 160), dtype="float16") = model_params[83]
            lv16_1 = R.call_tir(cls.fused_fused_decode_NT_matmul, (lv120, lv121, lv1304), out_sinfo=R.Tensor((nseq, 1, 15360), dtype="float16"))
            lv1307 = R.call_tir(cls.split, (lv16_1,), out_sinfo=[R.Tensor((nseq, 1, 5120), dtype="float16"), R.Tensor((nseq, 1, 5120), dtype="float16"), R.Tensor((nseq, 1, 5120), dtype="float16")])
            lv1308: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1307[0]
            lv1309 = R.call_tir(cls.reshape, (lv1308,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1310: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1307[1]
            lv1311 = R.call_tir(cls.reshape, (lv1310,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1312: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1307[2]
            lv1313 = R.call_tir(cls.reshape, (lv1312,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1314: R.Object = R.call_pure_packed("vm.builtin.paged_attention_kv_cache_append", lv1287, cls.kv_cache_transpose_append, lv1311, lv1313, R.prim_value(8), sinfo_args=(R.Object,))
            lv1315 = R.call_dps_packed("vm.builtin.paged_attention_kv_cache_attention", (lv1314, cls.attention, lv1309, R.prim_value(8), R.prim_value(1), R.prim_value(T.float32(1)), R.prim_value(10000)), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1316 = R.call_tir(cls.reshape1, (lv1315,), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv123: R.Tensor((5120, 640), dtype="uint32") = model_params[84]
            lv124: R.Tensor((5120, 160), dtype="float16") = model_params[85]
            lv96 = R.call_tir(cls.fused_fused_decode1_fused_NT_matmul1_add, (lv123, lv124, lv1316, lv95), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv85_2: R.Tensor((5120,), dtype="float16") = model_params[91]
            lv1320 = R.call_tir(cls.rms_norm1, (lv96, lv85_2), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv127: R.Tensor((27648, 640), dtype="uint32") = model_params[86]
            lv128: R.Tensor((27648, 160), dtype="float16") = model_params[87]
            lv17 = R.call_tir(cls.fused_fused_decode2_NT_matmul2, (lv127, lv128, lv1320), out_sinfo=R.Tensor((nseq, 1, 27648), dtype="float16"))
            lv130 = R.call_tir(cls.fused_split1_silu_multiply, (lv17,), out_sinfo=R.Tensor((nseq, 1, 13824), dtype="float16"))
            lv131: R.Tensor((5120, 1728), dtype="uint32") = model_params[88]
            lv132: R.Tensor((5120, 432), dtype="float16") = model_params[89]
            lv97_1 = R.call_tir(cls.fused_fused_decode3_fused_NT_matmul3_add, (lv131, lv132, lv130, lv96), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv90_2: R.Tensor((5120,), dtype="float16") = model_params[100]
            lv1331 = R.call_tir(cls.rms_norm1, (lv97_1, lv90_2), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv135: R.Tensor((15360, 640), dtype="uint32") = model_params[92]
            lv136: R.Tensor((15360, 160), dtype="float16") = model_params[93]
            lv18_1 = R.call_tir(cls.fused_fused_decode_NT_matmul, (lv135, lv136, lv1331), out_sinfo=R.Tensor((nseq, 1, 15360), dtype="float16"))
            lv1334 = R.call_tir(cls.split, (lv18_1,), out_sinfo=[R.Tensor((nseq, 1, 5120), dtype="float16"), R.Tensor((nseq, 1, 5120), dtype="float16"), R.Tensor((nseq, 1, 5120), dtype="float16")])
            lv1335: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1334[0]
            lv1336 = R.call_tir(cls.reshape, (lv1335,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1337: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1334[1]
            lv1338 = R.call_tir(cls.reshape, (lv1337,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1339: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1334[2]
            lv1340 = R.call_tir(cls.reshape, (lv1339,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1341: R.Object = R.call_pure_packed("vm.builtin.paged_attention_kv_cache_append", lv1314, cls.kv_cache_transpose_append, lv1338, lv1340, R.prim_value(9), sinfo_args=(R.Object,))
            lv1342 = R.call_dps_packed("vm.builtin.paged_attention_kv_cache_attention", (lv1341, cls.attention, lv1336, R.prim_value(9), R.prim_value(1), R.prim_value(T.float32(1)), R.prim_value(10000)), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1343 = R.call_tir(cls.reshape1, (lv1342,), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv138: R.Tensor((5120, 640), dtype="uint32") = model_params[94]
            lv139: R.Tensor((5120, 160), dtype="float16") = model_params[95]
            lv98_1 = R.call_tir(cls.fused_fused_decode1_fused_NT_matmul1_add, (lv138, lv139, lv1343, lv97_1), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv95_1: R.Tensor((5120,), dtype="float16") = model_params[101]
            lv1347 = R.call_tir(cls.rms_norm1, (lv98_1, lv95_1), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv142: R.Tensor((27648, 640), dtype="uint32") = model_params[96]
            lv143: R.Tensor((27648, 160), dtype="float16") = model_params[97]
            lv19_1 = R.call_tir(cls.fused_fused_decode2_NT_matmul2, (lv142, lv143, lv1347), out_sinfo=R.Tensor((nseq, 1, 27648), dtype="float16"))
            lv145 = R.call_tir(cls.fused_split1_silu_multiply, (lv19_1,), out_sinfo=R.Tensor((nseq, 1, 13824), dtype="float16"))
            lv146: R.Tensor((5120, 1728), dtype="uint32") = model_params[98]
            lv147: R.Tensor((5120, 432), dtype="float16") = model_params[99]
            lv99 = R.call_tir(cls.fused_fused_decode3_fused_NT_matmul3_add, (lv146, lv147, lv145, lv98_1), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv100_1: R.Tensor((5120,), dtype="float16") = model_params[110]
            lv1358 = R.call_tir(cls.rms_norm1, (lv99, lv100_1), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv150: R.Tensor((15360, 640), dtype="uint32") = model_params[102]
            lv151: R.Tensor((15360, 160), dtype="float16") = model_params[103]
            lv20_1 = R.call_tir(cls.fused_fused_decode_NT_matmul, (lv150, lv151, lv1358), out_sinfo=R.Tensor((nseq, 1, 15360), dtype="float16"))
            lv1361 = R.call_tir(cls.split, (lv20_1,), out_sinfo=[R.Tensor((nseq, 1, 5120), dtype="float16"), R.Tensor((nseq, 1, 5120), dtype="float16"), R.Tensor((nseq, 1, 5120), dtype="float16")])
            lv1362: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1361[0]
            lv1363 = R.call_tir(cls.reshape, (lv1362,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1364: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1361[1]
            lv1365 = R.call_tir(cls.reshape, (lv1364,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1366: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1361[2]
            lv1367 = R.call_tir(cls.reshape, (lv1366,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1368: R.Object = R.call_pure_packed("vm.builtin.paged_attention_kv_cache_append", lv1341, cls.kv_cache_transpose_append, lv1365, lv1367, R.prim_value(10), sinfo_args=(R.Object,))
            lv1369 = R.call_dps_packed("vm.builtin.paged_attention_kv_cache_attention", (lv1368, cls.attention, lv1363, R.prim_value(10), R.prim_value(1), R.prim_value(T.float32(1)), R.prim_value(10000)), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1370 = R.call_tir(cls.reshape1, (lv1369,), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv153: R.Tensor((5120, 640), dtype="uint32") = model_params[104]
            lv154: R.Tensor((5120, 160), dtype="float16") = model_params[105]
            lv100_2 = R.call_tir(cls.fused_fused_decode1_fused_NT_matmul1_add, (lv153, lv154, lv1370, lv99), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv105_1: R.Tensor((5120,), dtype="float16") = model_params[111]
            lv1374 = R.call_tir(cls.rms_norm1, (lv100_2, lv105_1), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv157: R.Tensor((27648, 640), dtype="uint32") = model_params[106]
            lv158: R.Tensor((27648, 160), dtype="float16") = model_params[107]
            lv21 = R.call_tir(cls.fused_fused_decode2_NT_matmul2, (lv157, lv158, lv1374), out_sinfo=R.Tensor((nseq, 1, 27648), dtype="float16"))
            lv160 = R.call_tir(cls.fused_split1_silu_multiply, (lv21,), out_sinfo=R.Tensor((nseq, 1, 13824), dtype="float16"))
            lv161: R.Tensor((5120, 1728), dtype="uint32") = model_params[108]
            lv162: R.Tensor((5120, 432), dtype="float16") = model_params[109]
            lv101_1 = R.call_tir(cls.fused_fused_decode3_fused_NT_matmul3_add, (lv161, lv162, lv160, lv100_2), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv110: R.Tensor((5120,), dtype="float16") = model_params[120]
            lv1385 = R.call_tir(cls.rms_norm1, (lv101_1, lv110), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv165: R.Tensor((15360, 640), dtype="uint32") = model_params[112]
            lv166: R.Tensor((15360, 160), dtype="float16") = model_params[113]
            lv22_1 = R.call_tir(cls.fused_fused_decode_NT_matmul, (lv165, lv166, lv1385), out_sinfo=R.Tensor((nseq, 1, 15360), dtype="float16"))
            lv1388 = R.call_tir(cls.split, (lv22_1,), out_sinfo=[R.Tensor((nseq, 1, 5120), dtype="float16"), R.Tensor((nseq, 1, 5120), dtype="float16"), R.Tensor((nseq, 1, 5120), dtype="float16")])
            lv1389: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1388[0]
            lv1390 = R.call_tir(cls.reshape, (lv1389,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1391: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1388[1]
            lv1392 = R.call_tir(cls.reshape, (lv1391,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1393: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1388[2]
            lv1394 = R.call_tir(cls.reshape, (lv1393,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1395: R.Object = R.call_pure_packed("vm.builtin.paged_attention_kv_cache_append", lv1368, cls.kv_cache_transpose_append, lv1392, lv1394, R.prim_value(11), sinfo_args=(R.Object,))
            lv1396 = R.call_dps_packed("vm.builtin.paged_attention_kv_cache_attention", (lv1395, cls.attention, lv1390, R.prim_value(11), R.prim_value(1), R.prim_value(T.float32(1)), R.prim_value(10000)), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1397 = R.call_tir(cls.reshape1, (lv1396,), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv168: R.Tensor((5120, 640), dtype="uint32") = model_params[114]
            lv169: R.Tensor((5120, 160), dtype="float16") = model_params[115]
            lv102_1 = R.call_tir(cls.fused_fused_decode1_fused_NT_matmul1_add, (lv168, lv169, lv1397, lv101_1), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv115_1: R.Tensor((5120,), dtype="float16") = model_params[121]
            lv1401 = R.call_tir(cls.rms_norm1, (lv102_1, lv115_1), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv172: R.Tensor((27648, 640), dtype="uint32") = model_params[116]
            lv173: R.Tensor((27648, 160), dtype="float16") = model_params[117]
            lv23_1 = R.call_tir(cls.fused_fused_decode2_NT_matmul2, (lv172, lv173, lv1401), out_sinfo=R.Tensor((nseq, 1, 27648), dtype="float16"))
            lv175 = R.call_tir(cls.fused_split1_silu_multiply, (lv23_1,), out_sinfo=R.Tensor((nseq, 1, 13824), dtype="float16"))
            lv176: R.Tensor((5120, 1728), dtype="uint32") = model_params[118]
            lv177: R.Tensor((5120, 432), dtype="float16") = model_params[119]
            lv103 = R.call_tir(cls.fused_fused_decode3_fused_NT_matmul3_add, (lv176, lv177, lv175, lv102_1), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv120_1: R.Tensor((5120,), dtype="float16") = model_params[130]
            lv1412 = R.call_tir(cls.rms_norm1, (lv103, lv120_1), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv180: R.Tensor((15360, 640), dtype="uint32") = model_params[122]
            lv181: R.Tensor((15360, 160), dtype="float16") = model_params[123]
            lv24 = R.call_tir(cls.fused_fused_decode_NT_matmul, (lv180, lv181, lv1412), out_sinfo=R.Tensor((nseq, 1, 15360), dtype="float16"))
            lv1415 = R.call_tir(cls.split, (lv24,), out_sinfo=[R.Tensor((nseq, 1, 5120), dtype="float16"), R.Tensor((nseq, 1, 5120), dtype="float16"), R.Tensor((nseq, 1, 5120), dtype="float16")])
            lv1416: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1415[0]
            lv1417 = R.call_tir(cls.reshape, (lv1416,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1418: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1415[1]
            lv1419 = R.call_tir(cls.reshape, (lv1418,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1420: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1415[2]
            lv1421 = R.call_tir(cls.reshape, (lv1420,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1422: R.Object = R.call_pure_packed("vm.builtin.paged_attention_kv_cache_append", lv1395, cls.kv_cache_transpose_append, lv1419, lv1421, R.prim_value(12), sinfo_args=(R.Object,))
            lv1423 = R.call_dps_packed("vm.builtin.paged_attention_kv_cache_attention", (lv1422, cls.attention, lv1417, R.prim_value(12), R.prim_value(1), R.prim_value(T.float32(1)), R.prim_value(10000)), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1424 = R.call_tir(cls.reshape1, (lv1423,), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv183: R.Tensor((5120, 640), dtype="uint32") = model_params[124]
            lv184: R.Tensor((5120, 160), dtype="float16") = model_params[125]
            lv104 = R.call_tir(cls.fused_fused_decode1_fused_NT_matmul1_add, (lv183, lv184, lv1424, lv103), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv125: R.Tensor((5120,), dtype="float16") = model_params[131]
            lv1428 = R.call_tir(cls.rms_norm1, (lv104, lv125), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv187: R.Tensor((27648, 640), dtype="uint32") = model_params[126]
            lv188: R.Tensor((27648, 160), dtype="float16") = model_params[127]
            lv25_2 = R.call_tir(cls.fused_fused_decode2_NT_matmul2, (lv187, lv188, lv1428), out_sinfo=R.Tensor((nseq, 1, 27648), dtype="float16"))
            lv190 = R.call_tir(cls.fused_split1_silu_multiply, (lv25_2,), out_sinfo=R.Tensor((nseq, 1, 13824), dtype="float16"))
            lv191: R.Tensor((5120, 1728), dtype="uint32") = model_params[128]
            lv192: R.Tensor((5120, 432), dtype="float16") = model_params[129]
            lv105_2 = R.call_tir(cls.fused_fused_decode3_fused_NT_matmul3_add, (lv191, lv192, lv190, lv104), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv130_1: R.Tensor((5120,), dtype="float16") = model_params[140]
            lv1439 = R.call_tir(cls.rms_norm1, (lv105_2, lv130_1), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv195: R.Tensor((15360, 640), dtype="uint32") = model_params[132]
            lv196: R.Tensor((15360, 160), dtype="float16") = model_params[133]
            lv26_1 = R.call_tir(cls.fused_fused_decode_NT_matmul, (lv195, lv196, lv1439), out_sinfo=R.Tensor((nseq, 1, 15360), dtype="float16"))
            lv1442 = R.call_tir(cls.split, (lv26_1,), out_sinfo=[R.Tensor((nseq, 1, 5120), dtype="float16"), R.Tensor((nseq, 1, 5120), dtype="float16"), R.Tensor((nseq, 1, 5120), dtype="float16")])
            lv1443: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1442[0]
            lv1444 = R.call_tir(cls.reshape, (lv1443,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1445: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1442[1]
            lv1446 = R.call_tir(cls.reshape, (lv1445,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1447: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1442[2]
            lv1448 = R.call_tir(cls.reshape, (lv1447,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1449: R.Object = R.call_pure_packed("vm.builtin.paged_attention_kv_cache_append", lv1422, cls.kv_cache_transpose_append, lv1446, lv1448, R.prim_value(13), sinfo_args=(R.Object,))
            lv1450 = R.call_dps_packed("vm.builtin.paged_attention_kv_cache_attention", (lv1449, cls.attention, lv1444, R.prim_value(13), R.prim_value(1), R.prim_value(T.float32(1)), R.prim_value(10000)), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1451 = R.call_tir(cls.reshape1, (lv1450,), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv198: R.Tensor((5120, 640), dtype="uint32") = model_params[134]
            lv199: R.Tensor((5120, 160), dtype="float16") = model_params[135]
            lv106_1 = R.call_tir(cls.fused_fused_decode1_fused_NT_matmul1_add, (lv198, lv199, lv1451, lv105_2), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv135_1: R.Tensor((5120,), dtype="float16") = model_params[141]
            lv1455 = R.call_tir(cls.rms_norm1, (lv106_1, lv135_1), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv202: R.Tensor((27648, 640), dtype="uint32") = model_params[136]
            lv203: R.Tensor((27648, 160), dtype="float16") = model_params[137]
            lv27_1 = R.call_tir(cls.fused_fused_decode2_NT_matmul2, (lv202, lv203, lv1455), out_sinfo=R.Tensor((nseq, 1, 27648), dtype="float16"))
            lv205 = R.call_tir(cls.fused_split1_silu_multiply, (lv27_1,), out_sinfo=R.Tensor((nseq, 1, 13824), dtype="float16"))
            lv206: R.Tensor((5120, 1728), dtype="uint32") = model_params[138]
            lv207: R.Tensor((5120, 432), dtype="float16") = model_params[139]
            lv107 = R.call_tir(cls.fused_fused_decode3_fused_NT_matmul3_add, (lv206, lv207, lv205, lv106_1), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv140: R.Tensor((5120,), dtype="float16") = model_params[150]
            lv1466 = R.call_tir(cls.rms_norm1, (lv107, lv140), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv210: R.Tensor((15360, 640), dtype="uint32") = model_params[142]
            lv211: R.Tensor((15360, 160), dtype="float16") = model_params[143]
            lv28 = R.call_tir(cls.fused_fused_decode_NT_matmul, (lv210, lv211, lv1466), out_sinfo=R.Tensor((nseq, 1, 15360), dtype="float16"))
            lv1469 = R.call_tir(cls.split, (lv28,), out_sinfo=[R.Tensor((nseq, 1, 5120), dtype="float16"), R.Tensor((nseq, 1, 5120), dtype="float16"), R.Tensor((nseq, 1, 5120), dtype="float16")])
            lv1470: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1469[0]
            lv1471 = R.call_tir(cls.reshape, (lv1470,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1472: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1469[1]
            lv1473 = R.call_tir(cls.reshape, (lv1472,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1474: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1469[2]
            lv1475 = R.call_tir(cls.reshape, (lv1474,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1476: R.Object = R.call_pure_packed("vm.builtin.paged_attention_kv_cache_append", lv1449, cls.kv_cache_transpose_append, lv1473, lv1475, R.prim_value(14), sinfo_args=(R.Object,))
            lv1477 = R.call_dps_packed("vm.builtin.paged_attention_kv_cache_attention", (lv1476, cls.attention, lv1471, R.prim_value(14), R.prim_value(1), R.prim_value(T.float32(1)), R.prim_value(10000)), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1478 = R.call_tir(cls.reshape1, (lv1477,), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv213: R.Tensor((5120, 640), dtype="uint32") = model_params[144]
            lv214: R.Tensor((5120, 160), dtype="float16") = model_params[145]
            lv108_1 = R.call_tir(cls.fused_fused_decode1_fused_NT_matmul1_add, (lv213, lv214, lv1478, lv107), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv145_1: R.Tensor((5120,), dtype="float16") = model_params[151]
            lv1482 = R.call_tir(cls.rms_norm1, (lv108_1, lv145_1), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv217: R.Tensor((27648, 640), dtype="uint32") = model_params[146]
            lv218: R.Tensor((27648, 160), dtype="float16") = model_params[147]
            lv29 = R.call_tir(cls.fused_fused_decode2_NT_matmul2, (lv217, lv218, lv1482), out_sinfo=R.Tensor((nseq, 1, 27648), dtype="float16"))
            lv220 = R.call_tir(cls.fused_split1_silu_multiply, (lv29,), out_sinfo=R.Tensor((nseq, 1, 13824), dtype="float16"))
            lv221: R.Tensor((5120, 1728), dtype="uint32") = model_params[148]
            lv222: R.Tensor((5120, 432), dtype="float16") = model_params[149]
            lv109_1 = R.call_tir(cls.fused_fused_decode3_fused_NT_matmul3_add, (lv221, lv222, lv220, lv108_1), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv150_1: R.Tensor((5120,), dtype="float16") = model_params[160]
            lv1493 = R.call_tir(cls.rms_norm1, (lv109_1, lv150_1), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv225: R.Tensor((15360, 640), dtype="uint32") = model_params[152]
            lv226: R.Tensor((15360, 160), dtype="float16") = model_params[153]
            lv30_2 = R.call_tir(cls.fused_fused_decode_NT_matmul, (lv225, lv226, lv1493), out_sinfo=R.Tensor((nseq, 1, 15360), dtype="float16"))
            lv1496 = R.call_tir(cls.split, (lv30_2,), out_sinfo=[R.Tensor((nseq, 1, 5120), dtype="float16"), R.Tensor((nseq, 1, 5120), dtype="float16"), R.Tensor((nseq, 1, 5120), dtype="float16")])
            lv1497: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1496[0]
            lv1498 = R.call_tir(cls.reshape, (lv1497,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1499: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1496[1]
            lv1500 = R.call_tir(cls.reshape, (lv1499,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1501: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1496[2]
            lv1502 = R.call_tir(cls.reshape, (lv1501,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1503: R.Object = R.call_pure_packed("vm.builtin.paged_attention_kv_cache_append", lv1476, cls.kv_cache_transpose_append, lv1500, lv1502, R.prim_value(15), sinfo_args=(R.Object,))
            lv1504 = R.call_dps_packed("vm.builtin.paged_attention_kv_cache_attention", (lv1503, cls.attention, lv1498, R.prim_value(15), R.prim_value(1), R.prim_value(T.float32(1)), R.prim_value(10000)), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1505 = R.call_tir(cls.reshape1, (lv1504,), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv228: R.Tensor((5120, 640), dtype="uint32") = model_params[154]
            lv229: R.Tensor((5120, 160), dtype="float16") = model_params[155]
            lv110_1 = R.call_tir(cls.fused_fused_decode1_fused_NT_matmul1_add, (lv228, lv229, lv1505, lv109_1), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv155: R.Tensor((5120,), dtype="float16") = model_params[161]
            lv1509 = R.call_tir(cls.rms_norm1, (lv110_1, lv155), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv232: R.Tensor((27648, 640), dtype="uint32") = model_params[156]
            lv233: R.Tensor((27648, 160), dtype="float16") = model_params[157]
            lv31_1 = R.call_tir(cls.fused_fused_decode2_NT_matmul2, (lv232, lv233, lv1509), out_sinfo=R.Tensor((nseq, 1, 27648), dtype="float16"))
            lv235 = R.call_tir(cls.fused_split1_silu_multiply, (lv31_1,), out_sinfo=R.Tensor((nseq, 1, 13824), dtype="float16"))
            lv236: R.Tensor((5120, 1728), dtype="uint32") = model_params[158]
            lv237: R.Tensor((5120, 432), dtype="float16") = model_params[159]
            lv111 = R.call_tir(cls.fused_fused_decode3_fused_NT_matmul3_add, (lv236, lv237, lv235, lv110_1), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv160_1: R.Tensor((5120,), dtype="float16") = model_params[170]
            lv1520 = R.call_tir(cls.rms_norm1, (lv111, lv160_1), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv240: R.Tensor((15360, 640), dtype="uint32") = model_params[162]
            lv241: R.Tensor((15360, 160), dtype="float16") = model_params[163]
            lv32 = R.call_tir(cls.fused_fused_decode_NT_matmul, (lv240, lv241, lv1520), out_sinfo=R.Tensor((nseq, 1, 15360), dtype="float16"))
            lv1523 = R.call_tir(cls.split, (lv32,), out_sinfo=[R.Tensor((nseq, 1, 5120), dtype="float16"), R.Tensor((nseq, 1, 5120), dtype="float16"), R.Tensor((nseq, 1, 5120), dtype="float16")])
            lv1524: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1523[0]
            lv1525 = R.call_tir(cls.reshape, (lv1524,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1526: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1523[1]
            lv1527 = R.call_tir(cls.reshape, (lv1526,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1528: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1523[2]
            lv1529 = R.call_tir(cls.reshape, (lv1528,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1530: R.Object = R.call_pure_packed("vm.builtin.paged_attention_kv_cache_append", lv1503, cls.kv_cache_transpose_append, lv1527, lv1529, R.prim_value(16), sinfo_args=(R.Object,))
            lv1531 = R.call_dps_packed("vm.builtin.paged_attention_kv_cache_attention", (lv1530, cls.attention, lv1525, R.prim_value(16), R.prim_value(1), R.prim_value(T.float32(1)), R.prim_value(10000)), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1532 = R.call_tir(cls.reshape1, (lv1531,), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv243: R.Tensor((5120, 640), dtype="uint32") = model_params[164]
            lv244: R.Tensor((5120, 160), dtype="float16") = model_params[165]
            lv112_1 = R.call_tir(cls.fused_fused_decode1_fused_NT_matmul1_add, (lv243, lv244, lv1532, lv111), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv165_1: R.Tensor((5120,), dtype="float16") = model_params[171]
            lv1536 = R.call_tir(cls.rms_norm1, (lv112_1, lv165_1), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv247: R.Tensor((27648, 640), dtype="uint32") = model_params[166]
            lv248: R.Tensor((27648, 160), dtype="float16") = model_params[167]
            lv33_1 = R.call_tir(cls.fused_fused_decode2_NT_matmul2, (lv247, lv248, lv1536), out_sinfo=R.Tensor((nseq, 1, 27648), dtype="float16"))
            lv250 = R.call_tir(cls.fused_split1_silu_multiply, (lv33_1,), out_sinfo=R.Tensor((nseq, 1, 13824), dtype="float16"))
            lv251: R.Tensor((5120, 1728), dtype="uint32") = model_params[168]
            lv252: R.Tensor((5120, 432), dtype="float16") = model_params[169]
            lv113_1 = R.call_tir(cls.fused_fused_decode3_fused_NT_matmul3_add, (lv251, lv252, lv250, lv112_1), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv170: R.Tensor((5120,), dtype="float16") = model_params[180]
            lv1547 = R.call_tir(cls.rms_norm1, (lv113_1, lv170), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv255: R.Tensor((15360, 640), dtype="uint32") = model_params[172]
            lv256: R.Tensor((15360, 160), dtype="float16") = model_params[173]
            lv34_1 = R.call_tir(cls.fused_fused_decode_NT_matmul, (lv255, lv256, lv1547), out_sinfo=R.Tensor((nseq, 1, 15360), dtype="float16"))
            lv1550 = R.call_tir(cls.split, (lv34_1,), out_sinfo=[R.Tensor((nseq, 1, 5120), dtype="float16"), R.Tensor((nseq, 1, 5120), dtype="float16"), R.Tensor((nseq, 1, 5120), dtype="float16")])
            lv1551: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1550[0]
            lv1552 = R.call_tir(cls.reshape, (lv1551,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1553: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1550[1]
            lv1554 = R.call_tir(cls.reshape, (lv1553,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1555: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1550[2]
            lv1556 = R.call_tir(cls.reshape, (lv1555,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1557: R.Object = R.call_pure_packed("vm.builtin.paged_attention_kv_cache_append", lv1530, cls.kv_cache_transpose_append, lv1554, lv1556, R.prim_value(17), sinfo_args=(R.Object,))
            lv1558 = R.call_dps_packed("vm.builtin.paged_attention_kv_cache_attention", (lv1557, cls.attention, lv1552, R.prim_value(17), R.prim_value(1), R.prim_value(T.float32(1)), R.prim_value(10000)), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1559 = R.call_tir(cls.reshape1, (lv1558,), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv258: R.Tensor((5120, 640), dtype="uint32") = model_params[174]
            lv259: R.Tensor((5120, 160), dtype="float16") = model_params[175]
            lv114 = R.call_tir(cls.fused_fused_decode1_fused_NT_matmul1_add, (lv258, lv259, lv1559, lv113_1), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv175_1: R.Tensor((5120,), dtype="float16") = model_params[181]
            lv1563 = R.call_tir(cls.rms_norm1, (lv114, lv175_1), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv262: R.Tensor((27648, 640), dtype="uint32") = model_params[176]
            lv263: R.Tensor((27648, 160), dtype="float16") = model_params[177]
            lv35_1 = R.call_tir(cls.fused_fused_decode2_NT_matmul2, (lv262, lv263, lv1563), out_sinfo=R.Tensor((nseq, 1, 27648), dtype="float16"))
            lv265 = R.call_tir(cls.fused_split1_silu_multiply, (lv35_1,), out_sinfo=R.Tensor((nseq, 1, 13824), dtype="float16"))
            lv266: R.Tensor((5120, 1728), dtype="uint32") = model_params[178]
            lv267: R.Tensor((5120, 432), dtype="float16") = model_params[179]
            lv115_2 = R.call_tir(cls.fused_fused_decode3_fused_NT_matmul3_add, (lv266, lv267, lv265, lv114), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv180_1: R.Tensor((5120,), dtype="float16") = model_params[190]
            lv1574 = R.call_tir(cls.rms_norm1, (lv115_2, lv180_1), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv270: R.Tensor((15360, 640), dtype="uint32") = model_params[182]
            lv271: R.Tensor((15360, 160), dtype="float16") = model_params[183]
            lv36 = R.call_tir(cls.fused_fused_decode_NT_matmul, (lv270, lv271, lv1574), out_sinfo=R.Tensor((nseq, 1, 15360), dtype="float16"))
            lv1577 = R.call_tir(cls.split, (lv36,), out_sinfo=[R.Tensor((nseq, 1, 5120), dtype="float16"), R.Tensor((nseq, 1, 5120), dtype="float16"), R.Tensor((nseq, 1, 5120), dtype="float16")])
            lv1578: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1577[0]
            lv1579 = R.call_tir(cls.reshape, (lv1578,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1580: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1577[1]
            lv1581 = R.call_tir(cls.reshape, (lv1580,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1582: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1577[2]
            lv1583 = R.call_tir(cls.reshape, (lv1582,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1584: R.Object = R.call_pure_packed("vm.builtin.paged_attention_kv_cache_append", lv1557, cls.kv_cache_transpose_append, lv1581, lv1583, R.prim_value(18), sinfo_args=(R.Object,))
            lv1585 = R.call_dps_packed("vm.builtin.paged_attention_kv_cache_attention", (lv1584, cls.attention, lv1579, R.prim_value(18), R.prim_value(1), R.prim_value(T.float32(1)), R.prim_value(10000)), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1586 = R.call_tir(cls.reshape1, (lv1585,), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv273: R.Tensor((5120, 640), dtype="uint32") = model_params[184]
            lv274: R.Tensor((5120, 160), dtype="float16") = model_params[185]
            lv116_1 = R.call_tir(cls.fused_fused_decode1_fused_NT_matmul1_add, (lv273, lv274, lv1586, lv115_2), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv185: R.Tensor((5120,), dtype="float16") = model_params[191]
            lv1590 = R.call_tir(cls.rms_norm1, (lv116_1, lv185), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv277: R.Tensor((27648, 640), dtype="uint32") = model_params[186]
            lv278: R.Tensor((27648, 160), dtype="float16") = model_params[187]
            lv37_1 = R.call_tir(cls.fused_fused_decode2_NT_matmul2, (lv277, lv278, lv1590), out_sinfo=R.Tensor((nseq, 1, 27648), dtype="float16"))
            lv280 = R.call_tir(cls.fused_split1_silu_multiply, (lv37_1,), out_sinfo=R.Tensor((nseq, 1, 13824), dtype="float16"))
            lv281: R.Tensor((5120, 1728), dtype="uint32") = model_params[188]
            lv282: R.Tensor((5120, 432), dtype="float16") = model_params[189]
            lv117_1 = R.call_tir(cls.fused_fused_decode3_fused_NT_matmul3_add, (lv281, lv282, lv280, lv116_1), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv190_1: R.Tensor((5120,), dtype="float16") = model_params[200]
            lv1601 = R.call_tir(cls.rms_norm1, (lv117_1, lv190_1), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv285: R.Tensor((15360, 640), dtype="uint32") = model_params[192]
            lv286: R.Tensor((15360, 160), dtype="float16") = model_params[193]
            lv38_1 = R.call_tir(cls.fused_fused_decode_NT_matmul, (lv285, lv286, lv1601), out_sinfo=R.Tensor((nseq, 1, 15360), dtype="float16"))
            lv1604 = R.call_tir(cls.split, (lv38_1,), out_sinfo=[R.Tensor((nseq, 1, 5120), dtype="float16"), R.Tensor((nseq, 1, 5120), dtype="float16"), R.Tensor((nseq, 1, 5120), dtype="float16")])
            lv1605: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1604[0]
            lv1606 = R.call_tir(cls.reshape, (lv1605,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1607: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1604[1]
            lv1608 = R.call_tir(cls.reshape, (lv1607,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1609: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1604[2]
            lv1610 = R.call_tir(cls.reshape, (lv1609,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1611: R.Object = R.call_pure_packed("vm.builtin.paged_attention_kv_cache_append", lv1584, cls.kv_cache_transpose_append, lv1608, lv1610, R.prim_value(19), sinfo_args=(R.Object,))
            lv1612 = R.call_dps_packed("vm.builtin.paged_attention_kv_cache_attention", (lv1611, cls.attention, lv1606, R.prim_value(19), R.prim_value(1), R.prim_value(T.float32(1)), R.prim_value(10000)), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1613 = R.call_tir(cls.reshape1, (lv1612,), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv288: R.Tensor((5120, 640), dtype="uint32") = model_params[194]
            lv289: R.Tensor((5120, 160), dtype="float16") = model_params[195]
            lv118 = R.call_tir(cls.fused_fused_decode1_fused_NT_matmul1_add, (lv288, lv289, lv1613, lv117_1), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv195_1: R.Tensor((5120,), dtype="float16") = model_params[201]
            lv1617 = R.call_tir(cls.rms_norm1, (lv118, lv195_1), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv292: R.Tensor((27648, 640), dtype="uint32") = model_params[196]
            lv293: R.Tensor((27648, 160), dtype="float16") = model_params[197]
            lv39 = R.call_tir(cls.fused_fused_decode2_NT_matmul2, (lv292, lv293, lv1617), out_sinfo=R.Tensor((nseq, 1, 27648), dtype="float16"))
            lv295 = R.call_tir(cls.fused_split1_silu_multiply, (lv39,), out_sinfo=R.Tensor((nseq, 1, 13824), dtype="float16"))
            lv296: R.Tensor((5120, 1728), dtype="uint32") = model_params[198]
            lv297: R.Tensor((5120, 432), dtype="float16") = model_params[199]
            lv119 = R.call_tir(cls.fused_fused_decode3_fused_NT_matmul3_add, (lv296, lv297, lv295, lv118), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv200: R.Tensor((5120,), dtype="float16") = model_params[210]
            lv1628 = R.call_tir(cls.rms_norm1, (lv119, lv200), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv300: R.Tensor((15360, 640), dtype="uint32") = model_params[202]
            lv301: R.Tensor((15360, 160), dtype="float16") = model_params[203]
            lv40_2 = R.call_tir(cls.fused_fused_decode_NT_matmul, (lv300, lv301, lv1628), out_sinfo=R.Tensor((nseq, 1, 15360), dtype="float16"))
            lv1631 = R.call_tir(cls.split, (lv40_2,), out_sinfo=[R.Tensor((nseq, 1, 5120), dtype="float16"), R.Tensor((nseq, 1, 5120), dtype="float16"), R.Tensor((nseq, 1, 5120), dtype="float16")])
            lv1632: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1631[0]
            lv1633 = R.call_tir(cls.reshape, (lv1632,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1634: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1631[1]
            lv1635 = R.call_tir(cls.reshape, (lv1634,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1636: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1631[2]
            lv1637 = R.call_tir(cls.reshape, (lv1636,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1638: R.Object = R.call_pure_packed("vm.builtin.paged_attention_kv_cache_append", lv1611, cls.kv_cache_transpose_append, lv1635, lv1637, R.prim_value(20), sinfo_args=(R.Object,))
            lv1639 = R.call_dps_packed("vm.builtin.paged_attention_kv_cache_attention", (lv1638, cls.attention, lv1633, R.prim_value(20), R.prim_value(1), R.prim_value(T.float32(1)), R.prim_value(10000)), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1640 = R.call_tir(cls.reshape1, (lv1639,), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv303: R.Tensor((5120, 640), dtype="uint32") = model_params[204]
            lv304: R.Tensor((5120, 160), dtype="float16") = model_params[205]
            lv120_2 = R.call_tir(cls.fused_fused_decode1_fused_NT_matmul1_add, (lv303, lv304, lv1640, lv119), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv205_1: R.Tensor((5120,), dtype="float16") = model_params[211]
            lv1644 = R.call_tir(cls.rms_norm1, (lv120_2, lv205_1), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv307: R.Tensor((27648, 640), dtype="uint32") = model_params[206]
            lv308: R.Tensor((27648, 160), dtype="float16") = model_params[207]
            lv41_1 = R.call_tir(cls.fused_fused_decode2_NT_matmul2, (lv307, lv308, lv1644), out_sinfo=R.Tensor((nseq, 1, 27648), dtype="float16"))
            lv310 = R.call_tir(cls.fused_split1_silu_multiply, (lv41_1,), out_sinfo=R.Tensor((nseq, 1, 13824), dtype="float16"))
            lv311: R.Tensor((5120, 1728), dtype="uint32") = model_params[208]
            lv312: R.Tensor((5120, 432), dtype="float16") = model_params[209]
            lv121_1 = R.call_tir(cls.fused_fused_decode3_fused_NT_matmul3_add, (lv311, lv312, lv310, lv120_2), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv210_1: R.Tensor((5120,), dtype="float16") = model_params[220]
            lv1655 = R.call_tir(cls.rms_norm1, (lv121_1, lv210_1), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv315: R.Tensor((15360, 640), dtype="uint32") = model_params[212]
            lv316: R.Tensor((15360, 160), dtype="float16") = model_params[213]
            lv42_1 = R.call_tir(cls.fused_fused_decode_NT_matmul, (lv315, lv316, lv1655), out_sinfo=R.Tensor((nseq, 1, 15360), dtype="float16"))
            lv1658 = R.call_tir(cls.split, (lv42_1,), out_sinfo=[R.Tensor((nseq, 1, 5120), dtype="float16"), R.Tensor((nseq, 1, 5120), dtype="float16"), R.Tensor((nseq, 1, 5120), dtype="float16")])
            lv1659: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1658[0]
            lv1660 = R.call_tir(cls.reshape, (lv1659,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1661: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1658[1]
            lv1662 = R.call_tir(cls.reshape, (lv1661,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1663: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1658[2]
            lv1664 = R.call_tir(cls.reshape, (lv1663,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1665: R.Object = R.call_pure_packed("vm.builtin.paged_attention_kv_cache_append", lv1638, cls.kv_cache_transpose_append, lv1662, lv1664, R.prim_value(21), sinfo_args=(R.Object,))
            lv1666 = R.call_dps_packed("vm.builtin.paged_attention_kv_cache_attention", (lv1665, cls.attention, lv1660, R.prim_value(21), R.prim_value(1), R.prim_value(T.float32(1)), R.prim_value(10000)), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1667 = R.call_tir(cls.reshape1, (lv1666,), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv318: R.Tensor((5120, 640), dtype="uint32") = model_params[214]
            lv319: R.Tensor((5120, 160), dtype="float16") = model_params[215]
            lv122 = R.call_tir(cls.fused_fused_decode1_fused_NT_matmul1_add, (lv318, lv319, lv1667, lv121_1), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv215: R.Tensor((5120,), dtype="float16") = model_params[221]
            lv1671 = R.call_tir(cls.rms_norm1, (lv122, lv215), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv322: R.Tensor((27648, 640), dtype="uint32") = model_params[216]
            lv323: R.Tensor((27648, 160), dtype="float16") = model_params[217]
            lv43 = R.call_tir(cls.fused_fused_decode2_NT_matmul2, (lv322, lv323, lv1671), out_sinfo=R.Tensor((nseq, 1, 27648), dtype="float16"))
            lv325 = R.call_tir(cls.fused_split1_silu_multiply, (lv43,), out_sinfo=R.Tensor((nseq, 1, 13824), dtype="float16"))
            lv326: R.Tensor((5120, 1728), dtype="uint32") = model_params[218]
            lv327: R.Tensor((5120, 432), dtype="float16") = model_params[219]
            lv123_1 = R.call_tir(cls.fused_fused_decode3_fused_NT_matmul3_add, (lv326, lv327, lv325, lv122), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv220_1: R.Tensor((5120,), dtype="float16") = model_params[230]
            lv1682 = R.call_tir(cls.rms_norm1, (lv123_1, lv220_1), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv330: R.Tensor((15360, 640), dtype="uint32") = model_params[222]
            lv331: R.Tensor((15360, 160), dtype="float16") = model_params[223]
            lv44 = R.call_tir(cls.fused_fused_decode_NT_matmul, (lv330, lv331, lv1682), out_sinfo=R.Tensor((nseq, 1, 15360), dtype="float16"))
            lv1685 = R.call_tir(cls.split, (lv44,), out_sinfo=[R.Tensor((nseq, 1, 5120), dtype="float16"), R.Tensor((nseq, 1, 5120), dtype="float16"), R.Tensor((nseq, 1, 5120), dtype="float16")])
            lv1686: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1685[0]
            lv1687 = R.call_tir(cls.reshape, (lv1686,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1688: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1685[1]
            lv1689 = R.call_tir(cls.reshape, (lv1688,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1690: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1685[2]
            lv1691 = R.call_tir(cls.reshape, (lv1690,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1692: R.Object = R.call_pure_packed("vm.builtin.paged_attention_kv_cache_append", lv1665, cls.kv_cache_transpose_append, lv1689, lv1691, R.prim_value(22), sinfo_args=(R.Object,))
            lv1693 = R.call_dps_packed("vm.builtin.paged_attention_kv_cache_attention", (lv1692, cls.attention, lv1687, R.prim_value(22), R.prim_value(1), R.prim_value(T.float32(1)), R.prim_value(10000)), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1694 = R.call_tir(cls.reshape1, (lv1693,), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv333: R.Tensor((5120, 640), dtype="uint32") = model_params[224]
            lv334: R.Tensor((5120, 160), dtype="float16") = model_params[225]
            lv124_1 = R.call_tir(cls.fused_fused_decode1_fused_NT_matmul1_add, (lv333, lv334, lv1694, lv123_1), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv225_1: R.Tensor((5120,), dtype="float16") = model_params[231]
            lv1698 = R.call_tir(cls.rms_norm1, (lv124_1, lv225_1), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv337: R.Tensor((27648, 640), dtype="uint32") = model_params[226]
            lv338: R.Tensor((27648, 160), dtype="float16") = model_params[227]
            lv45_2 = R.call_tir(cls.fused_fused_decode2_NT_matmul2, (lv337, lv338, lv1698), out_sinfo=R.Tensor((nseq, 1, 27648), dtype="float16"))
            lv340 = R.call_tir(cls.fused_split1_silu_multiply, (lv45_2,), out_sinfo=R.Tensor((nseq, 1, 13824), dtype="float16"))
            lv341: R.Tensor((5120, 1728), dtype="uint32") = model_params[228]
            lv342: R.Tensor((5120, 432), dtype="float16") = model_params[229]
            lv125_1 = R.call_tir(cls.fused_fused_decode3_fused_NT_matmul3_add, (lv341, lv342, lv340, lv124_1), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv230: R.Tensor((5120,), dtype="float16") = model_params[240]
            lv1709 = R.call_tir(cls.rms_norm1, (lv125_1, lv230), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv345: R.Tensor((15360, 640), dtype="uint32") = model_params[232]
            lv346: R.Tensor((15360, 160), dtype="float16") = model_params[233]
            lv46_1 = R.call_tir(cls.fused_fused_decode_NT_matmul, (lv345, lv346, lv1709), out_sinfo=R.Tensor((nseq, 1, 15360), dtype="float16"))
            lv1712 = R.call_tir(cls.split, (lv46_1,), out_sinfo=[R.Tensor((nseq, 1, 5120), dtype="float16"), R.Tensor((nseq, 1, 5120), dtype="float16"), R.Tensor((nseq, 1, 5120), dtype="float16")])
            lv1713: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1712[0]
            lv1714 = R.call_tir(cls.reshape, (lv1713,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1715: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1712[1]
            lv1716 = R.call_tir(cls.reshape, (lv1715,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1717: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1712[2]
            lv1718 = R.call_tir(cls.reshape, (lv1717,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1719: R.Object = R.call_pure_packed("vm.builtin.paged_attention_kv_cache_append", lv1692, cls.kv_cache_transpose_append, lv1716, lv1718, R.prim_value(23), sinfo_args=(R.Object,))
            lv1720 = R.call_dps_packed("vm.builtin.paged_attention_kv_cache_attention", (lv1719, cls.attention, lv1714, R.prim_value(23), R.prim_value(1), R.prim_value(T.float32(1)), R.prim_value(10000)), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1721 = R.call_tir(cls.reshape1, (lv1720,), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv348: R.Tensor((5120, 640), dtype="uint32") = model_params[234]
            lv349: R.Tensor((5120, 160), dtype="float16") = model_params[235]
            lv126 = R.call_tir(cls.fused_fused_decode1_fused_NT_matmul1_add, (lv348, lv349, lv1721, lv125_1), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv235_1: R.Tensor((5120,), dtype="float16") = model_params[241]
            lv1725 = R.call_tir(cls.rms_norm1, (lv126, lv235_1), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv352: R.Tensor((27648, 640), dtype="uint32") = model_params[236]
            lv353: R.Tensor((27648, 160), dtype="float16") = model_params[237]
            lv47 = R.call_tir(cls.fused_fused_decode2_NT_matmul2, (lv352, lv353, lv1725), out_sinfo=R.Tensor((nseq, 1, 27648), dtype="float16"))
            lv355 = R.call_tir(cls.fused_split1_silu_multiply, (lv47,), out_sinfo=R.Tensor((nseq, 1, 13824), dtype="float16"))
            lv356: R.Tensor((5120, 1728), dtype="uint32") = model_params[238]
            lv357: R.Tensor((5120, 432), dtype="float16") = model_params[239]
            lv127_1 = R.call_tir(cls.fused_fused_decode3_fused_NT_matmul3_add, (lv356, lv357, lv355, lv126), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv240_1: R.Tensor((5120,), dtype="float16") = model_params[250]
            lv1736 = R.call_tir(cls.rms_norm1, (lv127_1, lv240_1), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv360: R.Tensor((15360, 640), dtype="uint32") = model_params[242]
            lv361: R.Tensor((15360, 160), dtype="float16") = model_params[243]
            lv48_1 = R.call_tir(cls.fused_fused_decode_NT_matmul, (lv360, lv361, lv1736), out_sinfo=R.Tensor((nseq, 1, 15360), dtype="float16"))
            lv1739 = R.call_tir(cls.split, (lv48_1,), out_sinfo=[R.Tensor((nseq, 1, 5120), dtype="float16"), R.Tensor((nseq, 1, 5120), dtype="float16"), R.Tensor((nseq, 1, 5120), dtype="float16")])
            lv1740: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1739[0]
            lv1741 = R.call_tir(cls.reshape, (lv1740,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1742: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1739[1]
            lv1743 = R.call_tir(cls.reshape, (lv1742,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1744: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1739[2]
            lv1745 = R.call_tir(cls.reshape, (lv1744,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1746: R.Object = R.call_pure_packed("vm.builtin.paged_attention_kv_cache_append", lv1719, cls.kv_cache_transpose_append, lv1743, lv1745, R.prim_value(24), sinfo_args=(R.Object,))
            lv1747 = R.call_dps_packed("vm.builtin.paged_attention_kv_cache_attention", (lv1746, cls.attention, lv1741, R.prim_value(24), R.prim_value(1), R.prim_value(T.float32(1)), R.prim_value(10000)), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1748 = R.call_tir(cls.reshape1, (lv1747,), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv363: R.Tensor((5120, 640), dtype="uint32") = model_params[244]
            lv364: R.Tensor((5120, 160), dtype="float16") = model_params[245]
            lv128_1 = R.call_tir(cls.fused_fused_decode1_fused_NT_matmul1_add, (lv363, lv364, lv1748, lv127_1), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv245: R.Tensor((5120,), dtype="float16") = model_params[251]
            lv1752 = R.call_tir(cls.rms_norm1, (lv128_1, lv245), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv367: R.Tensor((27648, 640), dtype="uint32") = model_params[246]
            lv368: R.Tensor((27648, 160), dtype="float16") = model_params[247]
            lv49_1 = R.call_tir(cls.fused_fused_decode2_NT_matmul2, (lv367, lv368, lv1752), out_sinfo=R.Tensor((nseq, 1, 27648), dtype="float16"))
            lv370 = R.call_tir(cls.fused_split1_silu_multiply, (lv49_1,), out_sinfo=R.Tensor((nseq, 1, 13824), dtype="float16"))
            lv371: R.Tensor((5120, 1728), dtype="uint32") = model_params[248]
            lv372: R.Tensor((5120, 432), dtype="float16") = model_params[249]
            lv129 = R.call_tir(cls.fused_fused_decode3_fused_NT_matmul3_add, (lv371, lv372, lv370, lv128_1), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv250_1: R.Tensor((5120,), dtype="float16") = model_params[260]
            lv1763 = R.call_tir(cls.rms_norm1, (lv129, lv250_1), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv375: R.Tensor((15360, 640), dtype="uint32") = model_params[252]
            lv376: R.Tensor((15360, 160), dtype="float16") = model_params[253]
            lv50_1 = R.call_tir(cls.fused_fused_decode_NT_matmul, (lv375, lv376, lv1763), out_sinfo=R.Tensor((nseq, 1, 15360), dtype="float16"))
            lv1766 = R.call_tir(cls.split, (lv50_1,), out_sinfo=[R.Tensor((nseq, 1, 5120), dtype="float16"), R.Tensor((nseq, 1, 5120), dtype="float16"), R.Tensor((nseq, 1, 5120), dtype="float16")])
            lv1767: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1766[0]
            lv1768 = R.call_tir(cls.reshape, (lv1767,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1769: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1766[1]
            lv1770 = R.call_tir(cls.reshape, (lv1769,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1771: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1766[2]
            lv1772 = R.call_tir(cls.reshape, (lv1771,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1773: R.Object = R.call_pure_packed("vm.builtin.paged_attention_kv_cache_append", lv1746, cls.kv_cache_transpose_append, lv1770, lv1772, R.prim_value(25), sinfo_args=(R.Object,))
            lv1774 = R.call_dps_packed("vm.builtin.paged_attention_kv_cache_attention", (lv1773, cls.attention, lv1768, R.prim_value(25), R.prim_value(1), R.prim_value(T.float32(1)), R.prim_value(10000)), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1775 = R.call_tir(cls.reshape1, (lv1774,), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv378: R.Tensor((5120, 640), dtype="uint32") = model_params[254]
            lv379: R.Tensor((5120, 160), dtype="float16") = model_params[255]
            lv130_2 = R.call_tir(cls.fused_fused_decode1_fused_NT_matmul1_add, (lv378, lv379, lv1775, lv129), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv255_1: R.Tensor((5120,), dtype="float16") = model_params[261]
            lv1779 = R.call_tir(cls.rms_norm1, (lv130_2, lv255_1), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv382: R.Tensor((27648, 640), dtype="uint32") = model_params[256]
            lv383: R.Tensor((27648, 160), dtype="float16") = model_params[257]
            lv51 = R.call_tir(cls.fused_fused_decode2_NT_matmul2, (lv382, lv383, lv1779), out_sinfo=R.Tensor((nseq, 1, 27648), dtype="float16"))
            lv385 = R.call_tir(cls.fused_split1_silu_multiply, (lv51,), out_sinfo=R.Tensor((nseq, 1, 13824), dtype="float16"))
            lv386: R.Tensor((5120, 1728), dtype="uint32") = model_params[258]
            lv387: R.Tensor((5120, 432), dtype="float16") = model_params[259]
            lv131_1 = R.call_tir(cls.fused_fused_decode3_fused_NT_matmul3_add, (lv386, lv387, lv385, lv130_2), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv260: R.Tensor((5120,), dtype="float16") = model_params[270]
            lv1790 = R.call_tir(cls.rms_norm1, (lv131_1, lv260), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv390: R.Tensor((15360, 640), dtype="uint32") = model_params[262]
            lv391: R.Tensor((15360, 160), dtype="float16") = model_params[263]
            lv52_1 = R.call_tir(cls.fused_fused_decode_NT_matmul, (lv390, lv391, lv1790), out_sinfo=R.Tensor((nseq, 1, 15360), dtype="float16"))
            lv1793 = R.call_tir(cls.split, (lv52_1,), out_sinfo=[R.Tensor((nseq, 1, 5120), dtype="float16"), R.Tensor((nseq, 1, 5120), dtype="float16"), R.Tensor((nseq, 1, 5120), dtype="float16")])
            lv1794: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1793[0]
            lv1795 = R.call_tir(cls.reshape, (lv1794,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1796: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1793[1]
            lv1797 = R.call_tir(cls.reshape, (lv1796,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1798: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1793[2]
            lv1799 = R.call_tir(cls.reshape, (lv1798,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1800: R.Object = R.call_pure_packed("vm.builtin.paged_attention_kv_cache_append", lv1773, cls.kv_cache_transpose_append, lv1797, lv1799, R.prim_value(26), sinfo_args=(R.Object,))
            lv1801 = R.call_dps_packed("vm.builtin.paged_attention_kv_cache_attention", (lv1800, cls.attention, lv1795, R.prim_value(26), R.prim_value(1), R.prim_value(T.float32(1)), R.prim_value(10000)), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1802 = R.call_tir(cls.reshape1, (lv1801,), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv393: R.Tensor((5120, 640), dtype="uint32") = model_params[264]
            lv394: R.Tensor((5120, 160), dtype="float16") = model_params[265]
            lv132_1 = R.call_tir(cls.fused_fused_decode1_fused_NT_matmul1_add, (lv393, lv394, lv1802, lv131_1), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv265_1: R.Tensor((5120,), dtype="float16") = model_params[271]
            lv1806 = R.call_tir(cls.rms_norm1, (lv132_1, lv265_1), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv397: R.Tensor((27648, 640), dtype="uint32") = model_params[266]
            lv398: R.Tensor((27648, 160), dtype="float16") = model_params[267]
            lv53_1 = R.call_tir(cls.fused_fused_decode2_NT_matmul2, (lv397, lv398, lv1806), out_sinfo=R.Tensor((nseq, 1, 27648), dtype="float16"))
            lv400 = R.call_tir(cls.fused_split1_silu_multiply, (lv53_1,), out_sinfo=R.Tensor((nseq, 1, 13824), dtype="float16"))
            lv401: R.Tensor((5120, 1728), dtype="uint32") = model_params[268]
            lv402: R.Tensor((5120, 432), dtype="float16") = model_params[269]
            lv133 = R.call_tir(cls.fused_fused_decode3_fused_NT_matmul3_add, (lv401, lv402, lv400, lv132_1), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv270_1: R.Tensor((5120,), dtype="float16") = model_params[280]
            lv1817 = R.call_tir(cls.rms_norm1, (lv133, lv270_1), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv405: R.Tensor((15360, 640), dtype="uint32") = model_params[272]
            lv406: R.Tensor((15360, 160), dtype="float16") = model_params[273]
            lv54 = R.call_tir(cls.fused_fused_decode_NT_matmul, (lv405, lv406, lv1817), out_sinfo=R.Tensor((nseq, 1, 15360), dtype="float16"))
            lv1820 = R.call_tir(cls.split, (lv54,), out_sinfo=[R.Tensor((nseq, 1, 5120), dtype="float16"), R.Tensor((nseq, 1, 5120), dtype="float16"), R.Tensor((nseq, 1, 5120), dtype="float16")])
            lv1821: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1820[0]
            lv1822 = R.call_tir(cls.reshape, (lv1821,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1823: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1820[1]
            lv1824 = R.call_tir(cls.reshape, (lv1823,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1825: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1820[2]
            lv1826 = R.call_tir(cls.reshape, (lv1825,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1827: R.Object = R.call_pure_packed("vm.builtin.paged_attention_kv_cache_append", lv1800, cls.kv_cache_transpose_append, lv1824, lv1826, R.prim_value(27), sinfo_args=(R.Object,))
            lv1828 = R.call_dps_packed("vm.builtin.paged_attention_kv_cache_attention", (lv1827, cls.attention, lv1822, R.prim_value(27), R.prim_value(1), R.prim_value(T.float32(1)), R.prim_value(10000)), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1829 = R.call_tir(cls.reshape1, (lv1828,), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv408: R.Tensor((5120, 640), dtype="uint32") = model_params[274]
            lv409: R.Tensor((5120, 160), dtype="float16") = model_params[275]
            lv134 = R.call_tir(cls.fused_fused_decode1_fused_NT_matmul1_add, (lv408, lv409, lv1829, lv133), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv275: R.Tensor((5120,), dtype="float16") = model_params[281]
            lv1833 = R.call_tir(cls.rms_norm1, (lv134, lv275), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv412: R.Tensor((27648, 640), dtype="uint32") = model_params[276]
            lv413: R.Tensor((27648, 160), dtype="float16") = model_params[277]
            lv55_2 = R.call_tir(cls.fused_fused_decode2_NT_matmul2, (lv412, lv413, lv1833), out_sinfo=R.Tensor((nseq, 1, 27648), dtype="float16"))
            lv415 = R.call_tir(cls.fused_split1_silu_multiply, (lv55_2,), out_sinfo=R.Tensor((nseq, 1, 13824), dtype="float16"))
            lv416: R.Tensor((5120, 1728), dtype="uint32") = model_params[278]
            lv417: R.Tensor((5120, 432), dtype="float16") = model_params[279]
            lv135_2 = R.call_tir(cls.fused_fused_decode3_fused_NT_matmul3_add, (lv416, lv417, lv415, lv134), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv280_1: R.Tensor((5120,), dtype="float16") = model_params[290]
            lv1844 = R.call_tir(cls.rms_norm1, (lv135_2, lv280_1), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv420: R.Tensor((15360, 640), dtype="uint32") = model_params[282]
            lv421: R.Tensor((15360, 160), dtype="float16") = model_params[283]
            lv56_1 = R.call_tir(cls.fused_fused_decode_NT_matmul, (lv420, lv421, lv1844), out_sinfo=R.Tensor((nseq, 1, 15360), dtype="float16"))
            lv1847 = R.call_tir(cls.split, (lv56_1,), out_sinfo=[R.Tensor((nseq, 1, 5120), dtype="float16"), R.Tensor((nseq, 1, 5120), dtype="float16"), R.Tensor((nseq, 1, 5120), dtype="float16")])
            lv1848: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1847[0]
            lv1849 = R.call_tir(cls.reshape, (lv1848,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1850: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1847[1]
            lv1851 = R.call_tir(cls.reshape, (lv1850,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1852: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1847[2]
            lv1853 = R.call_tir(cls.reshape, (lv1852,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1854: R.Object = R.call_pure_packed("vm.builtin.paged_attention_kv_cache_append", lv1827, cls.kv_cache_transpose_append, lv1851, lv1853, R.prim_value(28), sinfo_args=(R.Object,))
            lv1855 = R.call_dps_packed("vm.builtin.paged_attention_kv_cache_attention", (lv1854, cls.attention, lv1849, R.prim_value(28), R.prim_value(1), R.prim_value(T.float32(1)), R.prim_value(10000)), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1856 = R.call_tir(cls.reshape1, (lv1855,), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv423: R.Tensor((5120, 640), dtype="uint32") = model_params[284]
            lv424: R.Tensor((5120, 160), dtype="float16") = model_params[285]
            lv136_1 = R.call_tir(cls.fused_fused_decode1_fused_NT_matmul1_add, (lv423, lv424, lv1856, lv135_2), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv285_1: R.Tensor((5120,), dtype="float16") = model_params[291]
            lv1860 = R.call_tir(cls.rms_norm1, (lv136_1, lv285_1), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv427: R.Tensor((27648, 640), dtype="uint32") = model_params[286]
            lv428: R.Tensor((27648, 160), dtype="float16") = model_params[287]
            lv57_1 = R.call_tir(cls.fused_fused_decode2_NT_matmul2, (lv427, lv428, lv1860), out_sinfo=R.Tensor((nseq, 1, 27648), dtype="float16"))
            lv430 = R.call_tir(cls.fused_split1_silu_multiply, (lv57_1,), out_sinfo=R.Tensor((nseq, 1, 13824), dtype="float16"))
            lv431: R.Tensor((5120, 1728), dtype="uint32") = model_params[288]
            lv432: R.Tensor((5120, 432), dtype="float16") = model_params[289]
            lv137 = R.call_tir(cls.fused_fused_decode3_fused_NT_matmul3_add, (lv431, lv432, lv430, lv136_1), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv290: R.Tensor((5120,), dtype="float16") = model_params[300]
            lv1871 = R.call_tir(cls.rms_norm1, (lv137, lv290), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv435: R.Tensor((15360, 640), dtype="uint32") = model_params[292]
            lv436: R.Tensor((15360, 160), dtype="float16") = model_params[293]
            lv58 = R.call_tir(cls.fused_fused_decode_NT_matmul, (lv435, lv436, lv1871), out_sinfo=R.Tensor((nseq, 1, 15360), dtype="float16"))
            lv1874 = R.call_tir(cls.split, (lv58,), out_sinfo=[R.Tensor((nseq, 1, 5120), dtype="float16"), R.Tensor((nseq, 1, 5120), dtype="float16"), R.Tensor((nseq, 1, 5120), dtype="float16")])
            lv1875: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1874[0]
            lv1876 = R.call_tir(cls.reshape, (lv1875,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1877: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1874[1]
            lv1878 = R.call_tir(cls.reshape, (lv1877,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1879: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1874[2]
            lv1880 = R.call_tir(cls.reshape, (lv1879,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1881: R.Object = R.call_pure_packed("vm.builtin.paged_attention_kv_cache_append", lv1854, cls.kv_cache_transpose_append, lv1878, lv1880, R.prim_value(29), sinfo_args=(R.Object,))
            lv1882 = R.call_dps_packed("vm.builtin.paged_attention_kv_cache_attention", (lv1881, cls.attention, lv1876, R.prim_value(29), R.prim_value(1), R.prim_value(T.float32(1)), R.prim_value(10000)), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1883 = R.call_tir(cls.reshape1, (lv1882,), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv438: R.Tensor((5120, 640), dtype="uint32") = model_params[294]
            lv439: R.Tensor((5120, 160), dtype="float16") = model_params[295]
            lv138_1 = R.call_tir(cls.fused_fused_decode1_fused_NT_matmul1_add, (lv438, lv439, lv1883, lv137), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv295_1: R.Tensor((5120,), dtype="float16") = model_params[301]
            lv1887 = R.call_tir(cls.rms_norm1, (lv138_1, lv295_1), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv442: R.Tensor((27648, 640), dtype="uint32") = model_params[296]
            lv443: R.Tensor((27648, 160), dtype="float16") = model_params[297]
            lv59 = R.call_tir(cls.fused_fused_decode2_NT_matmul2, (lv442, lv443, lv1887), out_sinfo=R.Tensor((nseq, 1, 27648), dtype="float16"))
            lv445 = R.call_tir(cls.fused_split1_silu_multiply, (lv59,), out_sinfo=R.Tensor((nseq, 1, 13824), dtype="float16"))
            lv446: R.Tensor((5120, 1728), dtype="uint32") = model_params[298]
            lv447: R.Tensor((5120, 432), dtype="float16") = model_params[299]
            lv139_1 = R.call_tir(cls.fused_fused_decode3_fused_NT_matmul3_add, (lv446, lv447, lv445, lv138_1), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv300_1: R.Tensor((5120,), dtype="float16") = model_params[310]
            lv1898 = R.call_tir(cls.rms_norm1, (lv139_1, lv300_1), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv450: R.Tensor((15360, 640), dtype="uint32") = model_params[302]
            lv451: R.Tensor((15360, 160), dtype="float16") = model_params[303]
            lv60_2 = R.call_tir(cls.fused_fused_decode_NT_matmul, (lv450, lv451, lv1898), out_sinfo=R.Tensor((nseq, 1, 15360), dtype="float16"))
            lv1901 = R.call_tir(cls.split, (lv60_2,), out_sinfo=[R.Tensor((nseq, 1, 5120), dtype="float16"), R.Tensor((nseq, 1, 5120), dtype="float16"), R.Tensor((nseq, 1, 5120), dtype="float16")])
            lv1902: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1901[0]
            lv1903 = R.call_tir(cls.reshape, (lv1902,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1904: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1901[1]
            lv1905 = R.call_tir(cls.reshape, (lv1904,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1906: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1901[2]
            lv1907 = R.call_tir(cls.reshape, (lv1906,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1908: R.Object = R.call_pure_packed("vm.builtin.paged_attention_kv_cache_append", lv1881, cls.kv_cache_transpose_append, lv1905, lv1907, R.prim_value(30), sinfo_args=(R.Object,))
            lv1909 = R.call_dps_packed("vm.builtin.paged_attention_kv_cache_attention", (lv1908, cls.attention, lv1903, R.prim_value(30), R.prim_value(1), R.prim_value(T.float32(1)), R.prim_value(10000)), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1910 = R.call_tir(cls.reshape1, (lv1909,), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv453: R.Tensor((5120, 640), dtype="uint32") = model_params[304]
            lv454: R.Tensor((5120, 160), dtype="float16") = model_params[305]
            lv140_1 = R.call_tir(cls.fused_fused_decode1_fused_NT_matmul1_add, (lv453, lv454, lv1910, lv139_1), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv305: R.Tensor((5120,), dtype="float16") = model_params[311]
            lv1914 = R.call_tir(cls.rms_norm1, (lv140_1, lv305), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv457: R.Tensor((27648, 640), dtype="uint32") = model_params[306]
            lv458: R.Tensor((27648, 160), dtype="float16") = model_params[307]
            lv61_1 = R.call_tir(cls.fused_fused_decode2_NT_matmul2, (lv457, lv458, lv1914), out_sinfo=R.Tensor((nseq, 1, 27648), dtype="float16"))
            lv460 = R.call_tir(cls.fused_split1_silu_multiply, (lv61_1,), out_sinfo=R.Tensor((nseq, 1, 13824), dtype="float16"))
            lv461: R.Tensor((5120, 1728), dtype="uint32") = model_params[308]
            lv462: R.Tensor((5120, 432), dtype="float16") = model_params[309]
            lv141 = R.call_tir(cls.fused_fused_decode3_fused_NT_matmul3_add, (lv461, lv462, lv460, lv140_1), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv310_1: R.Tensor((5120,), dtype="float16") = model_params[320]
            lv1925 = R.call_tir(cls.rms_norm1, (lv141, lv310_1), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv465: R.Tensor((15360, 640), dtype="uint32") = model_params[312]
            lv466: R.Tensor((15360, 160), dtype="float16") = model_params[313]
            lv62 = R.call_tir(cls.fused_fused_decode_NT_matmul, (lv465, lv466, lv1925), out_sinfo=R.Tensor((nseq, 1, 15360), dtype="float16"))
            lv1928 = R.call_tir(cls.split, (lv62,), out_sinfo=[R.Tensor((nseq, 1, 5120), dtype="float16"), R.Tensor((nseq, 1, 5120), dtype="float16"), R.Tensor((nseq, 1, 5120), dtype="float16")])
            lv1929: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1928[0]
            lv1930 = R.call_tir(cls.reshape, (lv1929,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1931: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1928[1]
            lv1932 = R.call_tir(cls.reshape, (lv1931,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1933: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1928[2]
            lv1934 = R.call_tir(cls.reshape, (lv1933,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1935: R.Object = R.call_pure_packed("vm.builtin.paged_attention_kv_cache_append", lv1908, cls.kv_cache_transpose_append, lv1932, lv1934, R.prim_value(31), sinfo_args=(R.Object,))
            lv1936 = R.call_dps_packed("vm.builtin.paged_attention_kv_cache_attention", (lv1935, cls.attention, lv1930, R.prim_value(31), R.prim_value(1), R.prim_value(T.float32(1)), R.prim_value(10000)), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1937 = R.call_tir(cls.reshape1, (lv1936,), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv468: R.Tensor((5120, 640), dtype="uint32") = model_params[314]
            lv469: R.Tensor((5120, 160), dtype="float16") = model_params[315]
            lv142_1 = R.call_tir(cls.fused_fused_decode1_fused_NT_matmul1_add, (lv468, lv469, lv1937, lv141), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv315_1: R.Tensor((5120,), dtype="float16") = model_params[321]
            lv1941 = R.call_tir(cls.rms_norm1, (lv142_1, lv315_1), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv472: R.Tensor((27648, 640), dtype="uint32") = model_params[316]
            lv473: R.Tensor((27648, 160), dtype="float16") = model_params[317]
            lv63_1 = R.call_tir(cls.fused_fused_decode2_NT_matmul2, (lv472, lv473, lv1941), out_sinfo=R.Tensor((nseq, 1, 27648), dtype="float16"))
            lv475 = R.call_tir(cls.fused_split1_silu_multiply, (lv63_1,), out_sinfo=R.Tensor((nseq, 1, 13824), dtype="float16"))
            lv476: R.Tensor((5120, 1728), dtype="uint32") = model_params[318]
            lv477: R.Tensor((5120, 432), dtype="float16") = model_params[319]
            lv143_1 = R.call_tir(cls.fused_fused_decode3_fused_NT_matmul3_add, (lv476, lv477, lv475, lv142_1), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv320: R.Tensor((5120,), dtype="float16") = model_params[330]
            lv1952 = R.call_tir(cls.rms_norm1, (lv143_1, lv320), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv480: R.Tensor((15360, 640), dtype="uint32") = model_params[322]
            lv481: R.Tensor((15360, 160), dtype="float16") = model_params[323]
            lv64_1 = R.call_tir(cls.fused_fused_decode_NT_matmul, (lv480, lv481, lv1952), out_sinfo=R.Tensor((nseq, 1, 15360), dtype="float16"))
            lv1955 = R.call_tir(cls.split, (lv64_1,), out_sinfo=[R.Tensor((nseq, 1, 5120), dtype="float16"), R.Tensor((nseq, 1, 5120), dtype="float16"), R.Tensor((nseq, 1, 5120), dtype="float16")])
            lv1956: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1955[0]
            lv1957 = R.call_tir(cls.reshape, (lv1956,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1958: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1955[1]
            lv1959 = R.call_tir(cls.reshape, (lv1958,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1960: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1955[2]
            lv1961 = R.call_tir(cls.reshape, (lv1960,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1962: R.Object = R.call_pure_packed("vm.builtin.paged_attention_kv_cache_append", lv1935, cls.kv_cache_transpose_append, lv1959, lv1961, R.prim_value(32), sinfo_args=(R.Object,))
            lv1963 = R.call_dps_packed("vm.builtin.paged_attention_kv_cache_attention", (lv1962, cls.attention, lv1957, R.prim_value(32), R.prim_value(1), R.prim_value(T.float32(1)), R.prim_value(10000)), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1964 = R.call_tir(cls.reshape1, (lv1963,), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv483: R.Tensor((5120, 640), dtype="uint32") = model_params[324]
            lv484: R.Tensor((5120, 160), dtype="float16") = model_params[325]
            lv144 = R.call_tir(cls.fused_fused_decode1_fused_NT_matmul1_add, (lv483, lv484, lv1964, lv143_1), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv325_1: R.Tensor((5120,), dtype="float16") = model_params[331]
            lv1968 = R.call_tir(cls.rms_norm1, (lv144, lv325_1), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv487: R.Tensor((27648, 640), dtype="uint32") = model_params[326]
            lv488: R.Tensor((27648, 160), dtype="float16") = model_params[327]
            lv65_1 = R.call_tir(cls.fused_fused_decode2_NT_matmul2, (lv487, lv488, lv1968), out_sinfo=R.Tensor((nseq, 1, 27648), dtype="float16"))
            lv490 = R.call_tir(cls.fused_split1_silu_multiply, (lv65_1,), out_sinfo=R.Tensor((nseq, 1, 13824), dtype="float16"))
            lv491: R.Tensor((5120, 1728), dtype="uint32") = model_params[328]
            lv492: R.Tensor((5120, 432), dtype="float16") = model_params[329]
            lv145_2 = R.call_tir(cls.fused_fused_decode3_fused_NT_matmul3_add, (lv491, lv492, lv490, lv144), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv330_1: R.Tensor((5120,), dtype="float16") = model_params[340]
            lv1979 = R.call_tir(cls.rms_norm1, (lv145_2, lv330_1), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv495: R.Tensor((15360, 640), dtype="uint32") = model_params[332]
            lv496: R.Tensor((15360, 160), dtype="float16") = model_params[333]
            lv66 = R.call_tir(cls.fused_fused_decode_NT_matmul, (lv495, lv496, lv1979), out_sinfo=R.Tensor((nseq, 1, 15360), dtype="float16"))
            lv1982 = R.call_tir(cls.split, (lv66,), out_sinfo=[R.Tensor((nseq, 1, 5120), dtype="float16"), R.Tensor((nseq, 1, 5120), dtype="float16"), R.Tensor((nseq, 1, 5120), dtype="float16")])
            lv1983: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1982[0]
            lv1984 = R.call_tir(cls.reshape, (lv1983,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1985: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1982[1]
            lv1986 = R.call_tir(cls.reshape, (lv1985,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1987: R.Tensor((nseq, 1, 5120), dtype="float16") = lv1982[2]
            lv1988 = R.call_tir(cls.reshape, (lv1987,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1989: R.Object = R.call_pure_packed("vm.builtin.paged_attention_kv_cache_append", lv1962, cls.kv_cache_transpose_append, lv1986, lv1988, R.prim_value(33), sinfo_args=(R.Object,))
            lv1990 = R.call_dps_packed("vm.builtin.paged_attention_kv_cache_attention", (lv1989, cls.attention, lv1984, R.prim_value(33), R.prim_value(1), R.prim_value(T.float32(1)), R.prim_value(10000)), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv1991 = R.call_tir(cls.reshape1, (lv1990,), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv498: R.Tensor((5120, 640), dtype="uint32") = model_params[334]
            lv499: R.Tensor((5120, 160), dtype="float16") = model_params[335]
            lv146_1 = R.call_tir(cls.fused_fused_decode1_fused_NT_matmul1_add, (lv498, lv499, lv1991, lv145_2), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv335: R.Tensor((5120,), dtype="float16") = model_params[341]
            lv1995 = R.call_tir(cls.rms_norm1, (lv146_1, lv335), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv502: R.Tensor((27648, 640), dtype="uint32") = model_params[336]
            lv503: R.Tensor((27648, 160), dtype="float16") = model_params[337]
            lv67_1 = R.call_tir(cls.fused_fused_decode2_NT_matmul2, (lv502, lv503, lv1995), out_sinfo=R.Tensor((nseq, 1, 27648), dtype="float16"))
            lv505 = R.call_tir(cls.fused_split1_silu_multiply, (lv67_1,), out_sinfo=R.Tensor((nseq, 1, 13824), dtype="float16"))
            lv506: R.Tensor((5120, 1728), dtype="uint32") = model_params[338]
            lv507: R.Tensor((5120, 432), dtype="float16") = model_params[339]
            lv147_1 = R.call_tir(cls.fused_fused_decode3_fused_NT_matmul3_add, (lv506, lv507, lv505, lv146_1), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv340_1: R.Tensor((5120,), dtype="float16") = model_params[350]
            lv2006 = R.call_tir(cls.rms_norm1, (lv147_1, lv340_1), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv510: R.Tensor((15360, 640), dtype="uint32") = model_params[342]
            lv511: R.Tensor((15360, 160), dtype="float16") = model_params[343]
            lv68_1 = R.call_tir(cls.fused_fused_decode_NT_matmul, (lv510, lv511, lv2006), out_sinfo=R.Tensor((nseq, 1, 15360), dtype="float16"))
            lv2009 = R.call_tir(cls.split, (lv68_1,), out_sinfo=[R.Tensor((nseq, 1, 5120), dtype="float16"), R.Tensor((nseq, 1, 5120), dtype="float16"), R.Tensor((nseq, 1, 5120), dtype="float16")])
            lv2010: R.Tensor((nseq, 1, 5120), dtype="float16") = lv2009[0]
            lv2011 = R.call_tir(cls.reshape, (lv2010,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv2012: R.Tensor((nseq, 1, 5120), dtype="float16") = lv2009[1]
            lv2013 = R.call_tir(cls.reshape, (lv2012,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv2014: R.Tensor((nseq, 1, 5120), dtype="float16") = lv2009[2]
            lv2015 = R.call_tir(cls.reshape, (lv2014,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv2016: R.Object = R.call_pure_packed("vm.builtin.paged_attention_kv_cache_append", lv1989, cls.kv_cache_transpose_append, lv2013, lv2015, R.prim_value(34), sinfo_args=(R.Object,))
            lv2017 = R.call_dps_packed("vm.builtin.paged_attention_kv_cache_attention", (lv2016, cls.attention, lv2011, R.prim_value(34), R.prim_value(1), R.prim_value(T.float32(1)), R.prim_value(10000)), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv2018 = R.call_tir(cls.reshape1, (lv2017,), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv513: R.Tensor((5120, 640), dtype="uint32") = model_params[344]
            lv514: R.Tensor((5120, 160), dtype="float16") = model_params[345]
            lv148 = R.call_tir(cls.fused_fused_decode1_fused_NT_matmul1_add, (lv513, lv514, lv2018, lv147_1), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv345_1: R.Tensor((5120,), dtype="float16") = model_params[351]
            lv2022 = R.call_tir(cls.rms_norm1, (lv148, lv345_1), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv517: R.Tensor((27648, 640), dtype="uint32") = model_params[346]
            lv518: R.Tensor((27648, 160), dtype="float16") = model_params[347]
            lv69 = R.call_tir(cls.fused_fused_decode2_NT_matmul2, (lv517, lv518, lv2022), out_sinfo=R.Tensor((nseq, 1, 27648), dtype="float16"))
            lv520 = R.call_tir(cls.fused_split1_silu_multiply, (lv69,), out_sinfo=R.Tensor((nseq, 1, 13824), dtype="float16"))
            lv521: R.Tensor((5120, 1728), dtype="uint32") = model_params[348]
            lv522: R.Tensor((5120, 432), dtype="float16") = model_params[349]
            lv149 = R.call_tir(cls.fused_fused_decode3_fused_NT_matmul3_add, (lv521, lv522, lv520, lv148), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv350: R.Tensor((5120,), dtype="float16") = model_params[360]
            lv2033 = R.call_tir(cls.rms_norm1, (lv149, lv350), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv525: R.Tensor((15360, 640), dtype="uint32") = model_params[352]
            lv526: R.Tensor((15360, 160), dtype="float16") = model_params[353]
            lv70_2 = R.call_tir(cls.fused_fused_decode_NT_matmul, (lv525, lv526, lv2033), out_sinfo=R.Tensor((nseq, 1, 15360), dtype="float16"))
            lv2036 = R.call_tir(cls.split, (lv70_2,), out_sinfo=[R.Tensor((nseq, 1, 5120), dtype="float16"), R.Tensor((nseq, 1, 5120), dtype="float16"), R.Tensor((nseq, 1, 5120), dtype="float16")])
            lv2037: R.Tensor((nseq, 1, 5120), dtype="float16") = lv2036[0]
            lv2038 = R.call_tir(cls.reshape, (lv2037,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv2039: R.Tensor((nseq, 1, 5120), dtype="float16") = lv2036[1]
            lv2040 = R.call_tir(cls.reshape, (lv2039,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv2041: R.Tensor((nseq, 1, 5120), dtype="float16") = lv2036[2]
            lv2042 = R.call_tir(cls.reshape, (lv2041,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv2043: R.Object = R.call_pure_packed("vm.builtin.paged_attention_kv_cache_append", lv2016, cls.kv_cache_transpose_append, lv2040, lv2042, R.prim_value(35), sinfo_args=(R.Object,))
            lv2044 = R.call_dps_packed("vm.builtin.paged_attention_kv_cache_attention", (lv2043, cls.attention, lv2038, R.prim_value(35), R.prim_value(1), R.prim_value(T.float32(1)), R.prim_value(10000)), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv2045 = R.call_tir(cls.reshape1, (lv2044,), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv528: R.Tensor((5120, 640), dtype="uint32") = model_params[354]
            lv529: R.Tensor((5120, 160), dtype="float16") = model_params[355]
            lv150_2 = R.call_tir(cls.fused_fused_decode1_fused_NT_matmul1_add, (lv528, lv529, lv2045, lv149), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv355_1: R.Tensor((5120,), dtype="float16") = model_params[361]
            lv2049 = R.call_tir(cls.rms_norm1, (lv150_2, lv355_1), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv532: R.Tensor((27648, 640), dtype="uint32") = model_params[356]
            lv533: R.Tensor((27648, 160), dtype="float16") = model_params[357]
            lv71_1 = R.call_tir(cls.fused_fused_decode2_NT_matmul2, (lv532, lv533, lv2049), out_sinfo=R.Tensor((nseq, 1, 27648), dtype="float16"))
            lv535 = R.call_tir(cls.fused_split1_silu_multiply, (lv71_1,), out_sinfo=R.Tensor((nseq, 1, 13824), dtype="float16"))
            lv536: R.Tensor((5120, 1728), dtype="uint32") = model_params[358]
            lv537: R.Tensor((5120, 432), dtype="float16") = model_params[359]
            lv151_1 = R.call_tir(cls.fused_fused_decode3_fused_NT_matmul3_add, (lv536, lv537, lv535, lv150_2), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv360_1: R.Tensor((5120,), dtype="float16") = model_params[370]
            lv2060 = R.call_tir(cls.rms_norm1, (lv151_1, lv360_1), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv540: R.Tensor((15360, 640), dtype="uint32") = model_params[362]
            lv541: R.Tensor((15360, 160), dtype="float16") = model_params[363]
            lv72_1 = R.call_tir(cls.fused_fused_decode_NT_matmul, (lv540, lv541, lv2060), out_sinfo=R.Tensor((nseq, 1, 15360), dtype="float16"))
            lv2063 = R.call_tir(cls.split, (lv72_1,), out_sinfo=[R.Tensor((nseq, 1, 5120), dtype="float16"), R.Tensor((nseq, 1, 5120), dtype="float16"), R.Tensor((nseq, 1, 5120), dtype="float16")])
            lv2064: R.Tensor((nseq, 1, 5120), dtype="float16") = lv2063[0]
            lv2065 = R.call_tir(cls.reshape, (lv2064,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv2066: R.Tensor((nseq, 1, 5120), dtype="float16") = lv2063[1]
            lv2067 = R.call_tir(cls.reshape, (lv2066,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv2068: R.Tensor((nseq, 1, 5120), dtype="float16") = lv2063[2]
            lv2069 = R.call_tir(cls.reshape, (lv2068,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv2070: R.Object = R.call_pure_packed("vm.builtin.paged_attention_kv_cache_append", lv2043, cls.kv_cache_transpose_append, lv2067, lv2069, R.prim_value(36), sinfo_args=(R.Object,))
            lv2071 = R.call_dps_packed("vm.builtin.paged_attention_kv_cache_attention", (lv2070, cls.attention, lv2065, R.prim_value(36), R.prim_value(1), R.prim_value(T.float32(1)), R.prim_value(10000)), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv2072 = R.call_tir(cls.reshape1, (lv2071,), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv543: R.Tensor((5120, 640), dtype="uint32") = model_params[364]
            lv544: R.Tensor((5120, 160), dtype="float16") = model_params[365]
            lv152 = R.call_tir(cls.fused_fused_decode1_fused_NT_matmul1_add, (lv543, lv544, lv2072, lv151_1), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv365: R.Tensor((5120,), dtype="float16") = model_params[371]
            lv2076 = R.call_tir(cls.rms_norm1, (lv152, lv365), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv547: R.Tensor((27648, 640), dtype="uint32") = model_params[366]
            lv548: R.Tensor((27648, 160), dtype="float16") = model_params[367]
            lv73 = R.call_tir(cls.fused_fused_decode2_NT_matmul2, (lv547, lv548, lv2076), out_sinfo=R.Tensor((nseq, 1, 27648), dtype="float16"))
            lv550 = R.call_tir(cls.fused_split1_silu_multiply, (lv73,), out_sinfo=R.Tensor((nseq, 1, 13824), dtype="float16"))
            lv551: R.Tensor((5120, 1728), dtype="uint32") = model_params[368]
            lv552: R.Tensor((5120, 432), dtype="float16") = model_params[369]
            lv153_1 = R.call_tir(cls.fused_fused_decode3_fused_NT_matmul3_add, (lv551, lv552, lv550, lv152), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv370_1: R.Tensor((5120,), dtype="float16") = model_params[380]
            lv2087 = R.call_tir(cls.rms_norm1, (lv153_1, lv370_1), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv555: R.Tensor((15360, 640), dtype="uint32") = model_params[372]
            lv556: R.Tensor((15360, 160), dtype="float16") = model_params[373]
            lv74 = R.call_tir(cls.fused_fused_decode_NT_matmul, (lv555, lv556, lv2087), out_sinfo=R.Tensor((nseq, 1, 15360), dtype="float16"))
            lv2090 = R.call_tir(cls.split, (lv74,), out_sinfo=[R.Tensor((nseq, 1, 5120), dtype="float16"), R.Tensor((nseq, 1, 5120), dtype="float16"), R.Tensor((nseq, 1, 5120), dtype="float16")])
            lv2091: R.Tensor((nseq, 1, 5120), dtype="float16") = lv2090[0]
            lv2092 = R.call_tir(cls.reshape, (lv2091,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv2093: R.Tensor((nseq, 1, 5120), dtype="float16") = lv2090[1]
            lv2094 = R.call_tir(cls.reshape, (lv2093,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv2095: R.Tensor((nseq, 1, 5120), dtype="float16") = lv2090[2]
            lv2096 = R.call_tir(cls.reshape, (lv2095,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv2097: R.Object = R.call_pure_packed("vm.builtin.paged_attention_kv_cache_append", lv2070, cls.kv_cache_transpose_append, lv2094, lv2096, R.prim_value(37), sinfo_args=(R.Object,))
            lv2098 = R.call_dps_packed("vm.builtin.paged_attention_kv_cache_attention", (lv2097, cls.attention, lv2092, R.prim_value(37), R.prim_value(1), R.prim_value(T.float32(1)), R.prim_value(10000)), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv2099 = R.call_tir(cls.reshape1, (lv2098,), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv558: R.Tensor((5120, 640), dtype="uint32") = model_params[374]
            lv559: R.Tensor((5120, 160), dtype="float16") = model_params[375]
            lv154_1 = R.call_tir(cls.fused_fused_decode1_fused_NT_matmul1_add, (lv558, lv559, lv2099, lv153_1), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv375_1: R.Tensor((5120,), dtype="float16") = model_params[381]
            lv2103 = R.call_tir(cls.rms_norm1, (lv154_1, lv375_1), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv562: R.Tensor((27648, 640), dtype="uint32") = model_params[376]
            lv563: R.Tensor((27648, 160), dtype="float16") = model_params[377]
            lv75_2 = R.call_tir(cls.fused_fused_decode2_NT_matmul2, (lv562, lv563, lv2103), out_sinfo=R.Tensor((nseq, 1, 27648), dtype="float16"))
            lv565 = R.call_tir(cls.fused_split1_silu_multiply, (lv75_2,), out_sinfo=R.Tensor((nseq, 1, 13824), dtype="float16"))
            lv566: R.Tensor((5120, 1728), dtype="uint32") = model_params[378]
            lv567: R.Tensor((5120, 432), dtype="float16") = model_params[379]
            lv155_1 = R.call_tir(cls.fused_fused_decode3_fused_NT_matmul3_add, (lv566, lv567, lv565, lv154_1), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv380: R.Tensor((5120,), dtype="float16") = model_params[390]
            lv2114 = R.call_tir(cls.rms_norm1, (lv155_1, lv380), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv570: R.Tensor((15360, 640), dtype="uint32") = model_params[382]
            lv571: R.Tensor((15360, 160), dtype="float16") = model_params[383]
            lv76_1 = R.call_tir(cls.fused_fused_decode_NT_matmul, (lv570, lv571, lv2114), out_sinfo=R.Tensor((nseq, 1, 15360), dtype="float16"))
            lv2117 = R.call_tir(cls.split, (lv76_1,), out_sinfo=[R.Tensor((nseq, 1, 5120), dtype="float16"), R.Tensor((nseq, 1, 5120), dtype="float16"), R.Tensor((nseq, 1, 5120), dtype="float16")])
            lv2118: R.Tensor((nseq, 1, 5120), dtype="float16") = lv2117[0]
            lv2119 = R.call_tir(cls.reshape, (lv2118,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv2120: R.Tensor((nseq, 1, 5120), dtype="float16") = lv2117[1]
            lv2121 = R.call_tir(cls.reshape, (lv2120,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv2122: R.Tensor((nseq, 1, 5120), dtype="float16") = lv2117[2]
            lv2123 = R.call_tir(cls.reshape, (lv2122,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv2124: R.Object = R.call_pure_packed("vm.builtin.paged_attention_kv_cache_append", lv2097, cls.kv_cache_transpose_append, lv2121, lv2123, R.prim_value(38), sinfo_args=(R.Object,))
            lv2125 = R.call_dps_packed("vm.builtin.paged_attention_kv_cache_attention", (lv2124, cls.attention, lv2119, R.prim_value(38), R.prim_value(1), R.prim_value(T.float32(1)), R.prim_value(10000)), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv2126 = R.call_tir(cls.reshape1, (lv2125,), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv573: R.Tensor((5120, 640), dtype="uint32") = model_params[384]
            lv574: R.Tensor((5120, 160), dtype="float16") = model_params[385]
            lv156 = R.call_tir(cls.fused_fused_decode1_fused_NT_matmul1_add, (lv573, lv574, lv2126, lv155_1), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv385_1: R.Tensor((5120,), dtype="float16") = model_params[391]
            lv2130 = R.call_tir(cls.rms_norm1, (lv156, lv385_1), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv577: R.Tensor((27648, 640), dtype="uint32") = model_params[386]
            lv578: R.Tensor((27648, 160), dtype="float16") = model_params[387]
            lv77 = R.call_tir(cls.fused_fused_decode2_NT_matmul2, (lv577, lv578, lv2130), out_sinfo=R.Tensor((nseq, 1, 27648), dtype="float16"))
            lv580 = R.call_tir(cls.fused_split1_silu_multiply, (lv77,), out_sinfo=R.Tensor((nseq, 1, 13824), dtype="float16"))
            lv581: R.Tensor((5120, 1728), dtype="uint32") = model_params[388]
            lv582: R.Tensor((5120, 432), dtype="float16") = model_params[389]
            lv157_1 = R.call_tir(cls.fused_fused_decode3_fused_NT_matmul3_add, (lv581, lv582, lv580, lv156), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv390_1: R.Tensor((5120,), dtype="float16") = model_params[400]
            lv2141 = R.call_tir(cls.rms_norm1, (lv157_1, lv390_1), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv585: R.Tensor((15360, 640), dtype="uint32") = model_params[392]
            lv586: R.Tensor((15360, 160), dtype="float16") = model_params[393]
            lv78_1 = R.call_tir(cls.fused_fused_decode_NT_matmul, (lv585, lv586, lv2141), out_sinfo=R.Tensor((nseq, 1, 15360), dtype="float16"))
            lv2144 = R.call_tir(cls.split, (lv78_1,), out_sinfo=[R.Tensor((nseq, 1, 5120), dtype="float16"), R.Tensor((nseq, 1, 5120), dtype="float16"), R.Tensor((nseq, 1, 5120), dtype="float16")])
            lv2145: R.Tensor((nseq, 1, 5120), dtype="float16") = lv2144[0]
            lv2146 = R.call_tir(cls.reshape, (lv2145,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv2147: R.Tensor((nseq, 1, 5120), dtype="float16") = lv2144[1]
            lv2148 = R.call_tir(cls.reshape, (lv2147,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv2149: R.Tensor((nseq, 1, 5120), dtype="float16") = lv2144[2]
            lv2150 = R.call_tir(cls.reshape, (lv2149,), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv2151: R.Object = R.call_pure_packed("vm.builtin.paged_attention_kv_cache_append", lv2124, cls.kv_cache_transpose_append, lv2148, lv2150, R.prim_value(39), sinfo_args=(R.Object,))
            lv2152 = R.call_dps_packed("vm.builtin.paged_attention_kv_cache_attention", (lv2151, cls.attention, lv2146, R.prim_value(39), R.prim_value(1), R.prim_value(T.float32(1)), R.prim_value(10000)), out_sinfo=R.Tensor((nseq, 1, 40, 128), dtype="float16"))
            lv2153 = R.call_tir(cls.reshape1, (lv2152,), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv588: R.Tensor((5120, 640), dtype="uint32") = model_params[394]
            lv589: R.Tensor((5120, 160), dtype="float16") = model_params[395]
            lv158_1 = R.call_tir(cls.fused_fused_decode1_fused_NT_matmul1_add, (lv588, lv589, lv2153, lv157_1), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv395: R.Tensor((5120,), dtype="float16") = model_params[401]
            lv2157 = R.call_tir(cls.rms_norm1, (lv158_1, lv395), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv592: R.Tensor((27648, 640), dtype="uint32") = model_params[396]
            lv593: R.Tensor((27648, 160), dtype="float16") = model_params[397]
            lv79_1 = R.call_tir(cls.fused_fused_decode2_NT_matmul2, (lv592, lv593, lv2157), out_sinfo=R.Tensor((nseq, 1, 27648), dtype="float16"))
            lv595 = R.call_tir(cls.fused_split1_silu_multiply, (lv79_1,), out_sinfo=R.Tensor((nseq, 1, 13824), dtype="float16"))
            lv596: R.Tensor((5120, 1728), dtype="uint32") = model_params[398]
            lv597: R.Tensor((5120, 432), dtype="float16") = model_params[399]
            lv159 = R.call_tir(cls.fused_fused_decode3_fused_NT_matmul3_add, (lv596, lv597, lv595, lv158_1), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv400_1: R.Tensor((5120,), dtype="float16") = model_params[402]
            lv2168 = R.call_tir(cls.rms_norm1, (lv159, lv400_1), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv2169 = R.call_tir(cls.slice1, (lv2168,), out_sinfo=R.Tensor((nseq, 1, 5120), dtype="float16"))
            lv600: R.Tensor((vocab_size, 640), dtype="uint32") = model_params[403]
            lv601: R.Tensor((vocab_size, 160), dtype="float16") = model_params[404]
            lv80_2 = R.call_tir(cls.fused_fused_decode4_fused_NT_matmul4_cast, (lv600, lv601, lv2169), out_sinfo=R.Tensor((nseq, 1, vocab_size), dtype="float32"))
            gv2: R.Tuple(R.Tensor((nseq, 1, vocab_size), dtype="float32"), R.Object) = lv80_2, lv2151
            R.output(gv2)
        return gv2

    @R.function
    def embed(input_ids: R.Tensor(("nseq", "n"), dtype="int32"), model_params: R.Tuple(R.Tensor(("vocab_size", 640), dtype="uint32"), R.Tensor(("vocab_size", 160), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor(("vocab_size_1", 640), dtype="uint32"), R.Tensor(("vocab_size_1", 160), dtype="float16"), R.Tensor(("cache_len", 128), dtype="float16"), R.Tensor(("cache_len_1", 128), dtype="float16"))) -> R.Tensor(("nseq", "n", 5120), dtype="float16"):
        nseq = T.int64()
        n = T.int64()
        vocab_size = T.int64()
        vocab_size_1 = T.int64()
        cache_len = T.int64()
        cache_len_1 = T.int64()
        R.func_attr({"num_input": 1, "tir_var_upper_bound": {"m": 4096, "n": 4096}})
        cls = Module
        with R.dataflow():
            lv = R.call_tir(cls.reshape3, (input_ids,), out_sinfo=R.Tensor((nseq * n,), dtype="int32"))
            lv604: R.Tensor((vocab_size, 640), dtype="uint32") = model_params[0]
            lv605: R.Tensor((vocab_size, 160), dtype="float16") = model_params[1]
            lv_1 = R.call_tir(cls.fused_fused_decode4_take, (lv604, lv605, lv), out_sinfo=R.Tensor((nseq * n, 5120), dtype="float16"), tir_vars=R.shape([n, nseq]))
            lv2 = R.call_tir(cls.reshape4, (lv_1,), out_sinfo=R.Tensor((nseq, n, 5120), dtype="float16"))
            gv: R.Tensor((nseq, n, 5120), dtype="float16") = lv2
            R.output(gv)
        return gv

    @R.function
    def get_metadata() -> R.Object:
        R.func_attr({"tir_var_upper_bound": {"m": 4096, "n": 4096}})
        return R.str("{\"model_name\": \"Llama-2-13b-chat-hf\", \"max_window_size\": 4096, \"stop_tokens\": [2], \"add_prefix_space\": false}")

    @R.function
    def prefill_with_embed(inputs_embeds: R.Tensor((1, "n", 5120), dtype="float16"), kv_cache: R.Object, model_params: R.Tuple(R.Tensor(("vocab_size_1", 640), dtype="uint32"), R.Tensor(("vocab_size_1", 160), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((15360, 640), dtype="uint32"), R.Tensor((15360, 160), dtype="float16"), R.Tensor((5120, 640), dtype="uint32"), R.Tensor((5120, 160), dtype="float16"), R.Tensor((27648, 640), dtype="uint32"), R.Tensor((27648, 160), dtype="float16"), R.Tensor((5120, 1728), dtype="uint32"), R.Tensor((5120, 432), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor((5120,), dtype="float16"), R.Tensor(("vocab_size", 640), dtype="uint32"), R.Tensor(("vocab_size", 160), dtype="float16"), R.Tensor(("cache_len", 128), dtype="float16"), R.Tensor(("cache_len", 128), dtype="float16"))) -> R.Tuple(R.Tensor((1, 1, "vocab_size"), dtype="float32"), R.Object):
        vocab_size = T.int64()
        n = T.int64()
        vocab_size_1 = T.int64()
        cache_len = T.int64()
        R.func_attr({"num_input": 2, "tir_var_upper_bound": {"m": 4096, "n": 4096}})
        cls = Module
        with R.dataflow():
            lv405: R.Tensor((5120,), dtype="float16") = model_params[10]
            lv3 = R.call_tir(cls.rms_norm, (inputs_embeds, lv405), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv607: R.Tensor((15360, 640), dtype="uint32") = model_params[2]
            lv608: R.Tensor((15360, 160), dtype="float16") = model_params[3]
            lv81 = R.call_tir(cls.fused_fused_decode_NT_matmul5, (lv607, lv608, lv3), out_sinfo=R.Tensor((1, n, 15360), dtype="float16"))
            lv6 = R.call_tir(cls.split2, (lv81,), out_sinfo=[R.Tensor((1, n, 5120), dtype="float16"), R.Tensor((1, n, 5120), dtype="float16"), R.Tensor((1, n, 5120), dtype="float16")])
            lv7: R.Tensor((1, n, 5120), dtype="float16") = lv6[0]
            lv8 = R.call_tir(cls.reshape5, (lv7,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv9: R.Tensor((1, n, 5120), dtype="float16") = lv6[1]
            lv10 = R.call_tir(cls.reshape5, (lv9,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv11: R.Tensor((1, n, 5120), dtype="float16") = lv6[2]
            lv12 = R.call_tir(cls.reshape5, (lv11,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv13: R.Object = R.call_pure_packed("vm.builtin.paged_attention_kv_cache_append", kv_cache, cls.kv_cache_transpose_append, lv10, lv12, R.prim_value(0), sinfo_args=(R.Object,))
            lv14 = R.call_dps_packed("vm.builtin.paged_attention_kv_cache_attention", (lv13, cls.attention, lv8, R.prim_value(0), R.prim_value(1), R.prim_value(T.float32(1)), R.prim_value(10000)), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv15 = R.call_tir(cls.reshape6, (lv14,), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv610: R.Tensor((5120, 640), dtype="uint32") = model_params[4]
            lv611: R.Tensor((5120, 160), dtype="float16") = model_params[5]
            lv = R.call_tir(cls.fused_fused_decode1_fused_NT_matmul6_add1, (lv610, lv611, lv15, inputs_embeds), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv410: R.Tensor((5120,), dtype="float16") = model_params[11]
            lv19 = R.call_tir(cls.rms_norm, (lv, lv410), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv614: R.Tensor((27648, 640), dtype="uint32") = model_params[6]
            lv615: R.Tensor((27648, 160), dtype="float16") = model_params[7]
            lv82 = R.call_tir(cls.fused_fused_decode2_NT_matmul7, (lv614, lv615, lv19), out_sinfo=R.Tensor((1, n, 27648), dtype="float16"))
            lv617 = R.call_tir(cls.fused_split3_silu1_multiply1, (lv82,), out_sinfo=R.Tensor((1, n, 13824), dtype="float16"))
            lv618: R.Tensor((5120, 1728), dtype="uint32") = model_params[8]
            lv619: R.Tensor((5120, 432), dtype="float16") = model_params[9]
            lv1 = R.call_tir(cls.fused_fused_decode3_fused_NT_matmul8_add1, (lv618, lv619, lv617, lv), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv415: R.Tensor((5120,), dtype="float16") = model_params[20]
            lv30 = R.call_tir(cls.rms_norm, (lv1, lv415), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv622: R.Tensor((15360, 640), dtype="uint32") = model_params[12]
            lv623: R.Tensor((15360, 160), dtype="float16") = model_params[13]
            lv83 = R.call_tir(cls.fused_fused_decode_NT_matmul5, (lv622, lv623, lv30), out_sinfo=R.Tensor((1, n, 15360), dtype="float16"))
            lv33 = R.call_tir(cls.split2, (lv83,), out_sinfo=[R.Tensor((1, n, 5120), dtype="float16"), R.Tensor((1, n, 5120), dtype="float16"), R.Tensor((1, n, 5120), dtype="float16")])
            lv34: R.Tensor((1, n, 5120), dtype="float16") = lv33[0]
            lv35 = R.call_tir(cls.reshape5, (lv34,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv36: R.Tensor((1, n, 5120), dtype="float16") = lv33[1]
            lv37 = R.call_tir(cls.reshape5, (lv36,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv38: R.Tensor((1, n, 5120), dtype="float16") = lv33[2]
            lv39 = R.call_tir(cls.reshape5, (lv38,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv40: R.Object = R.call_pure_packed("vm.builtin.paged_attention_kv_cache_append", lv13, cls.kv_cache_transpose_append, lv37, lv39, R.prim_value(1), sinfo_args=(R.Object,))
            lv41 = R.call_dps_packed("vm.builtin.paged_attention_kv_cache_attention", (lv40, cls.attention, lv35, R.prim_value(1), R.prim_value(1), R.prim_value(T.float32(1)), R.prim_value(10000)), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv42 = R.call_tir(cls.reshape6, (lv41,), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv625: R.Tensor((5120, 640), dtype="uint32") = model_params[14]
            lv626: R.Tensor((5120, 160), dtype="float16") = model_params[15]
            lv2 = R.call_tir(cls.fused_fused_decode1_fused_NT_matmul6_add1, (lv625, lv626, lv42, lv1), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv420: R.Tensor((5120,), dtype="float16") = model_params[21]
            lv46 = R.call_tir(cls.rms_norm, (lv2, lv420), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv629: R.Tensor((27648, 640), dtype="uint32") = model_params[16]
            lv630: R.Tensor((27648, 160), dtype="float16") = model_params[17]
            lv84 = R.call_tir(cls.fused_fused_decode2_NT_matmul7, (lv629, lv630, lv46), out_sinfo=R.Tensor((1, n, 27648), dtype="float16"))
            lv632 = R.call_tir(cls.fused_split3_silu1_multiply1, (lv84,), out_sinfo=R.Tensor((1, n, 13824), dtype="float16"))
            lv633: R.Tensor((5120, 1728), dtype="uint32") = model_params[18]
            lv634: R.Tensor((5120, 432), dtype="float16") = model_params[19]
            lv3_1 = R.call_tir(cls.fused_fused_decode3_fused_NT_matmul8_add1, (lv633, lv634, lv632, lv2), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv425: R.Tensor((5120,), dtype="float16") = model_params[30]
            lv57 = R.call_tir(cls.rms_norm, (lv3_1, lv425), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv637: R.Tensor((15360, 640), dtype="uint32") = model_params[22]
            lv638: R.Tensor((15360, 160), dtype="float16") = model_params[23]
            lv85 = R.call_tir(cls.fused_fused_decode_NT_matmul5, (lv637, lv638, lv57), out_sinfo=R.Tensor((1, n, 15360), dtype="float16"))
            lv60 = R.call_tir(cls.split2, (lv85,), out_sinfo=[R.Tensor((1, n, 5120), dtype="float16"), R.Tensor((1, n, 5120), dtype="float16"), R.Tensor((1, n, 5120), dtype="float16")])
            lv61: R.Tensor((1, n, 5120), dtype="float16") = lv60[0]
            lv62 = R.call_tir(cls.reshape5, (lv61,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv63: R.Tensor((1, n, 5120), dtype="float16") = lv60[1]
            lv64 = R.call_tir(cls.reshape5, (lv63,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv65: R.Tensor((1, n, 5120), dtype="float16") = lv60[2]
            lv66 = R.call_tir(cls.reshape5, (lv65,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv67: R.Object = R.call_pure_packed("vm.builtin.paged_attention_kv_cache_append", lv40, cls.kv_cache_transpose_append, lv64, lv66, R.prim_value(2), sinfo_args=(R.Object,))
            lv68 = R.call_dps_packed("vm.builtin.paged_attention_kv_cache_attention", (lv67, cls.attention, lv62, R.prim_value(2), R.prim_value(1), R.prim_value(T.float32(1)), R.prim_value(10000)), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv69 = R.call_tir(cls.reshape6, (lv68,), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv640: R.Tensor((5120, 640), dtype="uint32") = model_params[24]
            lv641: R.Tensor((5120, 160), dtype="float16") = model_params[25]
            lv4 = R.call_tir(cls.fused_fused_decode1_fused_NT_matmul6_add1, (lv640, lv641, lv69, lv3_1), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv430: R.Tensor((5120,), dtype="float16") = model_params[31]
            lv73 = R.call_tir(cls.rms_norm, (lv4, lv430), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv644: R.Tensor((27648, 640), dtype="uint32") = model_params[26]
            lv645: R.Tensor((27648, 160), dtype="float16") = model_params[27]
            lv86 = R.call_tir(cls.fused_fused_decode2_NT_matmul7, (lv644, lv645, lv73), out_sinfo=R.Tensor((1, n, 27648), dtype="float16"))
            lv647 = R.call_tir(cls.fused_split3_silu1_multiply1, (lv86,), out_sinfo=R.Tensor((1, n, 13824), dtype="float16"))
            lv648: R.Tensor((5120, 1728), dtype="uint32") = model_params[28]
            lv649: R.Tensor((5120, 432), dtype="float16") = model_params[29]
            lv5 = R.call_tir(cls.fused_fused_decode3_fused_NT_matmul8_add1, (lv648, lv649, lv647, lv4), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv435: R.Tensor((5120,), dtype="float16") = model_params[40]
            lv84_1 = R.call_tir(cls.rms_norm, (lv5, lv435), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv652: R.Tensor((15360, 640), dtype="uint32") = model_params[32]
            lv653: R.Tensor((15360, 160), dtype="float16") = model_params[33]
            lv87 = R.call_tir(cls.fused_fused_decode_NT_matmul5, (lv652, lv653, lv84_1), out_sinfo=R.Tensor((1, n, 15360), dtype="float16"))
            lv87_1 = R.call_tir(cls.split2, (lv87,), out_sinfo=[R.Tensor((1, n, 5120), dtype="float16"), R.Tensor((1, n, 5120), dtype="float16"), R.Tensor((1, n, 5120), dtype="float16")])
            lv88: R.Tensor((1, n, 5120), dtype="float16") = lv87_1[0]
            lv89 = R.call_tir(cls.reshape5, (lv88,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv90: R.Tensor((1, n, 5120), dtype="float16") = lv87_1[1]
            lv91 = R.call_tir(cls.reshape5, (lv90,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv92: R.Tensor((1, n, 5120), dtype="float16") = lv87_1[2]
            lv93 = R.call_tir(cls.reshape5, (lv92,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv94: R.Object = R.call_pure_packed("vm.builtin.paged_attention_kv_cache_append", lv67, cls.kv_cache_transpose_append, lv91, lv93, R.prim_value(3), sinfo_args=(R.Object,))
            lv95 = R.call_dps_packed("vm.builtin.paged_attention_kv_cache_attention", (lv94, cls.attention, lv89, R.prim_value(3), R.prim_value(1), R.prim_value(T.float32(1)), R.prim_value(10000)), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv96 = R.call_tir(cls.reshape6, (lv95,), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv655: R.Tensor((5120, 640), dtype="uint32") = model_params[34]
            lv656: R.Tensor((5120, 160), dtype="float16") = model_params[35]
            lv6_1 = R.call_tir(cls.fused_fused_decode1_fused_NT_matmul6_add1, (lv655, lv656, lv96, lv5), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv440: R.Tensor((5120,), dtype="float16") = model_params[41]
            lv100 = R.call_tir(cls.rms_norm, (lv6_1, lv440), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv659: R.Tensor((27648, 640), dtype="uint32") = model_params[36]
            lv660: R.Tensor((27648, 160), dtype="float16") = model_params[37]
            lv88_1 = R.call_tir(cls.fused_fused_decode2_NT_matmul7, (lv659, lv660, lv100), out_sinfo=R.Tensor((1, n, 27648), dtype="float16"))
            lv662 = R.call_tir(cls.fused_split3_silu1_multiply1, (lv88_1,), out_sinfo=R.Tensor((1, n, 13824), dtype="float16"))
            lv663: R.Tensor((5120, 1728), dtype="uint32") = model_params[38]
            lv664: R.Tensor((5120, 432), dtype="float16") = model_params[39]
            lv7_1 = R.call_tir(cls.fused_fused_decode3_fused_NT_matmul8_add1, (lv663, lv664, lv662, lv6_1), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv445: R.Tensor((5120,), dtype="float16") = model_params[50]
            lv111 = R.call_tir(cls.rms_norm, (lv7_1, lv445), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv667: R.Tensor((15360, 640), dtype="uint32") = model_params[42]
            lv668: R.Tensor((15360, 160), dtype="float16") = model_params[43]
            lv89_1 = R.call_tir(cls.fused_fused_decode_NT_matmul5, (lv667, lv668, lv111), out_sinfo=R.Tensor((1, n, 15360), dtype="float16"))
            lv114 = R.call_tir(cls.split2, (lv89_1,), out_sinfo=[R.Tensor((1, n, 5120), dtype="float16"), R.Tensor((1, n, 5120), dtype="float16"), R.Tensor((1, n, 5120), dtype="float16")])
            lv115: R.Tensor((1, n, 5120), dtype="float16") = lv114[0]
            lv116 = R.call_tir(cls.reshape5, (lv115,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv117: R.Tensor((1, n, 5120), dtype="float16") = lv114[1]
            lv118 = R.call_tir(cls.reshape5, (lv117,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv119: R.Tensor((1, n, 5120), dtype="float16") = lv114[2]
            lv120 = R.call_tir(cls.reshape5, (lv119,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv121: R.Object = R.call_pure_packed("vm.builtin.paged_attention_kv_cache_append", lv94, cls.kv_cache_transpose_append, lv118, lv120, R.prim_value(4), sinfo_args=(R.Object,))
            lv122 = R.call_dps_packed("vm.builtin.paged_attention_kv_cache_attention", (lv121, cls.attention, lv116, R.prim_value(4), R.prim_value(1), R.prim_value(T.float32(1)), R.prim_value(10000)), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv123 = R.call_tir(cls.reshape6, (lv122,), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv670: R.Tensor((5120, 640), dtype="uint32") = model_params[44]
            lv671: R.Tensor((5120, 160), dtype="float16") = model_params[45]
            lv8_1 = R.call_tir(cls.fused_fused_decode1_fused_NT_matmul6_add1, (lv670, lv671, lv123, lv7_1), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv450: R.Tensor((5120,), dtype="float16") = model_params[51]
            lv127 = R.call_tir(cls.rms_norm, (lv8_1, lv450), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv674: R.Tensor((27648, 640), dtype="uint32") = model_params[46]
            lv675: R.Tensor((27648, 160), dtype="float16") = model_params[47]
            lv90_1 = R.call_tir(cls.fused_fused_decode2_NT_matmul7, (lv674, lv675, lv127), out_sinfo=R.Tensor((1, n, 27648), dtype="float16"))
            lv677 = R.call_tir(cls.fused_split3_silu1_multiply1, (lv90_1,), out_sinfo=R.Tensor((1, n, 13824), dtype="float16"))
            lv678: R.Tensor((5120, 1728), dtype="uint32") = model_params[48]
            lv679: R.Tensor((5120, 432), dtype="float16") = model_params[49]
            lv9_1 = R.call_tir(cls.fused_fused_decode3_fused_NT_matmul8_add1, (lv678, lv679, lv677, lv8_1), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv455: R.Tensor((5120,), dtype="float16") = model_params[60]
            lv138 = R.call_tir(cls.rms_norm, (lv9_1, lv455), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv682: R.Tensor((15360, 640), dtype="uint32") = model_params[52]
            lv683: R.Tensor((15360, 160), dtype="float16") = model_params[53]
            lv91_1 = R.call_tir(cls.fused_fused_decode_NT_matmul5, (lv682, lv683, lv138), out_sinfo=R.Tensor((1, n, 15360), dtype="float16"))
            lv141 = R.call_tir(cls.split2, (lv91_1,), out_sinfo=[R.Tensor((1, n, 5120), dtype="float16"), R.Tensor((1, n, 5120), dtype="float16"), R.Tensor((1, n, 5120), dtype="float16")])
            lv142: R.Tensor((1, n, 5120), dtype="float16") = lv141[0]
            lv143 = R.call_tir(cls.reshape5, (lv142,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv144: R.Tensor((1, n, 5120), dtype="float16") = lv141[1]
            lv145 = R.call_tir(cls.reshape5, (lv144,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv146: R.Tensor((1, n, 5120), dtype="float16") = lv141[2]
            lv147 = R.call_tir(cls.reshape5, (lv146,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv148: R.Object = R.call_pure_packed("vm.builtin.paged_attention_kv_cache_append", lv121, cls.kv_cache_transpose_append, lv145, lv147, R.prim_value(5), sinfo_args=(R.Object,))
            lv149 = R.call_dps_packed("vm.builtin.paged_attention_kv_cache_attention", (lv148, cls.attention, lv143, R.prim_value(5), R.prim_value(1), R.prim_value(T.float32(1)), R.prim_value(10000)), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv150 = R.call_tir(cls.reshape6, (lv149,), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv685: R.Tensor((5120, 640), dtype="uint32") = model_params[54]
            lv686: R.Tensor((5120, 160), dtype="float16") = model_params[55]
            lv10_1 = R.call_tir(cls.fused_fused_decode1_fused_NT_matmul6_add1, (lv685, lv686, lv150, lv9_1), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv460: R.Tensor((5120,), dtype="float16") = model_params[61]
            lv154 = R.call_tir(cls.rms_norm, (lv10_1, lv460), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv689: R.Tensor((27648, 640), dtype="uint32") = model_params[56]
            lv690: R.Tensor((27648, 160), dtype="float16") = model_params[57]
            lv92_1 = R.call_tir(cls.fused_fused_decode2_NT_matmul7, (lv689, lv690, lv154), out_sinfo=R.Tensor((1, n, 27648), dtype="float16"))
            lv692 = R.call_tir(cls.fused_split3_silu1_multiply1, (lv92_1,), out_sinfo=R.Tensor((1, n, 13824), dtype="float16"))
            lv693: R.Tensor((5120, 1728), dtype="uint32") = model_params[58]
            lv694: R.Tensor((5120, 432), dtype="float16") = model_params[59]
            lv11_1 = R.call_tir(cls.fused_fused_decode3_fused_NT_matmul8_add1, (lv693, lv694, lv692, lv10_1), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv465: R.Tensor((5120,), dtype="float16") = model_params[70]
            lv165 = R.call_tir(cls.rms_norm, (lv11_1, lv465), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv697: R.Tensor((15360, 640), dtype="uint32") = model_params[62]
            lv698: R.Tensor((15360, 160), dtype="float16") = model_params[63]
            lv93_1 = R.call_tir(cls.fused_fused_decode_NT_matmul5, (lv697, lv698, lv165), out_sinfo=R.Tensor((1, n, 15360), dtype="float16"))
            lv168 = R.call_tir(cls.split2, (lv93_1,), out_sinfo=[R.Tensor((1, n, 5120), dtype="float16"), R.Tensor((1, n, 5120), dtype="float16"), R.Tensor((1, n, 5120), dtype="float16")])
            lv169: R.Tensor((1, n, 5120), dtype="float16") = lv168[0]
            lv170 = R.call_tir(cls.reshape5, (lv169,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv171: R.Tensor((1, n, 5120), dtype="float16") = lv168[1]
            lv172 = R.call_tir(cls.reshape5, (lv171,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv173: R.Tensor((1, n, 5120), dtype="float16") = lv168[2]
            lv174 = R.call_tir(cls.reshape5, (lv173,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv175: R.Object = R.call_pure_packed("vm.builtin.paged_attention_kv_cache_append", lv148, cls.kv_cache_transpose_append, lv172, lv174, R.prim_value(6), sinfo_args=(R.Object,))
            lv176 = R.call_dps_packed("vm.builtin.paged_attention_kv_cache_attention", (lv175, cls.attention, lv170, R.prim_value(6), R.prim_value(1), R.prim_value(T.float32(1)), R.prim_value(10000)), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv177 = R.call_tir(cls.reshape6, (lv176,), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv700: R.Tensor((5120, 640), dtype="uint32") = model_params[64]
            lv701: R.Tensor((5120, 160), dtype="float16") = model_params[65]
            lv12_1 = R.call_tir(cls.fused_fused_decode1_fused_NT_matmul6_add1, (lv700, lv701, lv177, lv11_1), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv470: R.Tensor((5120,), dtype="float16") = model_params[71]
            lv181 = R.call_tir(cls.rms_norm, (lv12_1, lv470), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv704: R.Tensor((27648, 640), dtype="uint32") = model_params[66]
            lv705: R.Tensor((27648, 160), dtype="float16") = model_params[67]
            lv94_1 = R.call_tir(cls.fused_fused_decode2_NT_matmul7, (lv704, lv705, lv181), out_sinfo=R.Tensor((1, n, 27648), dtype="float16"))
            lv707 = R.call_tir(cls.fused_split3_silu1_multiply1, (lv94_1,), out_sinfo=R.Tensor((1, n, 13824), dtype="float16"))
            lv708: R.Tensor((5120, 1728), dtype="uint32") = model_params[68]
            lv709: R.Tensor((5120, 432), dtype="float16") = model_params[69]
            lv13_1 = R.call_tir(cls.fused_fused_decode3_fused_NT_matmul8_add1, (lv708, lv709, lv707, lv12_1), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv475: R.Tensor((5120,), dtype="float16") = model_params[80]
            lv192 = R.call_tir(cls.rms_norm, (lv13_1, lv475), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv712: R.Tensor((15360, 640), dtype="uint32") = model_params[72]
            lv713: R.Tensor((15360, 160), dtype="float16") = model_params[73]
            lv95_1 = R.call_tir(cls.fused_fused_decode_NT_matmul5, (lv712, lv713, lv192), out_sinfo=R.Tensor((1, n, 15360), dtype="float16"))
            lv195 = R.call_tir(cls.split2, (lv95_1,), out_sinfo=[R.Tensor((1, n, 5120), dtype="float16"), R.Tensor((1, n, 5120), dtype="float16"), R.Tensor((1, n, 5120), dtype="float16")])
            lv196: R.Tensor((1, n, 5120), dtype="float16") = lv195[0]
            lv197 = R.call_tir(cls.reshape5, (lv196,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv198: R.Tensor((1, n, 5120), dtype="float16") = lv195[1]
            lv199 = R.call_tir(cls.reshape5, (lv198,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv200: R.Tensor((1, n, 5120), dtype="float16") = lv195[2]
            lv201 = R.call_tir(cls.reshape5, (lv200,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv202: R.Object = R.call_pure_packed("vm.builtin.paged_attention_kv_cache_append", lv175, cls.kv_cache_transpose_append, lv199, lv201, R.prim_value(7), sinfo_args=(R.Object,))
            lv203 = R.call_dps_packed("vm.builtin.paged_attention_kv_cache_attention", (lv202, cls.attention, lv197, R.prim_value(7), R.prim_value(1), R.prim_value(T.float32(1)), R.prim_value(10000)), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv204 = R.call_tir(cls.reshape6, (lv203,), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv715: R.Tensor((5120, 640), dtype="uint32") = model_params[74]
            lv716: R.Tensor((5120, 160), dtype="float16") = model_params[75]
            lv14_1 = R.call_tir(cls.fused_fused_decode1_fused_NT_matmul6_add1, (lv715, lv716, lv204, lv13_1), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv480: R.Tensor((5120,), dtype="float16") = model_params[81]
            lv208 = R.call_tir(cls.rms_norm, (lv14_1, lv480), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv719: R.Tensor((27648, 640), dtype="uint32") = model_params[76]
            lv720: R.Tensor((27648, 160), dtype="float16") = model_params[77]
            lv96_1 = R.call_tir(cls.fused_fused_decode2_NT_matmul7, (lv719, lv720, lv208), out_sinfo=R.Tensor((1, n, 27648), dtype="float16"))
            lv722 = R.call_tir(cls.fused_split3_silu1_multiply1, (lv96_1,), out_sinfo=R.Tensor((1, n, 13824), dtype="float16"))
            lv723: R.Tensor((5120, 1728), dtype="uint32") = model_params[78]
            lv724: R.Tensor((5120, 432), dtype="float16") = model_params[79]
            lv15_1 = R.call_tir(cls.fused_fused_decode3_fused_NT_matmul8_add1, (lv723, lv724, lv722, lv14_1), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv485: R.Tensor((5120,), dtype="float16") = model_params[90]
            lv219 = R.call_tir(cls.rms_norm, (lv15_1, lv485), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv727: R.Tensor((15360, 640), dtype="uint32") = model_params[82]
            lv728: R.Tensor((15360, 160), dtype="float16") = model_params[83]
            lv97 = R.call_tir(cls.fused_fused_decode_NT_matmul5, (lv727, lv728, lv219), out_sinfo=R.Tensor((1, n, 15360), dtype="float16"))
            lv222 = R.call_tir(cls.split2, (lv97,), out_sinfo=[R.Tensor((1, n, 5120), dtype="float16"), R.Tensor((1, n, 5120), dtype="float16"), R.Tensor((1, n, 5120), dtype="float16")])
            lv223: R.Tensor((1, n, 5120), dtype="float16") = lv222[0]
            lv224 = R.call_tir(cls.reshape5, (lv223,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv225: R.Tensor((1, n, 5120), dtype="float16") = lv222[1]
            lv226 = R.call_tir(cls.reshape5, (lv225,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv227: R.Tensor((1, n, 5120), dtype="float16") = lv222[2]
            lv228 = R.call_tir(cls.reshape5, (lv227,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv229: R.Object = R.call_pure_packed("vm.builtin.paged_attention_kv_cache_append", lv202, cls.kv_cache_transpose_append, lv226, lv228, R.prim_value(8), sinfo_args=(R.Object,))
            lv230 = R.call_dps_packed("vm.builtin.paged_attention_kv_cache_attention", (lv229, cls.attention, lv224, R.prim_value(8), R.prim_value(1), R.prim_value(T.float32(1)), R.prim_value(10000)), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv231 = R.call_tir(cls.reshape6, (lv230,), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv730: R.Tensor((5120, 640), dtype="uint32") = model_params[84]
            lv731: R.Tensor((5120, 160), dtype="float16") = model_params[85]
            lv16 = R.call_tir(cls.fused_fused_decode1_fused_NT_matmul6_add1, (lv730, lv731, lv231, lv15_1), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv490: R.Tensor((5120,), dtype="float16") = model_params[91]
            lv235 = R.call_tir(cls.rms_norm, (lv16, lv490), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv734: R.Tensor((27648, 640), dtype="uint32") = model_params[86]
            lv735: R.Tensor((27648, 160), dtype="float16") = model_params[87]
            lv98 = R.call_tir(cls.fused_fused_decode2_NT_matmul7, (lv734, lv735, lv235), out_sinfo=R.Tensor((1, n, 27648), dtype="float16"))
            lv737 = R.call_tir(cls.fused_split3_silu1_multiply1, (lv98,), out_sinfo=R.Tensor((1, n, 13824), dtype="float16"))
            lv738: R.Tensor((5120, 1728), dtype="uint32") = model_params[88]
            lv739: R.Tensor((5120, 432), dtype="float16") = model_params[89]
            lv17 = R.call_tir(cls.fused_fused_decode3_fused_NT_matmul8_add1, (lv738, lv739, lv737, lv16), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv495: R.Tensor((5120,), dtype="float16") = model_params[100]
            lv246 = R.call_tir(cls.rms_norm, (lv17, lv495), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv742: R.Tensor((15360, 640), dtype="uint32") = model_params[92]
            lv743: R.Tensor((15360, 160), dtype="float16") = model_params[93]
            lv99 = R.call_tir(cls.fused_fused_decode_NT_matmul5, (lv742, lv743, lv246), out_sinfo=R.Tensor((1, n, 15360), dtype="float16"))
            lv249 = R.call_tir(cls.split2, (lv99,), out_sinfo=[R.Tensor((1, n, 5120), dtype="float16"), R.Tensor((1, n, 5120), dtype="float16"), R.Tensor((1, n, 5120), dtype="float16")])
            lv250: R.Tensor((1, n, 5120), dtype="float16") = lv249[0]
            lv251 = R.call_tir(cls.reshape5, (lv250,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv252: R.Tensor((1, n, 5120), dtype="float16") = lv249[1]
            lv253 = R.call_tir(cls.reshape5, (lv252,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv254: R.Tensor((1, n, 5120), dtype="float16") = lv249[2]
            lv255 = R.call_tir(cls.reshape5, (lv254,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv256: R.Object = R.call_pure_packed("vm.builtin.paged_attention_kv_cache_append", lv229, cls.kv_cache_transpose_append, lv253, lv255, R.prim_value(9), sinfo_args=(R.Object,))
            lv257 = R.call_dps_packed("vm.builtin.paged_attention_kv_cache_attention", (lv256, cls.attention, lv251, R.prim_value(9), R.prim_value(1), R.prim_value(T.float32(1)), R.prim_value(10000)), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv258 = R.call_tir(cls.reshape6, (lv257,), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv745: R.Tensor((5120, 640), dtype="uint32") = model_params[94]
            lv746: R.Tensor((5120, 160), dtype="float16") = model_params[95]
            lv18 = R.call_tir(cls.fused_fused_decode1_fused_NT_matmul6_add1, (lv745, lv746, lv258, lv17), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv500: R.Tensor((5120,), dtype="float16") = model_params[101]
            lv262 = R.call_tir(cls.rms_norm, (lv18, lv500), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv749: R.Tensor((27648, 640), dtype="uint32") = model_params[96]
            lv750: R.Tensor((27648, 160), dtype="float16") = model_params[97]
            lv100_1 = R.call_tir(cls.fused_fused_decode2_NT_matmul7, (lv749, lv750, lv262), out_sinfo=R.Tensor((1, n, 27648), dtype="float16"))
            lv752 = R.call_tir(cls.fused_split3_silu1_multiply1, (lv100_1,), out_sinfo=R.Tensor((1, n, 13824), dtype="float16"))
            lv753: R.Tensor((5120, 1728), dtype="uint32") = model_params[98]
            lv754: R.Tensor((5120, 432), dtype="float16") = model_params[99]
            lv19_1 = R.call_tir(cls.fused_fused_decode3_fused_NT_matmul8_add1, (lv753, lv754, lv752, lv18), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv505: R.Tensor((5120,), dtype="float16") = model_params[110]
            lv273 = R.call_tir(cls.rms_norm, (lv19_1, lv505), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv757: R.Tensor((15360, 640), dtype="uint32") = model_params[102]
            lv758: R.Tensor((15360, 160), dtype="float16") = model_params[103]
            lv101 = R.call_tir(cls.fused_fused_decode_NT_matmul5, (lv757, lv758, lv273), out_sinfo=R.Tensor((1, n, 15360), dtype="float16"))
            lv276 = R.call_tir(cls.split2, (lv101,), out_sinfo=[R.Tensor((1, n, 5120), dtype="float16"), R.Tensor((1, n, 5120), dtype="float16"), R.Tensor((1, n, 5120), dtype="float16")])
            lv277: R.Tensor((1, n, 5120), dtype="float16") = lv276[0]
            lv278 = R.call_tir(cls.reshape5, (lv277,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv279: R.Tensor((1, n, 5120), dtype="float16") = lv276[1]
            lv280 = R.call_tir(cls.reshape5, (lv279,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv281: R.Tensor((1, n, 5120), dtype="float16") = lv276[2]
            lv282 = R.call_tir(cls.reshape5, (lv281,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv283: R.Object = R.call_pure_packed("vm.builtin.paged_attention_kv_cache_append", lv256, cls.kv_cache_transpose_append, lv280, lv282, R.prim_value(10), sinfo_args=(R.Object,))
            lv284 = R.call_dps_packed("vm.builtin.paged_attention_kv_cache_attention", (lv283, cls.attention, lv278, R.prim_value(10), R.prim_value(1), R.prim_value(T.float32(1)), R.prim_value(10000)), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv285 = R.call_tir(cls.reshape6, (lv284,), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv760: R.Tensor((5120, 640), dtype="uint32") = model_params[104]
            lv761: R.Tensor((5120, 160), dtype="float16") = model_params[105]
            lv20 = R.call_tir(cls.fused_fused_decode1_fused_NT_matmul6_add1, (lv760, lv761, lv285, lv19_1), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv510: R.Tensor((5120,), dtype="float16") = model_params[111]
            lv289 = R.call_tir(cls.rms_norm, (lv20, lv510), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv764: R.Tensor((27648, 640), dtype="uint32") = model_params[106]
            lv765: R.Tensor((27648, 160), dtype="float16") = model_params[107]
            lv102 = R.call_tir(cls.fused_fused_decode2_NT_matmul7, (lv764, lv765, lv289), out_sinfo=R.Tensor((1, n, 27648), dtype="float16"))
            lv767 = R.call_tir(cls.fused_split3_silu1_multiply1, (lv102,), out_sinfo=R.Tensor((1, n, 13824), dtype="float16"))
            lv768: R.Tensor((5120, 1728), dtype="uint32") = model_params[108]
            lv769: R.Tensor((5120, 432), dtype="float16") = model_params[109]
            lv21 = R.call_tir(cls.fused_fused_decode3_fused_NT_matmul8_add1, (lv768, lv769, lv767, lv20), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv515: R.Tensor((5120,), dtype="float16") = model_params[120]
            lv300 = R.call_tir(cls.rms_norm, (lv21, lv515), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv772: R.Tensor((15360, 640), dtype="uint32") = model_params[112]
            lv773: R.Tensor((15360, 160), dtype="float16") = model_params[113]
            lv103 = R.call_tir(cls.fused_fused_decode_NT_matmul5, (lv772, lv773, lv300), out_sinfo=R.Tensor((1, n, 15360), dtype="float16"))
            lv303 = R.call_tir(cls.split2, (lv103,), out_sinfo=[R.Tensor((1, n, 5120), dtype="float16"), R.Tensor((1, n, 5120), dtype="float16"), R.Tensor((1, n, 5120), dtype="float16")])
            lv304: R.Tensor((1, n, 5120), dtype="float16") = lv303[0]
            lv305 = R.call_tir(cls.reshape5, (lv304,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv306: R.Tensor((1, n, 5120), dtype="float16") = lv303[1]
            lv307 = R.call_tir(cls.reshape5, (lv306,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv308: R.Tensor((1, n, 5120), dtype="float16") = lv303[2]
            lv309 = R.call_tir(cls.reshape5, (lv308,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv310: R.Object = R.call_pure_packed("vm.builtin.paged_attention_kv_cache_append", lv283, cls.kv_cache_transpose_append, lv307, lv309, R.prim_value(11), sinfo_args=(R.Object,))
            lv311 = R.call_dps_packed("vm.builtin.paged_attention_kv_cache_attention", (lv310, cls.attention, lv305, R.prim_value(11), R.prim_value(1), R.prim_value(T.float32(1)), R.prim_value(10000)), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv312 = R.call_tir(cls.reshape6, (lv311,), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv775: R.Tensor((5120, 640), dtype="uint32") = model_params[114]
            lv776: R.Tensor((5120, 160), dtype="float16") = model_params[115]
            lv22 = R.call_tir(cls.fused_fused_decode1_fused_NT_matmul6_add1, (lv775, lv776, lv312, lv21), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv520: R.Tensor((5120,), dtype="float16") = model_params[121]
            lv316 = R.call_tir(cls.rms_norm, (lv22, lv520), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv779: R.Tensor((27648, 640), dtype="uint32") = model_params[116]
            lv780: R.Tensor((27648, 160), dtype="float16") = model_params[117]
            lv104 = R.call_tir(cls.fused_fused_decode2_NT_matmul7, (lv779, lv780, lv316), out_sinfo=R.Tensor((1, n, 27648), dtype="float16"))
            lv782 = R.call_tir(cls.fused_split3_silu1_multiply1, (lv104,), out_sinfo=R.Tensor((1, n, 13824), dtype="float16"))
            lv783: R.Tensor((5120, 1728), dtype="uint32") = model_params[118]
            lv784: R.Tensor((5120, 432), dtype="float16") = model_params[119]
            lv23 = R.call_tir(cls.fused_fused_decode3_fused_NT_matmul8_add1, (lv783, lv784, lv782, lv22), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv525: R.Tensor((5120,), dtype="float16") = model_params[130]
            lv327 = R.call_tir(cls.rms_norm, (lv23, lv525), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv787: R.Tensor((15360, 640), dtype="uint32") = model_params[122]
            lv788: R.Tensor((15360, 160), dtype="float16") = model_params[123]
            lv105 = R.call_tir(cls.fused_fused_decode_NT_matmul5, (lv787, lv788, lv327), out_sinfo=R.Tensor((1, n, 15360), dtype="float16"))
            lv330 = R.call_tir(cls.split2, (lv105,), out_sinfo=[R.Tensor((1, n, 5120), dtype="float16"), R.Tensor((1, n, 5120), dtype="float16"), R.Tensor((1, n, 5120), dtype="float16")])
            lv331: R.Tensor((1, n, 5120), dtype="float16") = lv330[0]
            lv332 = R.call_tir(cls.reshape5, (lv331,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv333: R.Tensor((1, n, 5120), dtype="float16") = lv330[1]
            lv334 = R.call_tir(cls.reshape5, (lv333,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv335: R.Tensor((1, n, 5120), dtype="float16") = lv330[2]
            lv336 = R.call_tir(cls.reshape5, (lv335,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv337: R.Object = R.call_pure_packed("vm.builtin.paged_attention_kv_cache_append", lv310, cls.kv_cache_transpose_append, lv334, lv336, R.prim_value(12), sinfo_args=(R.Object,))
            lv338 = R.call_dps_packed("vm.builtin.paged_attention_kv_cache_attention", (lv337, cls.attention, lv332, R.prim_value(12), R.prim_value(1), R.prim_value(T.float32(1)), R.prim_value(10000)), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv339 = R.call_tir(cls.reshape6, (lv338,), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv790: R.Tensor((5120, 640), dtype="uint32") = model_params[124]
            lv791: R.Tensor((5120, 160), dtype="float16") = model_params[125]
            lv24 = R.call_tir(cls.fused_fused_decode1_fused_NT_matmul6_add1, (lv790, lv791, lv339, lv23), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv530: R.Tensor((5120,), dtype="float16") = model_params[131]
            lv343 = R.call_tir(cls.rms_norm, (lv24, lv530), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv794: R.Tensor((27648, 640), dtype="uint32") = model_params[126]
            lv795: R.Tensor((27648, 160), dtype="float16") = model_params[127]
            lv106 = R.call_tir(cls.fused_fused_decode2_NT_matmul7, (lv794, lv795, lv343), out_sinfo=R.Tensor((1, n, 27648), dtype="float16"))
            lv797 = R.call_tir(cls.fused_split3_silu1_multiply1, (lv106,), out_sinfo=R.Tensor((1, n, 13824), dtype="float16"))
            lv798: R.Tensor((5120, 1728), dtype="uint32") = model_params[128]
            lv799: R.Tensor((5120, 432), dtype="float16") = model_params[129]
            lv25 = R.call_tir(cls.fused_fused_decode3_fused_NT_matmul8_add1, (lv798, lv799, lv797, lv24), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv535: R.Tensor((5120,), dtype="float16") = model_params[140]
            lv354 = R.call_tir(cls.rms_norm, (lv25, lv535), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv802: R.Tensor((15360, 640), dtype="uint32") = model_params[132]
            lv803: R.Tensor((15360, 160), dtype="float16") = model_params[133]
            lv107 = R.call_tir(cls.fused_fused_decode_NT_matmul5, (lv802, lv803, lv354), out_sinfo=R.Tensor((1, n, 15360), dtype="float16"))
            lv357 = R.call_tir(cls.split2, (lv107,), out_sinfo=[R.Tensor((1, n, 5120), dtype="float16"), R.Tensor((1, n, 5120), dtype="float16"), R.Tensor((1, n, 5120), dtype="float16")])
            lv358: R.Tensor((1, n, 5120), dtype="float16") = lv357[0]
            lv359 = R.call_tir(cls.reshape5, (lv358,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv360: R.Tensor((1, n, 5120), dtype="float16") = lv357[1]
            lv361 = R.call_tir(cls.reshape5, (lv360,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv362: R.Tensor((1, n, 5120), dtype="float16") = lv357[2]
            lv363 = R.call_tir(cls.reshape5, (lv362,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv364: R.Object = R.call_pure_packed("vm.builtin.paged_attention_kv_cache_append", lv337, cls.kv_cache_transpose_append, lv361, lv363, R.prim_value(13), sinfo_args=(R.Object,))
            lv365 = R.call_dps_packed("vm.builtin.paged_attention_kv_cache_attention", (lv364, cls.attention, lv359, R.prim_value(13), R.prim_value(1), R.prim_value(T.float32(1)), R.prim_value(10000)), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv366 = R.call_tir(cls.reshape6, (lv365,), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv805: R.Tensor((5120, 640), dtype="uint32") = model_params[134]
            lv806: R.Tensor((5120, 160), dtype="float16") = model_params[135]
            lv26 = R.call_tir(cls.fused_fused_decode1_fused_NT_matmul6_add1, (lv805, lv806, lv366, lv25), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv540: R.Tensor((5120,), dtype="float16") = model_params[141]
            lv370 = R.call_tir(cls.rms_norm, (lv26, lv540), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv809: R.Tensor((27648, 640), dtype="uint32") = model_params[136]
            lv810: R.Tensor((27648, 160), dtype="float16") = model_params[137]
            lv108 = R.call_tir(cls.fused_fused_decode2_NT_matmul7, (lv809, lv810, lv370), out_sinfo=R.Tensor((1, n, 27648), dtype="float16"))
            lv812 = R.call_tir(cls.fused_split3_silu1_multiply1, (lv108,), out_sinfo=R.Tensor((1, n, 13824), dtype="float16"))
            lv813: R.Tensor((5120, 1728), dtype="uint32") = model_params[138]
            lv814: R.Tensor((5120, 432), dtype="float16") = model_params[139]
            lv27 = R.call_tir(cls.fused_fused_decode3_fused_NT_matmul8_add1, (lv813, lv814, lv812, lv26), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv545: R.Tensor((5120,), dtype="float16") = model_params[150]
            lv381 = R.call_tir(cls.rms_norm, (lv27, lv545), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv817: R.Tensor((15360, 640), dtype="uint32") = model_params[142]
            lv818: R.Tensor((15360, 160), dtype="float16") = model_params[143]
            lv109 = R.call_tir(cls.fused_fused_decode_NT_matmul5, (lv817, lv818, lv381), out_sinfo=R.Tensor((1, n, 15360), dtype="float16"))
            lv384 = R.call_tir(cls.split2, (lv109,), out_sinfo=[R.Tensor((1, n, 5120), dtype="float16"), R.Tensor((1, n, 5120), dtype="float16"), R.Tensor((1, n, 5120), dtype="float16")])
            lv385: R.Tensor((1, n, 5120), dtype="float16") = lv384[0]
            lv386 = R.call_tir(cls.reshape5, (lv385,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv387: R.Tensor((1, n, 5120), dtype="float16") = lv384[1]
            lv388 = R.call_tir(cls.reshape5, (lv387,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv389: R.Tensor((1, n, 5120), dtype="float16") = lv384[2]
            lv390 = R.call_tir(cls.reshape5, (lv389,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv391: R.Object = R.call_pure_packed("vm.builtin.paged_attention_kv_cache_append", lv364, cls.kv_cache_transpose_append, lv388, lv390, R.prim_value(14), sinfo_args=(R.Object,))
            lv392 = R.call_dps_packed("vm.builtin.paged_attention_kv_cache_attention", (lv391, cls.attention, lv386, R.prim_value(14), R.prim_value(1), R.prim_value(T.float32(1)), R.prim_value(10000)), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv393 = R.call_tir(cls.reshape6, (lv392,), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv820: R.Tensor((5120, 640), dtype="uint32") = model_params[144]
            lv821: R.Tensor((5120, 160), dtype="float16") = model_params[145]
            lv28 = R.call_tir(cls.fused_fused_decode1_fused_NT_matmul6_add1, (lv820, lv821, lv393, lv27), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv550: R.Tensor((5120,), dtype="float16") = model_params[151]
            lv397 = R.call_tir(cls.rms_norm, (lv28, lv550), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv824: R.Tensor((27648, 640), dtype="uint32") = model_params[146]
            lv825: R.Tensor((27648, 160), dtype="float16") = model_params[147]
            lv110 = R.call_tir(cls.fused_fused_decode2_NT_matmul7, (lv824, lv825, lv397), out_sinfo=R.Tensor((1, n, 27648), dtype="float16"))
            lv827 = R.call_tir(cls.fused_split3_silu1_multiply1, (lv110,), out_sinfo=R.Tensor((1, n, 13824), dtype="float16"))
            lv828: R.Tensor((5120, 1728), dtype="uint32") = model_params[148]
            lv829: R.Tensor((5120, 432), dtype="float16") = model_params[149]
            lv29 = R.call_tir(cls.fused_fused_decode3_fused_NT_matmul8_add1, (lv828, lv829, lv827, lv28), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv555: R.Tensor((5120,), dtype="float16") = model_params[160]
            lv408 = R.call_tir(cls.rms_norm, (lv29, lv555), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv832: R.Tensor((15360, 640), dtype="uint32") = model_params[152]
            lv833: R.Tensor((15360, 160), dtype="float16") = model_params[153]
            lv111_1 = R.call_tir(cls.fused_fused_decode_NT_matmul5, (lv832, lv833, lv408), out_sinfo=R.Tensor((1, n, 15360), dtype="float16"))
            lv411 = R.call_tir(cls.split2, (lv111_1,), out_sinfo=[R.Tensor((1, n, 5120), dtype="float16"), R.Tensor((1, n, 5120), dtype="float16"), R.Tensor((1, n, 5120), dtype="float16")])
            lv412: R.Tensor((1, n, 5120), dtype="float16") = lv411[0]
            lv413 = R.call_tir(cls.reshape5, (lv412,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv414: R.Tensor((1, n, 5120), dtype="float16") = lv411[1]
            lv415_1 = R.call_tir(cls.reshape5, (lv414,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv416: R.Tensor((1, n, 5120), dtype="float16") = lv411[2]
            lv417 = R.call_tir(cls.reshape5, (lv416,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv418: R.Object = R.call_pure_packed("vm.builtin.paged_attention_kv_cache_append", lv391, cls.kv_cache_transpose_append, lv415_1, lv417, R.prim_value(15), sinfo_args=(R.Object,))
            lv419 = R.call_dps_packed("vm.builtin.paged_attention_kv_cache_attention", (lv418, cls.attention, lv413, R.prim_value(15), R.prim_value(1), R.prim_value(T.float32(1)), R.prim_value(10000)), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv420_1 = R.call_tir(cls.reshape6, (lv419,), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv835: R.Tensor((5120, 640), dtype="uint32") = model_params[154]
            lv836: R.Tensor((5120, 160), dtype="float16") = model_params[155]
            lv30_1 = R.call_tir(cls.fused_fused_decode1_fused_NT_matmul6_add1, (lv835, lv836, lv420_1, lv29), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv560: R.Tensor((5120,), dtype="float16") = model_params[161]
            lv424 = R.call_tir(cls.rms_norm, (lv30_1, lv560), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv839: R.Tensor((27648, 640), dtype="uint32") = model_params[156]
            lv840: R.Tensor((27648, 160), dtype="float16") = model_params[157]
            lv112 = R.call_tir(cls.fused_fused_decode2_NT_matmul7, (lv839, lv840, lv424), out_sinfo=R.Tensor((1, n, 27648), dtype="float16"))
            lv842 = R.call_tir(cls.fused_split3_silu1_multiply1, (lv112,), out_sinfo=R.Tensor((1, n, 13824), dtype="float16"))
            lv843: R.Tensor((5120, 1728), dtype="uint32") = model_params[158]
            lv844: R.Tensor((5120, 432), dtype="float16") = model_params[159]
            lv31 = R.call_tir(cls.fused_fused_decode3_fused_NT_matmul8_add1, (lv843, lv844, lv842, lv30_1), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv565: R.Tensor((5120,), dtype="float16") = model_params[170]
            lv435_1 = R.call_tir(cls.rms_norm, (lv31, lv565), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv847: R.Tensor((15360, 640), dtype="uint32") = model_params[162]
            lv848: R.Tensor((15360, 160), dtype="float16") = model_params[163]
            lv113 = R.call_tir(cls.fused_fused_decode_NT_matmul5, (lv847, lv848, lv435_1), out_sinfo=R.Tensor((1, n, 15360), dtype="float16"))
            lv438 = R.call_tir(cls.split2, (lv113,), out_sinfo=[R.Tensor((1, n, 5120), dtype="float16"), R.Tensor((1, n, 5120), dtype="float16"), R.Tensor((1, n, 5120), dtype="float16")])
            lv439: R.Tensor((1, n, 5120), dtype="float16") = lv438[0]
            lv440_1 = R.call_tir(cls.reshape5, (lv439,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv441: R.Tensor((1, n, 5120), dtype="float16") = lv438[1]
            lv442 = R.call_tir(cls.reshape5, (lv441,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv443: R.Tensor((1, n, 5120), dtype="float16") = lv438[2]
            lv444 = R.call_tir(cls.reshape5, (lv443,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv445_1: R.Object = R.call_pure_packed("vm.builtin.paged_attention_kv_cache_append", lv418, cls.kv_cache_transpose_append, lv442, lv444, R.prim_value(16), sinfo_args=(R.Object,))
            lv446 = R.call_dps_packed("vm.builtin.paged_attention_kv_cache_attention", (lv445_1, cls.attention, lv440_1, R.prim_value(16), R.prim_value(1), R.prim_value(T.float32(1)), R.prim_value(10000)), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv447 = R.call_tir(cls.reshape6, (lv446,), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv850: R.Tensor((5120, 640), dtype="uint32") = model_params[164]
            lv851: R.Tensor((5120, 160), dtype="float16") = model_params[165]
            lv32 = R.call_tir(cls.fused_fused_decode1_fused_NT_matmul6_add1, (lv850, lv851, lv447, lv31), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv570: R.Tensor((5120,), dtype="float16") = model_params[171]
            lv451 = R.call_tir(cls.rms_norm, (lv32, lv570), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv854: R.Tensor((27648, 640), dtype="uint32") = model_params[166]
            lv855: R.Tensor((27648, 160), dtype="float16") = model_params[167]
            lv114_1 = R.call_tir(cls.fused_fused_decode2_NT_matmul7, (lv854, lv855, lv451), out_sinfo=R.Tensor((1, n, 27648), dtype="float16"))
            lv857 = R.call_tir(cls.fused_split3_silu1_multiply1, (lv114_1,), out_sinfo=R.Tensor((1, n, 13824), dtype="float16"))
            lv858: R.Tensor((5120, 1728), dtype="uint32") = model_params[168]
            lv859: R.Tensor((5120, 432), dtype="float16") = model_params[169]
            lv33_1 = R.call_tir(cls.fused_fused_decode3_fused_NT_matmul8_add1, (lv858, lv859, lv857, lv32), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv575: R.Tensor((5120,), dtype="float16") = model_params[180]
            lv462 = R.call_tir(cls.rms_norm, (lv33_1, lv575), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv862: R.Tensor((15360, 640), dtype="uint32") = model_params[172]
            lv863: R.Tensor((15360, 160), dtype="float16") = model_params[173]
            lv115_1 = R.call_tir(cls.fused_fused_decode_NT_matmul5, (lv862, lv863, lv462), out_sinfo=R.Tensor((1, n, 15360), dtype="float16"))
            lv465_1 = R.call_tir(cls.split2, (lv115_1,), out_sinfo=[R.Tensor((1, n, 5120), dtype="float16"), R.Tensor((1, n, 5120), dtype="float16"), R.Tensor((1, n, 5120), dtype="float16")])
            lv466: R.Tensor((1, n, 5120), dtype="float16") = lv465_1[0]
            lv467 = R.call_tir(cls.reshape5, (lv466,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv468: R.Tensor((1, n, 5120), dtype="float16") = lv465_1[1]
            lv469 = R.call_tir(cls.reshape5, (lv468,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv470_1: R.Tensor((1, n, 5120), dtype="float16") = lv465_1[2]
            lv471 = R.call_tir(cls.reshape5, (lv470_1,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv472: R.Object = R.call_pure_packed("vm.builtin.paged_attention_kv_cache_append", lv445_1, cls.kv_cache_transpose_append, lv469, lv471, R.prim_value(17), sinfo_args=(R.Object,))
            lv473 = R.call_dps_packed("vm.builtin.paged_attention_kv_cache_attention", (lv472, cls.attention, lv467, R.prim_value(17), R.prim_value(1), R.prim_value(T.float32(1)), R.prim_value(10000)), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv474 = R.call_tir(cls.reshape6, (lv473,), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv865: R.Tensor((5120, 640), dtype="uint32") = model_params[174]
            lv866: R.Tensor((5120, 160), dtype="float16") = model_params[175]
            lv34_1 = R.call_tir(cls.fused_fused_decode1_fused_NT_matmul6_add1, (lv865, lv866, lv474, lv33_1), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv580: R.Tensor((5120,), dtype="float16") = model_params[181]
            lv478 = R.call_tir(cls.rms_norm, (lv34_1, lv580), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv869: R.Tensor((27648, 640), dtype="uint32") = model_params[176]
            lv870: R.Tensor((27648, 160), dtype="float16") = model_params[177]
            lv116_1 = R.call_tir(cls.fused_fused_decode2_NT_matmul7, (lv869, lv870, lv478), out_sinfo=R.Tensor((1, n, 27648), dtype="float16"))
            lv872 = R.call_tir(cls.fused_split3_silu1_multiply1, (lv116_1,), out_sinfo=R.Tensor((1, n, 13824), dtype="float16"))
            lv873: R.Tensor((5120, 1728), dtype="uint32") = model_params[178]
            lv874: R.Tensor((5120, 432), dtype="float16") = model_params[179]
            lv35_1 = R.call_tir(cls.fused_fused_decode3_fused_NT_matmul8_add1, (lv873, lv874, lv872, lv34_1), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv585: R.Tensor((5120,), dtype="float16") = model_params[190]
            lv489 = R.call_tir(cls.rms_norm, (lv35_1, lv585), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv877: R.Tensor((15360, 640), dtype="uint32") = model_params[182]
            lv878: R.Tensor((15360, 160), dtype="float16") = model_params[183]
            lv117_1 = R.call_tir(cls.fused_fused_decode_NT_matmul5, (lv877, lv878, lv489), out_sinfo=R.Tensor((1, n, 15360), dtype="float16"))
            lv492 = R.call_tir(cls.split2, (lv117_1,), out_sinfo=[R.Tensor((1, n, 5120), dtype="float16"), R.Tensor((1, n, 5120), dtype="float16"), R.Tensor((1, n, 5120), dtype="float16")])
            lv493: R.Tensor((1, n, 5120), dtype="float16") = lv492[0]
            lv494 = R.call_tir(cls.reshape5, (lv493,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv495_1: R.Tensor((1, n, 5120), dtype="float16") = lv492[1]
            lv496 = R.call_tir(cls.reshape5, (lv495_1,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv497: R.Tensor((1, n, 5120), dtype="float16") = lv492[2]
            lv498 = R.call_tir(cls.reshape5, (lv497,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv499: R.Object = R.call_pure_packed("vm.builtin.paged_attention_kv_cache_append", lv472, cls.kv_cache_transpose_append, lv496, lv498, R.prim_value(18), sinfo_args=(R.Object,))
            lv500_1 = R.call_dps_packed("vm.builtin.paged_attention_kv_cache_attention", (lv499, cls.attention, lv494, R.prim_value(18), R.prim_value(1), R.prim_value(T.float32(1)), R.prim_value(10000)), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv501 = R.call_tir(cls.reshape6, (lv500_1,), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv880: R.Tensor((5120, 640), dtype="uint32") = model_params[184]
            lv881: R.Tensor((5120, 160), dtype="float16") = model_params[185]
            lv36_1 = R.call_tir(cls.fused_fused_decode1_fused_NT_matmul6_add1, (lv880, lv881, lv501, lv35_1), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv590: R.Tensor((5120,), dtype="float16") = model_params[191]
            lv505_1 = R.call_tir(cls.rms_norm, (lv36_1, lv590), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv884: R.Tensor((27648, 640), dtype="uint32") = model_params[186]
            lv885: R.Tensor((27648, 160), dtype="float16") = model_params[187]
            lv118_1 = R.call_tir(cls.fused_fused_decode2_NT_matmul7, (lv884, lv885, lv505_1), out_sinfo=R.Tensor((1, n, 27648), dtype="float16"))
            lv887 = R.call_tir(cls.fused_split3_silu1_multiply1, (lv118_1,), out_sinfo=R.Tensor((1, n, 13824), dtype="float16"))
            lv888: R.Tensor((5120, 1728), dtype="uint32") = model_params[188]
            lv889: R.Tensor((5120, 432), dtype="float16") = model_params[189]
            lv37_1 = R.call_tir(cls.fused_fused_decode3_fused_NT_matmul8_add1, (lv888, lv889, lv887, lv36_1), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv595: R.Tensor((5120,), dtype="float16") = model_params[200]
            lv516 = R.call_tir(cls.rms_norm, (lv37_1, lv595), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv892: R.Tensor((15360, 640), dtype="uint32") = model_params[192]
            lv893: R.Tensor((15360, 160), dtype="float16") = model_params[193]
            lv119_1 = R.call_tir(cls.fused_fused_decode_NT_matmul5, (lv892, lv893, lv516), out_sinfo=R.Tensor((1, n, 15360), dtype="float16"))
            lv519 = R.call_tir(cls.split2, (lv119_1,), out_sinfo=[R.Tensor((1, n, 5120), dtype="float16"), R.Tensor((1, n, 5120), dtype="float16"), R.Tensor((1, n, 5120), dtype="float16")])
            lv520_1: R.Tensor((1, n, 5120), dtype="float16") = lv519[0]
            lv521 = R.call_tir(cls.reshape5, (lv520_1,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv522: R.Tensor((1, n, 5120), dtype="float16") = lv519[1]
            lv523 = R.call_tir(cls.reshape5, (lv522,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv524: R.Tensor((1, n, 5120), dtype="float16") = lv519[2]
            lv525_1 = R.call_tir(cls.reshape5, (lv524,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv526: R.Object = R.call_pure_packed("vm.builtin.paged_attention_kv_cache_append", lv499, cls.kv_cache_transpose_append, lv523, lv525_1, R.prim_value(19), sinfo_args=(R.Object,))
            lv527 = R.call_dps_packed("vm.builtin.paged_attention_kv_cache_attention", (lv526, cls.attention, lv521, R.prim_value(19), R.prim_value(1), R.prim_value(T.float32(1)), R.prim_value(10000)), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv528 = R.call_tir(cls.reshape6, (lv527,), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv895: R.Tensor((5120, 640), dtype="uint32") = model_params[194]
            lv896: R.Tensor((5120, 160), dtype="float16") = model_params[195]
            lv38_1 = R.call_tir(cls.fused_fused_decode1_fused_NT_matmul6_add1, (lv895, lv896, lv528, lv37_1), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv600: R.Tensor((5120,), dtype="float16") = model_params[201]
            lv532 = R.call_tir(cls.rms_norm, (lv38_1, lv600), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv899: R.Tensor((27648, 640), dtype="uint32") = model_params[196]
            lv900: R.Tensor((27648, 160), dtype="float16") = model_params[197]
            lv120_1 = R.call_tir(cls.fused_fused_decode2_NT_matmul7, (lv899, lv900, lv532), out_sinfo=R.Tensor((1, n, 27648), dtype="float16"))
            lv902 = R.call_tir(cls.fused_split3_silu1_multiply1, (lv120_1,), out_sinfo=R.Tensor((1, n, 13824), dtype="float16"))
            lv903: R.Tensor((5120, 1728), dtype="uint32") = model_params[198]
            lv904: R.Tensor((5120, 432), dtype="float16") = model_params[199]
            lv39_1 = R.call_tir(cls.fused_fused_decode3_fused_NT_matmul8_add1, (lv903, lv904, lv902, lv38_1), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv605: R.Tensor((5120,), dtype="float16") = model_params[210]
            lv543 = R.call_tir(cls.rms_norm, (lv39_1, lv605), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv907: R.Tensor((15360, 640), dtype="uint32") = model_params[202]
            lv908: R.Tensor((15360, 160), dtype="float16") = model_params[203]
            lv121_1 = R.call_tir(cls.fused_fused_decode_NT_matmul5, (lv907, lv908, lv543), out_sinfo=R.Tensor((1, n, 15360), dtype="float16"))
            lv546 = R.call_tir(cls.split2, (lv121_1,), out_sinfo=[R.Tensor((1, n, 5120), dtype="float16"), R.Tensor((1, n, 5120), dtype="float16"), R.Tensor((1, n, 5120), dtype="float16")])
            lv547: R.Tensor((1, n, 5120), dtype="float16") = lv546[0]
            lv548 = R.call_tir(cls.reshape5, (lv547,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv549: R.Tensor((1, n, 5120), dtype="float16") = lv546[1]
            lv550_1 = R.call_tir(cls.reshape5, (lv549,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv551: R.Tensor((1, n, 5120), dtype="float16") = lv546[2]
            lv552 = R.call_tir(cls.reshape5, (lv551,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv553: R.Object = R.call_pure_packed("vm.builtin.paged_attention_kv_cache_append", lv526, cls.kv_cache_transpose_append, lv550_1, lv552, R.prim_value(20), sinfo_args=(R.Object,))
            lv554 = R.call_dps_packed("vm.builtin.paged_attention_kv_cache_attention", (lv553, cls.attention, lv548, R.prim_value(20), R.prim_value(1), R.prim_value(T.float32(1)), R.prim_value(10000)), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv555_1 = R.call_tir(cls.reshape6, (lv554,), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv910: R.Tensor((5120, 640), dtype="uint32") = model_params[204]
            lv911: R.Tensor((5120, 160), dtype="float16") = model_params[205]
            lv40_1 = R.call_tir(cls.fused_fused_decode1_fused_NT_matmul6_add1, (lv910, lv911, lv555_1, lv39_1), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv610_1: R.Tensor((5120,), dtype="float16") = model_params[211]
            lv559 = R.call_tir(cls.rms_norm, (lv40_1, lv610_1), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv914: R.Tensor((27648, 640), dtype="uint32") = model_params[206]
            lv915: R.Tensor((27648, 160), dtype="float16") = model_params[207]
            lv122_1 = R.call_tir(cls.fused_fused_decode2_NT_matmul7, (lv914, lv915, lv559), out_sinfo=R.Tensor((1, n, 27648), dtype="float16"))
            lv917 = R.call_tir(cls.fused_split3_silu1_multiply1, (lv122_1,), out_sinfo=R.Tensor((1, n, 13824), dtype="float16"))
            lv918: R.Tensor((5120, 1728), dtype="uint32") = model_params[208]
            lv919: R.Tensor((5120, 432), dtype="float16") = model_params[209]
            lv41_1 = R.call_tir(cls.fused_fused_decode3_fused_NT_matmul8_add1, (lv918, lv919, lv917, lv40_1), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv615_1: R.Tensor((5120,), dtype="float16") = model_params[220]
            lv570_1 = R.call_tir(cls.rms_norm, (lv41_1, lv615_1), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv922: R.Tensor((15360, 640), dtype="uint32") = model_params[212]
            lv923: R.Tensor((15360, 160), dtype="float16") = model_params[213]
            lv123_1 = R.call_tir(cls.fused_fused_decode_NT_matmul5, (lv922, lv923, lv570_1), out_sinfo=R.Tensor((1, n, 15360), dtype="float16"))
            lv573 = R.call_tir(cls.split2, (lv123_1,), out_sinfo=[R.Tensor((1, n, 5120), dtype="float16"), R.Tensor((1, n, 5120), dtype="float16"), R.Tensor((1, n, 5120), dtype="float16")])
            lv574: R.Tensor((1, n, 5120), dtype="float16") = lv573[0]
            lv575_1 = R.call_tir(cls.reshape5, (lv574,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv576: R.Tensor((1, n, 5120), dtype="float16") = lv573[1]
            lv577 = R.call_tir(cls.reshape5, (lv576,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv578: R.Tensor((1, n, 5120), dtype="float16") = lv573[2]
            lv579 = R.call_tir(cls.reshape5, (lv578,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv580_1: R.Object = R.call_pure_packed("vm.builtin.paged_attention_kv_cache_append", lv553, cls.kv_cache_transpose_append, lv577, lv579, R.prim_value(21), sinfo_args=(R.Object,))
            lv581 = R.call_dps_packed("vm.builtin.paged_attention_kv_cache_attention", (lv580_1, cls.attention, lv575_1, R.prim_value(21), R.prim_value(1), R.prim_value(T.float32(1)), R.prim_value(10000)), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv582 = R.call_tir(cls.reshape6, (lv581,), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv925: R.Tensor((5120, 640), dtype="uint32") = model_params[214]
            lv926: R.Tensor((5120, 160), dtype="float16") = model_params[215]
            lv42_1 = R.call_tir(cls.fused_fused_decode1_fused_NT_matmul6_add1, (lv925, lv926, lv582, lv41_1), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv620: R.Tensor((5120,), dtype="float16") = model_params[221]
            lv586 = R.call_tir(cls.rms_norm, (lv42_1, lv620), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv929: R.Tensor((27648, 640), dtype="uint32") = model_params[216]
            lv930: R.Tensor((27648, 160), dtype="float16") = model_params[217]
            lv124 = R.call_tir(cls.fused_fused_decode2_NT_matmul7, (lv929, lv930, lv586), out_sinfo=R.Tensor((1, n, 27648), dtype="float16"))
            lv932 = R.call_tir(cls.fused_split3_silu1_multiply1, (lv124,), out_sinfo=R.Tensor((1, n, 13824), dtype="float16"))
            lv933: R.Tensor((5120, 1728), dtype="uint32") = model_params[218]
            lv934: R.Tensor((5120, 432), dtype="float16") = model_params[219]
            lv43 = R.call_tir(cls.fused_fused_decode3_fused_NT_matmul8_add1, (lv933, lv934, lv932, lv42_1), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv625_1: R.Tensor((5120,), dtype="float16") = model_params[230]
            lv597 = R.call_tir(cls.rms_norm, (lv43, lv625_1), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv937: R.Tensor((15360, 640), dtype="uint32") = model_params[222]
            lv938: R.Tensor((15360, 160), dtype="float16") = model_params[223]
            lv125 = R.call_tir(cls.fused_fused_decode_NT_matmul5, (lv937, lv938, lv597), out_sinfo=R.Tensor((1, n, 15360), dtype="float16"))
            lv600_1 = R.call_tir(cls.split2, (lv125,), out_sinfo=[R.Tensor((1, n, 5120), dtype="float16"), R.Tensor((1, n, 5120), dtype="float16"), R.Tensor((1, n, 5120), dtype="float16")])
            lv601: R.Tensor((1, n, 5120), dtype="float16") = lv600_1[0]
            lv602 = R.call_tir(cls.reshape5, (lv601,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv603: R.Tensor((1, n, 5120), dtype="float16") = lv600_1[1]
            lv604 = R.call_tir(cls.reshape5, (lv603,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv605_1: R.Tensor((1, n, 5120), dtype="float16") = lv600_1[2]
            lv606 = R.call_tir(cls.reshape5, (lv605_1,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv607_1: R.Object = R.call_pure_packed("vm.builtin.paged_attention_kv_cache_append", lv580_1, cls.kv_cache_transpose_append, lv604, lv606, R.prim_value(22), sinfo_args=(R.Object,))
            lv608_1 = R.call_dps_packed("vm.builtin.paged_attention_kv_cache_attention", (lv607_1, cls.attention, lv602, R.prim_value(22), R.prim_value(1), R.prim_value(T.float32(1)), R.prim_value(10000)), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv609 = R.call_tir(cls.reshape6, (lv608_1,), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv940: R.Tensor((5120, 640), dtype="uint32") = model_params[224]
            lv941: R.Tensor((5120, 160), dtype="float16") = model_params[225]
            lv44 = R.call_tir(cls.fused_fused_decode1_fused_NT_matmul6_add1, (lv940, lv941, lv609, lv43), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv630_1: R.Tensor((5120,), dtype="float16") = model_params[231]
            lv613 = R.call_tir(cls.rms_norm, (lv44, lv630_1), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv944: R.Tensor((27648, 640), dtype="uint32") = model_params[226]
            lv945: R.Tensor((27648, 160), dtype="float16") = model_params[227]
            lv126 = R.call_tir(cls.fused_fused_decode2_NT_matmul7, (lv944, lv945, lv613), out_sinfo=R.Tensor((1, n, 27648), dtype="float16"))
            lv947 = R.call_tir(cls.fused_split3_silu1_multiply1, (lv126,), out_sinfo=R.Tensor((1, n, 13824), dtype="float16"))
            lv948: R.Tensor((5120, 1728), dtype="uint32") = model_params[228]
            lv949: R.Tensor((5120, 432), dtype="float16") = model_params[229]
            lv45 = R.call_tir(cls.fused_fused_decode3_fused_NT_matmul8_add1, (lv948, lv949, lv947, lv44), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv635: R.Tensor((5120,), dtype="float16") = model_params[240]
            lv624 = R.call_tir(cls.rms_norm, (lv45, lv635), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv952: R.Tensor((15360, 640), dtype="uint32") = model_params[232]
            lv953: R.Tensor((15360, 160), dtype="float16") = model_params[233]
            lv127_1 = R.call_tir(cls.fused_fused_decode_NT_matmul5, (lv952, lv953, lv624), out_sinfo=R.Tensor((1, n, 15360), dtype="float16"))
            lv627 = R.call_tir(cls.split2, (lv127_1,), out_sinfo=[R.Tensor((1, n, 5120), dtype="float16"), R.Tensor((1, n, 5120), dtype="float16"), R.Tensor((1, n, 5120), dtype="float16")])
            lv628: R.Tensor((1, n, 5120), dtype="float16") = lv627[0]
            lv629_1 = R.call_tir(cls.reshape5, (lv628,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv630_2: R.Tensor((1, n, 5120), dtype="float16") = lv627[1]
            lv631 = R.call_tir(cls.reshape5, (lv630_2,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv632_1: R.Tensor((1, n, 5120), dtype="float16") = lv627[2]
            lv633_1 = R.call_tir(cls.reshape5, (lv632_1,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv634_1: R.Object = R.call_pure_packed("vm.builtin.paged_attention_kv_cache_append", lv607_1, cls.kv_cache_transpose_append, lv631, lv633_1, R.prim_value(23), sinfo_args=(R.Object,))
            lv635_1 = R.call_dps_packed("vm.builtin.paged_attention_kv_cache_attention", (lv634_1, cls.attention, lv629_1, R.prim_value(23), R.prim_value(1), R.prim_value(T.float32(1)), R.prim_value(10000)), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv636 = R.call_tir(cls.reshape6, (lv635_1,), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv955: R.Tensor((5120, 640), dtype="uint32") = model_params[234]
            lv956: R.Tensor((5120, 160), dtype="float16") = model_params[235]
            lv46_1 = R.call_tir(cls.fused_fused_decode1_fused_NT_matmul6_add1, (lv955, lv956, lv636, lv45), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv640_1: R.Tensor((5120,), dtype="float16") = model_params[241]
            lv640_2 = R.call_tir(cls.rms_norm, (lv46_1, lv640_1), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv959: R.Tensor((27648, 640), dtype="uint32") = model_params[236]
            lv960: R.Tensor((27648, 160), dtype="float16") = model_params[237]
            lv128 = R.call_tir(cls.fused_fused_decode2_NT_matmul7, (lv959, lv960, lv640_2), out_sinfo=R.Tensor((1, n, 27648), dtype="float16"))
            lv962 = R.call_tir(cls.fused_split3_silu1_multiply1, (lv128,), out_sinfo=R.Tensor((1, n, 13824), dtype="float16"))
            lv963: R.Tensor((5120, 1728), dtype="uint32") = model_params[238]
            lv964: R.Tensor((5120, 432), dtype="float16") = model_params[239]
            lv47 = R.call_tir(cls.fused_fused_decode3_fused_NT_matmul8_add1, (lv963, lv964, lv962, lv46_1), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv645_1: R.Tensor((5120,), dtype="float16") = model_params[250]
            lv651 = R.call_tir(cls.rms_norm, (lv47, lv645_1), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv967: R.Tensor((15360, 640), dtype="uint32") = model_params[242]
            lv968: R.Tensor((15360, 160), dtype="float16") = model_params[243]
            lv129 = R.call_tir(cls.fused_fused_decode_NT_matmul5, (lv967, lv968, lv651), out_sinfo=R.Tensor((1, n, 15360), dtype="float16"))
            lv654 = R.call_tir(cls.split2, (lv129,), out_sinfo=[R.Tensor((1, n, 5120), dtype="float16"), R.Tensor((1, n, 5120), dtype="float16"), R.Tensor((1, n, 5120), dtype="float16")])
            lv655_1: R.Tensor((1, n, 5120), dtype="float16") = lv654[0]
            lv656_1 = R.call_tir(cls.reshape5, (lv655_1,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv657: R.Tensor((1, n, 5120), dtype="float16") = lv654[1]
            lv658 = R.call_tir(cls.reshape5, (lv657,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv659_1: R.Tensor((1, n, 5120), dtype="float16") = lv654[2]
            lv660_1 = R.call_tir(cls.reshape5, (lv659_1,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv661: R.Object = R.call_pure_packed("vm.builtin.paged_attention_kv_cache_append", lv634_1, cls.kv_cache_transpose_append, lv658, lv660_1, R.prim_value(24), sinfo_args=(R.Object,))
            lv662_1 = R.call_dps_packed("vm.builtin.paged_attention_kv_cache_attention", (lv661, cls.attention, lv656_1, R.prim_value(24), R.prim_value(1), R.prim_value(T.float32(1)), R.prim_value(10000)), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv663_1 = R.call_tir(cls.reshape6, (lv662_1,), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv970: R.Tensor((5120, 640), dtype="uint32") = model_params[244]
            lv971: R.Tensor((5120, 160), dtype="float16") = model_params[245]
            lv48 = R.call_tir(cls.fused_fused_decode1_fused_NT_matmul6_add1, (lv970, lv971, lv663_1, lv47), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv650: R.Tensor((5120,), dtype="float16") = model_params[251]
            lv667_1 = R.call_tir(cls.rms_norm, (lv48, lv650), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv974: R.Tensor((27648, 640), dtype="uint32") = model_params[246]
            lv975: R.Tensor((27648, 160), dtype="float16") = model_params[247]
            lv130 = R.call_tir(cls.fused_fused_decode2_NT_matmul7, (lv974, lv975, lv667_1), out_sinfo=R.Tensor((1, n, 27648), dtype="float16"))
            lv977 = R.call_tir(cls.fused_split3_silu1_multiply1, (lv130,), out_sinfo=R.Tensor((1, n, 13824), dtype="float16"))
            lv978: R.Tensor((5120, 1728), dtype="uint32") = model_params[248]
            lv979: R.Tensor((5120, 432), dtype="float16") = model_params[249]
            lv49 = R.call_tir(cls.fused_fused_decode3_fused_NT_matmul8_add1, (lv978, lv979, lv977, lv48), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv655_2: R.Tensor((5120,), dtype="float16") = model_params[260]
            lv678_1 = R.call_tir(cls.rms_norm, (lv49, lv655_2), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv982: R.Tensor((15360, 640), dtype="uint32") = model_params[252]
            lv983: R.Tensor((15360, 160), dtype="float16") = model_params[253]
            lv131 = R.call_tir(cls.fused_fused_decode_NT_matmul5, (lv982, lv983, lv678_1), out_sinfo=R.Tensor((1, n, 15360), dtype="float16"))
            lv681 = R.call_tir(cls.split2, (lv131,), out_sinfo=[R.Tensor((1, n, 5120), dtype="float16"), R.Tensor((1, n, 5120), dtype="float16"), R.Tensor((1, n, 5120), dtype="float16")])
            lv682_1: R.Tensor((1, n, 5120), dtype="float16") = lv681[0]
            lv683_1 = R.call_tir(cls.reshape5, (lv682_1,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv684: R.Tensor((1, n, 5120), dtype="float16") = lv681[1]
            lv685_1 = R.call_tir(cls.reshape5, (lv684,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv686_1: R.Tensor((1, n, 5120), dtype="float16") = lv681[2]
            lv687 = R.call_tir(cls.reshape5, (lv686_1,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv688: R.Object = R.call_pure_packed("vm.builtin.paged_attention_kv_cache_append", lv661, cls.kv_cache_transpose_append, lv685_1, lv687, R.prim_value(25), sinfo_args=(R.Object,))
            lv689_1 = R.call_dps_packed("vm.builtin.paged_attention_kv_cache_attention", (lv688, cls.attention, lv683_1, R.prim_value(25), R.prim_value(1), R.prim_value(T.float32(1)), R.prim_value(10000)), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv690_1 = R.call_tir(cls.reshape6, (lv689_1,), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv985: R.Tensor((5120, 640), dtype="uint32") = model_params[254]
            lv986: R.Tensor((5120, 160), dtype="float16") = model_params[255]
            lv50 = R.call_tir(cls.fused_fused_decode1_fused_NT_matmul6_add1, (lv985, lv986, lv690_1, lv49), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv660_2: R.Tensor((5120,), dtype="float16") = model_params[261]
            lv694_1 = R.call_tir(cls.rms_norm, (lv50, lv660_2), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv989: R.Tensor((27648, 640), dtype="uint32") = model_params[256]
            lv990: R.Tensor((27648, 160), dtype="float16") = model_params[257]
            lv132 = R.call_tir(cls.fused_fused_decode2_NT_matmul7, (lv989, lv990, lv694_1), out_sinfo=R.Tensor((1, n, 27648), dtype="float16"))
            lv992 = R.call_tir(cls.fused_split3_silu1_multiply1, (lv132,), out_sinfo=R.Tensor((1, n, 13824), dtype="float16"))
            lv993: R.Tensor((5120, 1728), dtype="uint32") = model_params[258]
            lv994: R.Tensor((5120, 432), dtype="float16") = model_params[259]
            lv51 = R.call_tir(cls.fused_fused_decode3_fused_NT_matmul8_add1, (lv993, lv994, lv992, lv50), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv665: R.Tensor((5120,), dtype="float16") = model_params[270]
            lv705_1 = R.call_tir(cls.rms_norm, (lv51, lv665), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv997: R.Tensor((15360, 640), dtype="uint32") = model_params[262]
            lv998: R.Tensor((15360, 160), dtype="float16") = model_params[263]
            lv133 = R.call_tir(cls.fused_fused_decode_NT_matmul5, (lv997, lv998, lv705_1), out_sinfo=R.Tensor((1, n, 15360), dtype="float16"))
            lv708_1 = R.call_tir(cls.split2, (lv133,), out_sinfo=[R.Tensor((1, n, 5120), dtype="float16"), R.Tensor((1, n, 5120), dtype="float16"), R.Tensor((1, n, 5120), dtype="float16")])
            lv709_1: R.Tensor((1, n, 5120), dtype="float16") = lv708_1[0]
            lv710 = R.call_tir(cls.reshape5, (lv709_1,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv711: R.Tensor((1, n, 5120), dtype="float16") = lv708_1[1]
            lv712_1 = R.call_tir(cls.reshape5, (lv711,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv713_1: R.Tensor((1, n, 5120), dtype="float16") = lv708_1[2]
            lv714 = R.call_tir(cls.reshape5, (lv713_1,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv715_1: R.Object = R.call_pure_packed("vm.builtin.paged_attention_kv_cache_append", lv688, cls.kv_cache_transpose_append, lv712_1, lv714, R.prim_value(26), sinfo_args=(R.Object,))
            lv716_1 = R.call_dps_packed("vm.builtin.paged_attention_kv_cache_attention", (lv715_1, cls.attention, lv710, R.prim_value(26), R.prim_value(1), R.prim_value(T.float32(1)), R.prim_value(10000)), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv717 = R.call_tir(cls.reshape6, (lv716_1,), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv1000: R.Tensor((5120, 640), dtype="uint32") = model_params[264]
            lv1001: R.Tensor((5120, 160), dtype="float16") = model_params[265]
            lv52 = R.call_tir(cls.fused_fused_decode1_fused_NT_matmul6_add1, (lv1000, lv1001, lv717, lv51), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv670_1: R.Tensor((5120,), dtype="float16") = model_params[271]
            lv721 = R.call_tir(cls.rms_norm, (lv52, lv670_1), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv1004: R.Tensor((27648, 640), dtype="uint32") = model_params[266]
            lv1005: R.Tensor((27648, 160), dtype="float16") = model_params[267]
            lv134 = R.call_tir(cls.fused_fused_decode2_NT_matmul7, (lv1004, lv1005, lv721), out_sinfo=R.Tensor((1, n, 27648), dtype="float16"))
            lv1007 = R.call_tir(cls.fused_split3_silu1_multiply1, (lv134,), out_sinfo=R.Tensor((1, n, 13824), dtype="float16"))
            lv1008: R.Tensor((5120, 1728), dtype="uint32") = model_params[268]
            lv1009: R.Tensor((5120, 432), dtype="float16") = model_params[269]
            lv53 = R.call_tir(cls.fused_fused_decode3_fused_NT_matmul8_add1, (lv1008, lv1009, lv1007, lv52), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv675_1: R.Tensor((5120,), dtype="float16") = model_params[280]
            lv732 = R.call_tir(cls.rms_norm, (lv53, lv675_1), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv1012: R.Tensor((15360, 640), dtype="uint32") = model_params[272]
            lv1013: R.Tensor((15360, 160), dtype="float16") = model_params[273]
            lv135 = R.call_tir(cls.fused_fused_decode_NT_matmul5, (lv1012, lv1013, lv732), out_sinfo=R.Tensor((1, n, 15360), dtype="float16"))
            lv735_1 = R.call_tir(cls.split2, (lv135,), out_sinfo=[R.Tensor((1, n, 5120), dtype="float16"), R.Tensor((1, n, 5120), dtype="float16"), R.Tensor((1, n, 5120), dtype="float16")])
            lv736: R.Tensor((1, n, 5120), dtype="float16") = lv735_1[0]
            lv737_1 = R.call_tir(cls.reshape5, (lv736,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv738_1: R.Tensor((1, n, 5120), dtype="float16") = lv735_1[1]
            lv739_1 = R.call_tir(cls.reshape5, (lv738_1,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv740: R.Tensor((1, n, 5120), dtype="float16") = lv735_1[2]
            lv741 = R.call_tir(cls.reshape5, (lv740,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv742_1: R.Object = R.call_pure_packed("vm.builtin.paged_attention_kv_cache_append", lv715_1, cls.kv_cache_transpose_append, lv739_1, lv741, R.prim_value(27), sinfo_args=(R.Object,))
            lv743_1 = R.call_dps_packed("vm.builtin.paged_attention_kv_cache_attention", (lv742_1, cls.attention, lv737_1, R.prim_value(27), R.prim_value(1), R.prim_value(T.float32(1)), R.prim_value(10000)), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv744 = R.call_tir(cls.reshape6, (lv743_1,), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv1015: R.Tensor((5120, 640), dtype="uint32") = model_params[274]
            lv1016: R.Tensor((5120, 160), dtype="float16") = model_params[275]
            lv54 = R.call_tir(cls.fused_fused_decode1_fused_NT_matmul6_add1, (lv1015, lv1016, lv744, lv53), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv680: R.Tensor((5120,), dtype="float16") = model_params[281]
            lv748 = R.call_tir(cls.rms_norm, (lv54, lv680), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv1019: R.Tensor((27648, 640), dtype="uint32") = model_params[276]
            lv1020: R.Tensor((27648, 160), dtype="float16") = model_params[277]
            lv136 = R.call_tir(cls.fused_fused_decode2_NT_matmul7, (lv1019, lv1020, lv748), out_sinfo=R.Tensor((1, n, 27648), dtype="float16"))
            lv1022 = R.call_tir(cls.fused_split3_silu1_multiply1, (lv136,), out_sinfo=R.Tensor((1, n, 13824), dtype="float16"))
            lv1023: R.Tensor((5120, 1728), dtype="uint32") = model_params[278]
            lv1024: R.Tensor((5120, 432), dtype="float16") = model_params[279]
            lv55 = R.call_tir(cls.fused_fused_decode3_fused_NT_matmul8_add1, (lv1023, lv1024, lv1022, lv54), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv685_2: R.Tensor((5120,), dtype="float16") = model_params[290]
            lv759 = R.call_tir(cls.rms_norm, (lv55, lv685_2), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv1027: R.Tensor((15360, 640), dtype="uint32") = model_params[282]
            lv1028: R.Tensor((15360, 160), dtype="float16") = model_params[283]
            lv137 = R.call_tir(cls.fused_fused_decode_NT_matmul5, (lv1027, lv1028, lv759), out_sinfo=R.Tensor((1, n, 15360), dtype="float16"))
            lv762 = R.call_tir(cls.split2, (lv137,), out_sinfo=[R.Tensor((1, n, 5120), dtype="float16"), R.Tensor((1, n, 5120), dtype="float16"), R.Tensor((1, n, 5120), dtype="float16")])
            lv763: R.Tensor((1, n, 5120), dtype="float16") = lv762[0]
            lv764_1 = R.call_tir(cls.reshape5, (lv763,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv765_1: R.Tensor((1, n, 5120), dtype="float16") = lv762[1]
            lv766 = R.call_tir(cls.reshape5, (lv765_1,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv767_1: R.Tensor((1, n, 5120), dtype="float16") = lv762[2]
            lv768_1 = R.call_tir(cls.reshape5, (lv767_1,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv769_1: R.Object = R.call_pure_packed("vm.builtin.paged_attention_kv_cache_append", lv742_1, cls.kv_cache_transpose_append, lv766, lv768_1, R.prim_value(28), sinfo_args=(R.Object,))
            lv770 = R.call_dps_packed("vm.builtin.paged_attention_kv_cache_attention", (lv769_1, cls.attention, lv764_1, R.prim_value(28), R.prim_value(1), R.prim_value(T.float32(1)), R.prim_value(10000)), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv771 = R.call_tir(cls.reshape6, (lv770,), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv1030: R.Tensor((5120, 640), dtype="uint32") = model_params[284]
            lv1031: R.Tensor((5120, 160), dtype="float16") = model_params[285]
            lv56 = R.call_tir(cls.fused_fused_decode1_fused_NT_matmul6_add1, (lv1030, lv1031, lv771, lv55), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv690_2: R.Tensor((5120,), dtype="float16") = model_params[291]
            lv775_1 = R.call_tir(cls.rms_norm, (lv56, lv690_2), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv1034: R.Tensor((27648, 640), dtype="uint32") = model_params[286]
            lv1035: R.Tensor((27648, 160), dtype="float16") = model_params[287]
            lv138_1 = R.call_tir(cls.fused_fused_decode2_NT_matmul7, (lv1034, lv1035, lv775_1), out_sinfo=R.Tensor((1, n, 27648), dtype="float16"))
            lv1037 = R.call_tir(cls.fused_split3_silu1_multiply1, (lv138_1,), out_sinfo=R.Tensor((1, n, 13824), dtype="float16"))
            lv1038: R.Tensor((5120, 1728), dtype="uint32") = model_params[288]
            lv1039: R.Tensor((5120, 432), dtype="float16") = model_params[289]
            lv57_1 = R.call_tir(cls.fused_fused_decode3_fused_NT_matmul8_add1, (lv1038, lv1039, lv1037, lv56), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv695: R.Tensor((5120,), dtype="float16") = model_params[300]
            lv786 = R.call_tir(cls.rms_norm, (lv57_1, lv695), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv1042: R.Tensor((15360, 640), dtype="uint32") = model_params[292]
            lv1043: R.Tensor((15360, 160), dtype="float16") = model_params[293]
            lv139 = R.call_tir(cls.fused_fused_decode_NT_matmul5, (lv1042, lv1043, lv786), out_sinfo=R.Tensor((1, n, 15360), dtype="float16"))
            lv789 = R.call_tir(cls.split2, (lv139,), out_sinfo=[R.Tensor((1, n, 5120), dtype="float16"), R.Tensor((1, n, 5120), dtype="float16"), R.Tensor((1, n, 5120), dtype="float16")])
            lv790_1: R.Tensor((1, n, 5120), dtype="float16") = lv789[0]
            lv791_1 = R.call_tir(cls.reshape5, (lv790_1,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv792: R.Tensor((1, n, 5120), dtype="float16") = lv789[1]
            lv793 = R.call_tir(cls.reshape5, (lv792,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv794_1: R.Tensor((1, n, 5120), dtype="float16") = lv789[2]
            lv795_1 = R.call_tir(cls.reshape5, (lv794_1,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv796: R.Object = R.call_pure_packed("vm.builtin.paged_attention_kv_cache_append", lv769_1, cls.kv_cache_transpose_append, lv793, lv795_1, R.prim_value(29), sinfo_args=(R.Object,))
            lv797_1 = R.call_dps_packed("vm.builtin.paged_attention_kv_cache_attention", (lv796, cls.attention, lv791_1, R.prim_value(29), R.prim_value(1), R.prim_value(T.float32(1)), R.prim_value(10000)), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv798_1 = R.call_tir(cls.reshape6, (lv797_1,), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv1045: R.Tensor((5120, 640), dtype="uint32") = model_params[294]
            lv1046: R.Tensor((5120, 160), dtype="float16") = model_params[295]
            lv58 = R.call_tir(cls.fused_fused_decode1_fused_NT_matmul6_add1, (lv1045, lv1046, lv798_1, lv57_1), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv700_1: R.Tensor((5120,), dtype="float16") = model_params[301]
            lv802_1 = R.call_tir(cls.rms_norm, (lv58, lv700_1), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv1049: R.Tensor((27648, 640), dtype="uint32") = model_params[296]
            lv1050: R.Tensor((27648, 160), dtype="float16") = model_params[297]
            lv140 = R.call_tir(cls.fused_fused_decode2_NT_matmul7, (lv1049, lv1050, lv802_1), out_sinfo=R.Tensor((1, n, 27648), dtype="float16"))
            lv1052 = R.call_tir(cls.fused_split3_silu1_multiply1, (lv140,), out_sinfo=R.Tensor((1, n, 13824), dtype="float16"))
            lv1053: R.Tensor((5120, 1728), dtype="uint32") = model_params[298]
            lv1054: R.Tensor((5120, 432), dtype="float16") = model_params[299]
            lv59 = R.call_tir(cls.fused_fused_decode3_fused_NT_matmul8_add1, (lv1053, lv1054, lv1052, lv58), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv705_2: R.Tensor((5120,), dtype="float16") = model_params[310]
            lv813_1 = R.call_tir(cls.rms_norm, (lv59, lv705_2), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv1057: R.Tensor((15360, 640), dtype="uint32") = model_params[302]
            lv1058: R.Tensor((15360, 160), dtype="float16") = model_params[303]
            lv141_1 = R.call_tir(cls.fused_fused_decode_NT_matmul5, (lv1057, lv1058, lv813_1), out_sinfo=R.Tensor((1, n, 15360), dtype="float16"))
            lv816 = R.call_tir(cls.split2, (lv141_1,), out_sinfo=[R.Tensor((1, n, 5120), dtype="float16"), R.Tensor((1, n, 5120), dtype="float16"), R.Tensor((1, n, 5120), dtype="float16")])
            lv817_1: R.Tensor((1, n, 5120), dtype="float16") = lv816[0]
            lv818_1 = R.call_tir(cls.reshape5, (lv817_1,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv819: R.Tensor((1, n, 5120), dtype="float16") = lv816[1]
            lv820_1 = R.call_tir(cls.reshape5, (lv819,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv821_1: R.Tensor((1, n, 5120), dtype="float16") = lv816[2]
            lv822 = R.call_tir(cls.reshape5, (lv821_1,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv823: R.Object = R.call_pure_packed("vm.builtin.paged_attention_kv_cache_append", lv796, cls.kv_cache_transpose_append, lv820_1, lv822, R.prim_value(30), sinfo_args=(R.Object,))
            lv824_1 = R.call_dps_packed("vm.builtin.paged_attention_kv_cache_attention", (lv823, cls.attention, lv818_1, R.prim_value(30), R.prim_value(1), R.prim_value(T.float32(1)), R.prim_value(10000)), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv825_1 = R.call_tir(cls.reshape6, (lv824_1,), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv1060: R.Tensor((5120, 640), dtype="uint32") = model_params[304]
            lv1061: R.Tensor((5120, 160), dtype="float16") = model_params[305]
            lv60_1 = R.call_tir(cls.fused_fused_decode1_fused_NT_matmul6_add1, (lv1060, lv1061, lv825_1, lv59), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv710_1: R.Tensor((5120,), dtype="float16") = model_params[311]
            lv829_1 = R.call_tir(cls.rms_norm, (lv60_1, lv710_1), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv1064: R.Tensor((27648, 640), dtype="uint32") = model_params[306]
            lv1065: R.Tensor((27648, 160), dtype="float16") = model_params[307]
            lv142_1 = R.call_tir(cls.fused_fused_decode2_NT_matmul7, (lv1064, lv1065, lv829_1), out_sinfo=R.Tensor((1, n, 27648), dtype="float16"))
            lv1067 = R.call_tir(cls.fused_split3_silu1_multiply1, (lv142_1,), out_sinfo=R.Tensor((1, n, 13824), dtype="float16"))
            lv1068: R.Tensor((5120, 1728), dtype="uint32") = model_params[308]
            lv1069: R.Tensor((5120, 432), dtype="float16") = model_params[309]
            lv61_1 = R.call_tir(cls.fused_fused_decode3_fused_NT_matmul8_add1, (lv1068, lv1069, lv1067, lv60_1), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv715_2: R.Tensor((5120,), dtype="float16") = model_params[320]
            lv840_1 = R.call_tir(cls.rms_norm, (lv61_1, lv715_2), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv1072: R.Tensor((15360, 640), dtype="uint32") = model_params[312]
            lv1073: R.Tensor((15360, 160), dtype="float16") = model_params[313]
            lv143_1 = R.call_tir(cls.fused_fused_decode_NT_matmul5, (lv1072, lv1073, lv840_1), out_sinfo=R.Tensor((1, n, 15360), dtype="float16"))
            lv843_1 = R.call_tir(cls.split2, (lv143_1,), out_sinfo=[R.Tensor((1, n, 5120), dtype="float16"), R.Tensor((1, n, 5120), dtype="float16"), R.Tensor((1, n, 5120), dtype="float16")])
            lv844_1: R.Tensor((1, n, 5120), dtype="float16") = lv843_1[0]
            lv845 = R.call_tir(cls.reshape5, (lv844_1,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv846: R.Tensor((1, n, 5120), dtype="float16") = lv843_1[1]
            lv847_1 = R.call_tir(cls.reshape5, (lv846,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv848_1: R.Tensor((1, n, 5120), dtype="float16") = lv843_1[2]
            lv849 = R.call_tir(cls.reshape5, (lv848_1,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv850_1: R.Object = R.call_pure_packed("vm.builtin.paged_attention_kv_cache_append", lv823, cls.kv_cache_transpose_append, lv847_1, lv849, R.prim_value(31), sinfo_args=(R.Object,))
            lv851_1 = R.call_dps_packed("vm.builtin.paged_attention_kv_cache_attention", (lv850_1, cls.attention, lv845, R.prim_value(31), R.prim_value(1), R.prim_value(T.float32(1)), R.prim_value(10000)), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv852 = R.call_tir(cls.reshape6, (lv851_1,), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv1075: R.Tensor((5120, 640), dtype="uint32") = model_params[314]
            lv1076: R.Tensor((5120, 160), dtype="float16") = model_params[315]
            lv62_1 = R.call_tir(cls.fused_fused_decode1_fused_NT_matmul6_add1, (lv1075, lv1076, lv852, lv61_1), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv720_1: R.Tensor((5120,), dtype="float16") = model_params[321]
            lv856 = R.call_tir(cls.rms_norm, (lv62_1, lv720_1), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv1079: R.Tensor((27648, 640), dtype="uint32") = model_params[316]
            lv1080: R.Tensor((27648, 160), dtype="float16") = model_params[317]
            lv144_1 = R.call_tir(cls.fused_fused_decode2_NT_matmul7, (lv1079, lv1080, lv856), out_sinfo=R.Tensor((1, n, 27648), dtype="float16"))
            lv1082 = R.call_tir(cls.fused_split3_silu1_multiply1, (lv144_1,), out_sinfo=R.Tensor((1, n, 13824), dtype="float16"))
            lv1083: R.Tensor((5120, 1728), dtype="uint32") = model_params[318]
            lv1084: R.Tensor((5120, 432), dtype="float16") = model_params[319]
            lv63_1 = R.call_tir(cls.fused_fused_decode3_fused_NT_matmul8_add1, (lv1083, lv1084, lv1082, lv62_1), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv725: R.Tensor((5120,), dtype="float16") = model_params[330]
            lv867 = R.call_tir(cls.rms_norm, (lv63_1, lv725), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv1087: R.Tensor((15360, 640), dtype="uint32") = model_params[322]
            lv1088: R.Tensor((15360, 160), dtype="float16") = model_params[323]
            lv145_1 = R.call_tir(cls.fused_fused_decode_NT_matmul5, (lv1087, lv1088, lv867), out_sinfo=R.Tensor((1, n, 15360), dtype="float16"))
            lv870_1 = R.call_tir(cls.split2, (lv145_1,), out_sinfo=[R.Tensor((1, n, 5120), dtype="float16"), R.Tensor((1, n, 5120), dtype="float16"), R.Tensor((1, n, 5120), dtype="float16")])
            lv871: R.Tensor((1, n, 5120), dtype="float16") = lv870_1[0]
            lv872_1 = R.call_tir(cls.reshape5, (lv871,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv873_1: R.Tensor((1, n, 5120), dtype="float16") = lv870_1[1]
            lv874_1 = R.call_tir(cls.reshape5, (lv873_1,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv875: R.Tensor((1, n, 5120), dtype="float16") = lv870_1[2]
            lv876 = R.call_tir(cls.reshape5, (lv875,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv877_1: R.Object = R.call_pure_packed("vm.builtin.paged_attention_kv_cache_append", lv850_1, cls.kv_cache_transpose_append, lv874_1, lv876, R.prim_value(32), sinfo_args=(R.Object,))
            lv878_1 = R.call_dps_packed("vm.builtin.paged_attention_kv_cache_attention", (lv877_1, cls.attention, lv872_1, R.prim_value(32), R.prim_value(1), R.prim_value(T.float32(1)), R.prim_value(10000)), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv879 = R.call_tir(cls.reshape6, (lv878_1,), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv1090: R.Tensor((5120, 640), dtype="uint32") = model_params[324]
            lv1091: R.Tensor((5120, 160), dtype="float16") = model_params[325]
            lv64_1 = R.call_tir(cls.fused_fused_decode1_fused_NT_matmul6_add1, (lv1090, lv1091, lv879, lv63_1), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv730_1: R.Tensor((5120,), dtype="float16") = model_params[331]
            lv883 = R.call_tir(cls.rms_norm, (lv64_1, lv730_1), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv1094: R.Tensor((27648, 640), dtype="uint32") = model_params[326]
            lv1095: R.Tensor((27648, 160), dtype="float16") = model_params[327]
            lv146_1 = R.call_tir(cls.fused_fused_decode2_NT_matmul7, (lv1094, lv1095, lv883), out_sinfo=R.Tensor((1, n, 27648), dtype="float16"))
            lv1097 = R.call_tir(cls.fused_split3_silu1_multiply1, (lv146_1,), out_sinfo=R.Tensor((1, n, 13824), dtype="float16"))
            lv1098: R.Tensor((5120, 1728), dtype="uint32") = model_params[328]
            lv1099: R.Tensor((5120, 432), dtype="float16") = model_params[329]
            lv65_1 = R.call_tir(cls.fused_fused_decode3_fused_NT_matmul8_add1, (lv1098, lv1099, lv1097, lv64_1), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv735_2: R.Tensor((5120,), dtype="float16") = model_params[340]
            lv894 = R.call_tir(cls.rms_norm, (lv65_1, lv735_2), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv1102: R.Tensor((15360, 640), dtype="uint32") = model_params[332]
            lv1103: R.Tensor((15360, 160), dtype="float16") = model_params[333]
            lv147_1 = R.call_tir(cls.fused_fused_decode_NT_matmul5, (lv1102, lv1103, lv894), out_sinfo=R.Tensor((1, n, 15360), dtype="float16"))
            lv897 = R.call_tir(cls.split2, (lv147_1,), out_sinfo=[R.Tensor((1, n, 5120), dtype="float16"), R.Tensor((1, n, 5120), dtype="float16"), R.Tensor((1, n, 5120), dtype="float16")])
            lv898: R.Tensor((1, n, 5120), dtype="float16") = lv897[0]
            lv899_1 = R.call_tir(cls.reshape5, (lv898,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv900_1: R.Tensor((1, n, 5120), dtype="float16") = lv897[1]
            lv901 = R.call_tir(cls.reshape5, (lv900_1,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv902_1: R.Tensor((1, n, 5120), dtype="float16") = lv897[2]
            lv903_1 = R.call_tir(cls.reshape5, (lv902_1,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv904_1: R.Object = R.call_pure_packed("vm.builtin.paged_attention_kv_cache_append", lv877_1, cls.kv_cache_transpose_append, lv901, lv903_1, R.prim_value(33), sinfo_args=(R.Object,))
            lv905 = R.call_dps_packed("vm.builtin.paged_attention_kv_cache_attention", (lv904_1, cls.attention, lv899_1, R.prim_value(33), R.prim_value(1), R.prim_value(T.float32(1)), R.prim_value(10000)), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv906 = R.call_tir(cls.reshape6, (lv905,), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv1105: R.Tensor((5120, 640), dtype="uint32") = model_params[334]
            lv1106: R.Tensor((5120, 160), dtype="float16") = model_params[335]
            lv66_1 = R.call_tir(cls.fused_fused_decode1_fused_NT_matmul6_add1, (lv1105, lv1106, lv906, lv65_1), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv740_1: R.Tensor((5120,), dtype="float16") = model_params[341]
            lv910_1 = R.call_tir(cls.rms_norm, (lv66_1, lv740_1), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv1109: R.Tensor((27648, 640), dtype="uint32") = model_params[336]
            lv1110: R.Tensor((27648, 160), dtype="float16") = model_params[337]
            lv148_1 = R.call_tir(cls.fused_fused_decode2_NT_matmul7, (lv1109, lv1110, lv910_1), out_sinfo=R.Tensor((1, n, 27648), dtype="float16"))
            lv1112 = R.call_tir(cls.fused_split3_silu1_multiply1, (lv148_1,), out_sinfo=R.Tensor((1, n, 13824), dtype="float16"))
            lv1113: R.Tensor((5120, 1728), dtype="uint32") = model_params[338]
            lv1114: R.Tensor((5120, 432), dtype="float16") = model_params[339]
            lv67_1 = R.call_tir(cls.fused_fused_decode3_fused_NT_matmul8_add1, (lv1113, lv1114, lv1112, lv66_1), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv745_1: R.Tensor((5120,), dtype="float16") = model_params[350]
            lv921 = R.call_tir(cls.rms_norm, (lv67_1, lv745_1), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv1117: R.Tensor((15360, 640), dtype="uint32") = model_params[342]
            lv1118: R.Tensor((15360, 160), dtype="float16") = model_params[343]
            lv149_1 = R.call_tir(cls.fused_fused_decode_NT_matmul5, (lv1117, lv1118, lv921), out_sinfo=R.Tensor((1, n, 15360), dtype="float16"))
            lv924 = R.call_tir(cls.split2, (lv149_1,), out_sinfo=[R.Tensor((1, n, 5120), dtype="float16"), R.Tensor((1, n, 5120), dtype="float16"), R.Tensor((1, n, 5120), dtype="float16")])
            lv925_1: R.Tensor((1, n, 5120), dtype="float16") = lv924[0]
            lv926_1 = R.call_tir(cls.reshape5, (lv925_1,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv927: R.Tensor((1, n, 5120), dtype="float16") = lv924[1]
            lv928 = R.call_tir(cls.reshape5, (lv927,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv929_1: R.Tensor((1, n, 5120), dtype="float16") = lv924[2]
            lv930_1 = R.call_tir(cls.reshape5, (lv929_1,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv931: R.Object = R.call_pure_packed("vm.builtin.paged_attention_kv_cache_append", lv904_1, cls.kv_cache_transpose_append, lv928, lv930_1, R.prim_value(34), sinfo_args=(R.Object,))
            lv932_1 = R.call_dps_packed("vm.builtin.paged_attention_kv_cache_attention", (lv931, cls.attention, lv926_1, R.prim_value(34), R.prim_value(1), R.prim_value(T.float32(1)), R.prim_value(10000)), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv933_1 = R.call_tir(cls.reshape6, (lv932_1,), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv1120: R.Tensor((5120, 640), dtype="uint32") = model_params[344]
            lv1121: R.Tensor((5120, 160), dtype="float16") = model_params[345]
            lv68_1 = R.call_tir(cls.fused_fused_decode1_fused_NT_matmul6_add1, (lv1120, lv1121, lv933_1, lv67_1), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv750_1: R.Tensor((5120,), dtype="float16") = model_params[351]
            lv937_1 = R.call_tir(cls.rms_norm, (lv68_1, lv750_1), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv1124: R.Tensor((27648, 640), dtype="uint32") = model_params[346]
            lv1125: R.Tensor((27648, 160), dtype="float16") = model_params[347]
            lv150_1 = R.call_tir(cls.fused_fused_decode2_NT_matmul7, (lv1124, lv1125, lv937_1), out_sinfo=R.Tensor((1, n, 27648), dtype="float16"))
            lv1127 = R.call_tir(cls.fused_split3_silu1_multiply1, (lv150_1,), out_sinfo=R.Tensor((1, n, 13824), dtype="float16"))
            lv1128: R.Tensor((5120, 1728), dtype="uint32") = model_params[348]
            lv1129: R.Tensor((5120, 432), dtype="float16") = model_params[349]
            lv69_1 = R.call_tir(cls.fused_fused_decode3_fused_NT_matmul8_add1, (lv1128, lv1129, lv1127, lv68_1), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv755: R.Tensor((5120,), dtype="float16") = model_params[360]
            lv948_1 = R.call_tir(cls.rms_norm, (lv69_1, lv755), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv1132: R.Tensor((15360, 640), dtype="uint32") = model_params[352]
            lv1133: R.Tensor((15360, 160), dtype="float16") = model_params[353]
            lv151 = R.call_tir(cls.fused_fused_decode_NT_matmul5, (lv1132, lv1133, lv948_1), out_sinfo=R.Tensor((1, n, 15360), dtype="float16"))
            lv951 = R.call_tir(cls.split2, (lv151,), out_sinfo=[R.Tensor((1, n, 5120), dtype="float16"), R.Tensor((1, n, 5120), dtype="float16"), R.Tensor((1, n, 5120), dtype="float16")])
            lv952_1: R.Tensor((1, n, 5120), dtype="float16") = lv951[0]
            lv953_1 = R.call_tir(cls.reshape5, (lv952_1,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv954: R.Tensor((1, n, 5120), dtype="float16") = lv951[1]
            lv955_1 = R.call_tir(cls.reshape5, (lv954,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv956_1: R.Tensor((1, n, 5120), dtype="float16") = lv951[2]
            lv957 = R.call_tir(cls.reshape5, (lv956_1,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv958: R.Object = R.call_pure_packed("vm.builtin.paged_attention_kv_cache_append", lv931, cls.kv_cache_transpose_append, lv955_1, lv957, R.prim_value(35), sinfo_args=(R.Object,))
            lv959_1 = R.call_dps_packed("vm.builtin.paged_attention_kv_cache_attention", (lv958, cls.attention, lv953_1, R.prim_value(35), R.prim_value(1), R.prim_value(T.float32(1)), R.prim_value(10000)), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv960_1 = R.call_tir(cls.reshape6, (lv959_1,), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv1135: R.Tensor((5120, 640), dtype="uint32") = model_params[354]
            lv1136: R.Tensor((5120, 160), dtype="float16") = model_params[355]
            lv70 = R.call_tir(cls.fused_fused_decode1_fused_NT_matmul6_add1, (lv1135, lv1136, lv960_1, lv69_1), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv760_1: R.Tensor((5120,), dtype="float16") = model_params[361]
            lv964_1 = R.call_tir(cls.rms_norm, (lv70, lv760_1), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv1139: R.Tensor((27648, 640), dtype="uint32") = model_params[356]
            lv1140: R.Tensor((27648, 160), dtype="float16") = model_params[357]
            lv152 = R.call_tir(cls.fused_fused_decode2_NT_matmul7, (lv1139, lv1140, lv964_1), out_sinfo=R.Tensor((1, n, 27648), dtype="float16"))
            lv1142 = R.call_tir(cls.fused_split3_silu1_multiply1, (lv152,), out_sinfo=R.Tensor((1, n, 13824), dtype="float16"))
            lv1143: R.Tensor((5120, 1728), dtype="uint32") = model_params[358]
            lv1144: R.Tensor((5120, 432), dtype="float16") = model_params[359]
            lv71 = R.call_tir(cls.fused_fused_decode3_fused_NT_matmul8_add1, (lv1143, lv1144, lv1142, lv70), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv765_2: R.Tensor((5120,), dtype="float16") = model_params[370]
            lv975_1 = R.call_tir(cls.rms_norm, (lv71, lv765_2), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv1147: R.Tensor((15360, 640), dtype="uint32") = model_params[362]
            lv1148: R.Tensor((15360, 160), dtype="float16") = model_params[363]
            lv153 = R.call_tir(cls.fused_fused_decode_NT_matmul5, (lv1147, lv1148, lv975_1), out_sinfo=R.Tensor((1, n, 15360), dtype="float16"))
            lv978_1 = R.call_tir(cls.split2, (lv153,), out_sinfo=[R.Tensor((1, n, 5120), dtype="float16"), R.Tensor((1, n, 5120), dtype="float16"), R.Tensor((1, n, 5120), dtype="float16")])
            lv979_1: R.Tensor((1, n, 5120), dtype="float16") = lv978_1[0]
            lv980 = R.call_tir(cls.reshape5, (lv979_1,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv981: R.Tensor((1, n, 5120), dtype="float16") = lv978_1[1]
            lv982_1 = R.call_tir(cls.reshape5, (lv981,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv983_1: R.Tensor((1, n, 5120), dtype="float16") = lv978_1[2]
            lv984 = R.call_tir(cls.reshape5, (lv983_1,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv985_1: R.Object = R.call_pure_packed("vm.builtin.paged_attention_kv_cache_append", lv958, cls.kv_cache_transpose_append, lv982_1, lv984, R.prim_value(36), sinfo_args=(R.Object,))
            lv986_1 = R.call_dps_packed("vm.builtin.paged_attention_kv_cache_attention", (lv985_1, cls.attention, lv980, R.prim_value(36), R.prim_value(1), R.prim_value(T.float32(1)), R.prim_value(10000)), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv987 = R.call_tir(cls.reshape6, (lv986_1,), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv1150: R.Tensor((5120, 640), dtype="uint32") = model_params[364]
            lv1151: R.Tensor((5120, 160), dtype="float16") = model_params[365]
            lv72 = R.call_tir(cls.fused_fused_decode1_fused_NT_matmul6_add1, (lv1150, lv1151, lv987, lv71), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv770_1: R.Tensor((5120,), dtype="float16") = model_params[371]
            lv991 = R.call_tir(cls.rms_norm, (lv72, lv770_1), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv1154: R.Tensor((27648, 640), dtype="uint32") = model_params[366]
            lv1155: R.Tensor((27648, 160), dtype="float16") = model_params[367]
            lv154_1 = R.call_tir(cls.fused_fused_decode2_NT_matmul7, (lv1154, lv1155, lv991), out_sinfo=R.Tensor((1, n, 27648), dtype="float16"))
            lv1157 = R.call_tir(cls.fused_split3_silu1_multiply1, (lv154_1,), out_sinfo=R.Tensor((1, n, 13824), dtype="float16"))
            lv1158: R.Tensor((5120, 1728), dtype="uint32") = model_params[368]
            lv1159: R.Tensor((5120, 432), dtype="float16") = model_params[369]
            lv73_1 = R.call_tir(cls.fused_fused_decode3_fused_NT_matmul8_add1, (lv1158, lv1159, lv1157, lv72), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv775_2: R.Tensor((5120,), dtype="float16") = model_params[380]
            lv1002 = R.call_tir(cls.rms_norm, (lv73_1, lv775_2), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv1162: R.Tensor((15360, 640), dtype="uint32") = model_params[372]
            lv1163: R.Tensor((15360, 160), dtype="float16") = model_params[373]
            lv155 = R.call_tir(cls.fused_fused_decode_NT_matmul5, (lv1162, lv1163, lv1002), out_sinfo=R.Tensor((1, n, 15360), dtype="float16"))
            lv1005_1 = R.call_tir(cls.split2, (lv155,), out_sinfo=[R.Tensor((1, n, 5120), dtype="float16"), R.Tensor((1, n, 5120), dtype="float16"), R.Tensor((1, n, 5120), dtype="float16")])
            lv1006: R.Tensor((1, n, 5120), dtype="float16") = lv1005_1[0]
            lv1007_1 = R.call_tir(cls.reshape5, (lv1006,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv1008_1: R.Tensor((1, n, 5120), dtype="float16") = lv1005_1[1]
            lv1009_1 = R.call_tir(cls.reshape5, (lv1008_1,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv1010: R.Tensor((1, n, 5120), dtype="float16") = lv1005_1[2]
            lv1011 = R.call_tir(cls.reshape5, (lv1010,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv1012_1: R.Object = R.call_pure_packed("vm.builtin.paged_attention_kv_cache_append", lv985_1, cls.kv_cache_transpose_append, lv1009_1, lv1011, R.prim_value(37), sinfo_args=(R.Object,))
            lv1013_1 = R.call_dps_packed("vm.builtin.paged_attention_kv_cache_attention", (lv1012_1, cls.attention, lv1007_1, R.prim_value(37), R.prim_value(1), R.prim_value(T.float32(1)), R.prim_value(10000)), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv1014 = R.call_tir(cls.reshape6, (lv1013_1,), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv1165: R.Tensor((5120, 640), dtype="uint32") = model_params[374]
            lv1166: R.Tensor((5120, 160), dtype="float16") = model_params[375]
            lv74 = R.call_tir(cls.fused_fused_decode1_fused_NT_matmul6_add1, (lv1165, lv1166, lv1014, lv73_1), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv780_1: R.Tensor((5120,), dtype="float16") = model_params[381]
            lv1018 = R.call_tir(cls.rms_norm, (lv74, lv780_1), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv1169: R.Tensor((27648, 640), dtype="uint32") = model_params[376]
            lv1170: R.Tensor((27648, 160), dtype="float16") = model_params[377]
            lv156 = R.call_tir(cls.fused_fused_decode2_NT_matmul7, (lv1169, lv1170, lv1018), out_sinfo=R.Tensor((1, n, 27648), dtype="float16"))
            lv1172 = R.call_tir(cls.fused_split3_silu1_multiply1, (lv156,), out_sinfo=R.Tensor((1, n, 13824), dtype="float16"))
            lv1173: R.Tensor((5120, 1728), dtype="uint32") = model_params[378]
            lv1174: R.Tensor((5120, 432), dtype="float16") = model_params[379]
            lv75 = R.call_tir(cls.fused_fused_decode3_fused_NT_matmul8_add1, (lv1173, lv1174, lv1172, lv74), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv785: R.Tensor((5120,), dtype="float16") = model_params[390]
            lv1029 = R.call_tir(cls.rms_norm, (lv75, lv785), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv1177: R.Tensor((15360, 640), dtype="uint32") = model_params[382]
            lv1178: R.Tensor((15360, 160), dtype="float16") = model_params[383]
            lv157 = R.call_tir(cls.fused_fused_decode_NT_matmul5, (lv1177, lv1178, lv1029), out_sinfo=R.Tensor((1, n, 15360), dtype="float16"))
            lv1032 = R.call_tir(cls.split2, (lv157,), out_sinfo=[R.Tensor((1, n, 5120), dtype="float16"), R.Tensor((1, n, 5120), dtype="float16"), R.Tensor((1, n, 5120), dtype="float16")])
            lv1033: R.Tensor((1, n, 5120), dtype="float16") = lv1032[0]
            lv1034_1 = R.call_tir(cls.reshape5, (lv1033,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv1035_1: R.Tensor((1, n, 5120), dtype="float16") = lv1032[1]
            lv1036 = R.call_tir(cls.reshape5, (lv1035_1,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv1037_1: R.Tensor((1, n, 5120), dtype="float16") = lv1032[2]
            lv1038_1 = R.call_tir(cls.reshape5, (lv1037_1,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv1039_1: R.Object = R.call_pure_packed("vm.builtin.paged_attention_kv_cache_append", lv1012_1, cls.kv_cache_transpose_append, lv1036, lv1038_1, R.prim_value(38), sinfo_args=(R.Object,))
            lv1040 = R.call_dps_packed("vm.builtin.paged_attention_kv_cache_attention", (lv1039_1, cls.attention, lv1034_1, R.prim_value(38), R.prim_value(1), R.prim_value(T.float32(1)), R.prim_value(10000)), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv1041 = R.call_tir(cls.reshape6, (lv1040,), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv1180: R.Tensor((5120, 640), dtype="uint32") = model_params[384]
            lv1181: R.Tensor((5120, 160), dtype="float16") = model_params[385]
            lv76 = R.call_tir(cls.fused_fused_decode1_fused_NT_matmul6_add1, (lv1180, lv1181, lv1041, lv75), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv790_2: R.Tensor((5120,), dtype="float16") = model_params[391]
            lv1045_1 = R.call_tir(cls.rms_norm, (lv76, lv790_2), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv1184: R.Tensor((27648, 640), dtype="uint32") = model_params[386]
            lv1185: R.Tensor((27648, 160), dtype="float16") = model_params[387]
            lv158 = R.call_tir(cls.fused_fused_decode2_NT_matmul7, (lv1184, lv1185, lv1045_1), out_sinfo=R.Tensor((1, n, 27648), dtype="float16"))
            lv1187 = R.call_tir(cls.fused_split3_silu1_multiply1, (lv158,), out_sinfo=R.Tensor((1, n, 13824), dtype="float16"))
            lv1188: R.Tensor((5120, 1728), dtype="uint32") = model_params[388]
            lv1189: R.Tensor((5120, 432), dtype="float16") = model_params[389]
            lv77 = R.call_tir(cls.fused_fused_decode3_fused_NT_matmul8_add1, (lv1188, lv1189, lv1187, lv76), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv795_2: R.Tensor((5120,), dtype="float16") = model_params[400]
            lv1056 = R.call_tir(cls.rms_norm, (lv77, lv795_2), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv1192: R.Tensor((15360, 640), dtype="uint32") = model_params[392]
            lv1193: R.Tensor((15360, 160), dtype="float16") = model_params[393]
            lv159 = R.call_tir(cls.fused_fused_decode_NT_matmul5, (lv1192, lv1193, lv1056), out_sinfo=R.Tensor((1, n, 15360), dtype="float16"))
            lv1059 = R.call_tir(cls.split2, (lv159,), out_sinfo=[R.Tensor((1, n, 5120), dtype="float16"), R.Tensor((1, n, 5120), dtype="float16"), R.Tensor((1, n, 5120), dtype="float16")])
            lv1060_1: R.Tensor((1, n, 5120), dtype="float16") = lv1059[0]
            lv1061_1 = R.call_tir(cls.reshape5, (lv1060_1,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv1062: R.Tensor((1, n, 5120), dtype="float16") = lv1059[1]
            lv1063 = R.call_tir(cls.reshape5, (lv1062,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv1064_1: R.Tensor((1, n, 5120), dtype="float16") = lv1059[2]
            lv1065_1 = R.call_tir(cls.reshape5, (lv1064_1,), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv1066: R.Object = R.call_pure_packed("vm.builtin.paged_attention_kv_cache_append", lv1039_1, cls.kv_cache_transpose_append, lv1063, lv1065_1, R.prim_value(39), sinfo_args=(R.Object,))
            lv1067_1 = R.call_dps_packed("vm.builtin.paged_attention_kv_cache_attention", (lv1066, cls.attention, lv1061_1, R.prim_value(39), R.prim_value(1), R.prim_value(T.float32(1)), R.prim_value(10000)), out_sinfo=R.Tensor((1, n, 40, 128), dtype="float16"))
            lv1068_1 = R.call_tir(cls.reshape6, (lv1067_1,), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv1195: R.Tensor((5120, 640), dtype="uint32") = model_params[394]
            lv1196: R.Tensor((5120, 160), dtype="float16") = model_params[395]
            lv78 = R.call_tir(cls.fused_fused_decode1_fused_NT_matmul6_add1, (lv1195, lv1196, lv1068_1, lv77), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv800: R.Tensor((5120,), dtype="float16") = model_params[401]
            lv1072_1 = R.call_tir(cls.rms_norm, (lv78, lv800), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv1199: R.Tensor((27648, 640), dtype="uint32") = model_params[396]
            lv1200: R.Tensor((27648, 160), dtype="float16") = model_params[397]
            lv160 = R.call_tir(cls.fused_fused_decode2_NT_matmul7, (lv1199, lv1200, lv1072_1), out_sinfo=R.Tensor((1, n, 27648), dtype="float16"))
            lv1202 = R.call_tir(cls.fused_split3_silu1_multiply1, (lv160,), out_sinfo=R.Tensor((1, n, 13824), dtype="float16"))
            lv1203: R.Tensor((5120, 1728), dtype="uint32") = model_params[398]
            lv1204: R.Tensor((5120, 432), dtype="float16") = model_params[399]
            lv79 = R.call_tir(cls.fused_fused_decode3_fused_NT_matmul8_add1, (lv1203, lv1204, lv1202, lv78), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv805_1: R.Tensor((5120,), dtype="float16") = model_params[402]
            lv1083_1 = R.call_tir(cls.rms_norm, (lv79, lv805_1), out_sinfo=R.Tensor((1, n, 5120), dtype="float16"))
            lv1084_1 = R.call_tir(cls.slice, (lv1083_1,), out_sinfo=R.Tensor((1, 1, 5120), dtype="float16"))
            lv1207: R.Tensor((vocab_size, 640), dtype="uint32") = model_params[403]
            lv1208: R.Tensor((vocab_size, 160), dtype="float16") = model_params[404]
            lv161 = R.call_tir(cls.fused_fused_decode4_fused_NT_matmul9_cast1, (lv1207, lv1208, lv1084_1), out_sinfo=R.Tensor((1, 1, vocab_size), dtype="float32"))
            gv1: R.Tuple(R.Tensor((1, 1, vocab_size), dtype="float32"), R.Object) = lv161, lv1066
            R.output(gv1)
        return gv1

    @R.function
    def softmax_with_temperature(logits: R.Tensor(("nseq", 1, "vocab_size"), dtype="float32"), temperature: R.Tensor(("nseq",), dtype="float32")) -> R.Tensor(("nseq", 1, "vocab_size"), dtype="float32"):
        nseq = T.int64()
        vocab_size = T.int64()
        R.func_attr({"tir_var_upper_bound": {"m": 4096, "n": 4096}})
        cls = Module
        with R.dataflow():
            lv2174 = R.call_tir(cls.reshape2, (temperature,), out_sinfo=R.Tensor((nseq, 1, 1), dtype="float32"))
            lv2175 = R.call_tir(cls.divide, (logits, lv2174), out_sinfo=R.Tensor((nseq, 1, vocab_size), dtype="float32"))
            lv2176 = R.call_tir(cls.softmax, (lv2175,), out_sinfo=R.Tensor((nseq, 1, vocab_size), dtype="float32"))
            gv4: R.Tensor((nseq, 1, vocab_size), dtype="float32") = lv2176
            R.output(gv4)
        return gv4

from dlight_bench import DlightBench
from tvm.script import tir as T

@T.prim_func(private=True)
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

@T.prim_func(private=True)
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

@T.prim_func(private=True)
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

@T.prim_func(private=True)
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

@T.prim_func(private=True)
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

@T.prim_func(private=True)
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

@T.prim_func(private=True)
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

@T.prim_func(private=True)
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

@T.prim_func(private=True)
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

@T.prim_func(private=True)
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

@T.prim_func(private=True)
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

@T.prim_func(private=True)
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

@T.prim_func(private=True)
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

@T.prim_func(private=True)
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
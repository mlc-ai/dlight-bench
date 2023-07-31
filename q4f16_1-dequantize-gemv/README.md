## NK q4f16_1 dequantize gemv

We specifically consider the following workload:

```py
@T.prim_func
def NK_degemv(
    A_q: T.Buffer((T.int64(N), T.int64(K // 8)), "uint32"), 
    A_scale: T.Buffer((T.int64(N), T.int64(K // 32)), "float16"), 
    V: T.Buffer((T.int64(K)), "float16"), 
    C: T.Buffer((T.int64(N)), "float16")
):
    T.func_attr({"tir.noalias": T.bool(True)})
    # with T.block("root"):
    A = T.alloc_buffer((T.int64(N), T.int64(K)), "float16")
    for i, j in T.grid(T.int64(N), T.int64(K)):
        with T.block("decode"):
            v_i, v_j = T.axis.remap("SS", [i, j])
            T.reads(A_q[v_i, v_j // T.int64(8)], A_scale[v_i, v_j // T.int64(32)])
            T.writes(A[v_i, v_j])
            A[v_i, v_j] = (T.Cast("float16", T.bitwise_and(T.shift_right(A_q[v_i, v_j // T.int64(8)], T.Cast("uint32", v_j % T.int64(8)) * T.uint32(4)), T.uint32(15))) - T.float16(7)) * A_scale[v_i, v_j // T.int64(32)]
    for i2, k in T.grid(T.int64(N), T.int64(K)):
        with T.block("gemv"):
            v_i2, v_k = T.axis.remap("SR", [i2, k])
            T.reads(V[v_k], A[v_i2, v_k])
            T.writes(C[v_i2])
            with T.init():
                C[v_i2] = T.float16(0)
            C[v_i2] = C[v_i2] + V[v_k] * A[v_i2, v_k]

```

Generically, we consider the following schedule space:

* Each thread block handles several contiguous rows
* Inside each thread block, we use `TS x TR` thread layout, where `TS` spreads over N, and `TR` spreads over K to do split reduction.
* Suppose each thread handles `TILE_S` contiguous rows, and `TILE_R` contiguous columns at a time. Since it's q4f16_1 dequantized gemm, we force `TILE_R` to be multiple of 8. Hence, each thread handles `TILE_S * TILE_R` elements.
* Suppose we read `VEC_LOAD` elements a time during load, and handle `VEC_C` elements a time during computation, then comes 2 orthogonal schedule decisions
  
  * do vector load over N or K
  * do vector computation over or K

which forms 4 different schedule templates, and for each template, we can freely configure

* TS = threadIdx.x, TR = threadIdx.y or TS = threadIdx.y, TR = threadIdx.x
* Value of `TS`, `TR`, `TILE_S`, `TILE_R`, `VEC_LOAD`, `VEC_C`
* More over, we take layout convertion into account, whose transformation (N, K) -> (N, K, n, k) can be decided by two numbers `n`, `k`.

In all, it forms the schedule space we consider.

We also set some heurstics to prune out some schedules

- N % `n` == 0, K % `k` == 0
- We don't consider `TILE_S > n` cases.
- In order to achieve coalesced read of contiguous threads

  - If threadIdx.x is lay over the same axis with `VEC_LOAD`, tile size of that dimension has to be exactly `VEC_LOAD` (over N) or `VEC_LOAD * 8` (over K)
  - If threadIdx.x is lay over a different axis with `VEC_LOAD`, tile size of threadIdx.x dimension has to be 1.

There are some other schedule decisions:

- whether to load the vector into shared memory
- Unroll factor

## Benchmark

We measure the numbers with L2 cache flushed, n=1, repeat=100, cache_flush_bytes=256*10**6
- shape: 12288 x 4096

|             | dlight (ms) | ours (ms) |
|:-----------:|:-----------:|:-----------:|
| NVIDIA 4090 |     0.0476        |   0.0470          |
| NVIDIA 2070 |      0.0817       |    0.0780         |
| Radeon RX 7900 XTX|  0.162 | 0.079  |
|    M1 pro   |      0.443       |     0.388        |
|  Adreno 740 |     3.84    |   0.59          |

- shape: 15360 x 5120

|             | dlight (ms) | ours (ms) |
|:-------------:|:-------------:|:-------------:|
| NVIDIA 4090 |     0.071        |    0.070         |
| NVIDIA 2070 |     0.164        |      0.117       |
| Radeon RX 7900 XTX| 0.130 | 0.115 |
| M1 pro      |    0.518         |      0.481       |
| Adreno 740  |    5.84         |    1.02         |

Notes:

Overall, to get the above results in a reasonable time
- We don't consider tile `k` for all benchmarks
- We only consider tile `n` for Adreno
- Only CUDA and Metal will do shared load of vector
- CUDA and Metal set unroll factors to 256, while Adreno sets to 8
- Only `VEC_LOAD` over N and `VEC_C` over compute (schedule 1) is considered

**TODO**: Mali

## Discussions

- Does layout help?

For Adreno, the answer is yes. 
```
====
schedule 1: TAG_S=threadIdx.x	TAG_R=threadIdx.y	vec_load=1	vec_c=4	tile_s=1	tile_r=8	tr=1	ts=64	n=64	k=1
Time (ms): 0.595679	Total Bytes (MB): 27.031250	Memory (GB/s): 45.378908
====
schedule 1: TAG_S=threadIdx.x	TAG_R=threadIdx.y	vec_load=1	vec_c=4	tile_s=1	tile_r=8	tr=1	ts=64	n=16	k=1
Time (ms): 1.110792	Total Bytes (MB): 27.031250	Memory (GB/s): 24.335121
====
schedule 1: TAG_S=threadIdx.x	TAG_R=threadIdx.y	vec_load=1	vec_c=4	tile_s=1	tile_r=8	tr=1	ts=64	n=8	k=1
Time (ms): 1.241569	Total Bytes (MB): 27.031250	Memory (GB/s): 21.771842
====
schedule 1: TAG_S=threadIdx.x	TAG_R=threadIdx.y	vec_load=1	vec_c=4	tile_s=1	tile_r=8	tr=1	ts=64	n=4	k=1
Time (ms): 4.083517	Total Bytes (MB): 27.031250	Memory (GB/s): 6.619599
```

For the above schedule, changing `n` from 1 to 64 constantly increases the performance by a lot. But the schedule is simply laying tx over N and do normal reduction (tr = 1), so this is likely to suggest q4f16_0 is better for this case

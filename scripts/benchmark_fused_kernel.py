"""Micro-benchmark: FP16 SDPA vs drop-in TurboQuant cache path vs Triton fused.

Not a full Llama end-to-end benchmark — that lands with Phase 9c when the HF
attention hook is wired up. This measures the per-layer attention step at
Llama-3.1-8B-ish shapes so we can see whether the fused kernel beats the
"dequantize-then-SDPA" path on HBM-bound decode.
"""

from __future__ import annotations

import argparse
import math
import time

import torch
import torch.nn.functional as F

from turboquant.integrations.hf_cache import _BatchedKeyStore, _BatchedValueStore
from turboquant.kernels.decompress_attn import fused_decompress_attention_triton
from turboquant.quantizer_mse_torch import BatchedTurboQuantMSETorch
from turboquant.quantizer_prod_torch import BatchedTurboQuantProdTorch


def build_state(B, H_kv, S, D, key_bits, value_bits, device):
    torch.manual_seed(0)
    k = torch.randn(B, H_kv, S, D, device=device, dtype=torch.float16)
    v = torch.randn(B, H_kv, S, D, device=device, dtype=torch.float16)
    kq = BatchedTurboQuantProdTorch(
        D, key_bits, H_kv, list(range(H_kv)), [100 + h for h in range(H_kv)], device
    )
    vq = BatchedTurboQuantMSETorch(
        D, value_bits, H_kv, [500 + h for h in range(H_kv)], device
    )
    kb = k.permute(1, 0, 2, 3).reshape(H_kv, B * S, D)
    vb = v.permute(1, 0, 2, 3).reshape(H_kv, B * S, D)
    q_k, nk = kq.quantize_with_norm(kb)
    q_v, nv = vq.quantize_with_norm(vb)
    ks = _BatchedKeyStore(key_bits=key_bits, head_dim=D)
    vs = _BatchedValueStore(value_bits=value_bits, head_dim=D)
    ks.append(q_k, nk, B, S)
    vs.append(q_v, nv, B, S)

    k_deq = kq.dequantize_with_norm(q_k, nk).reshape(H_kv, B, S, D).permute(1, 0, 2, 3)
    v_deq = vq.dequantize_with_norm(q_v, nv).reshape(H_kv, B, S, D).permute(1, 0, 2, 3)
    return ks, vs, kq, vq, k_deq.to(torch.float16), v_deq.to(torch.float16)


def timed(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000.0  # ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--batch", type=int, default=1)
    args = ap.parse_args()

    device = "cuda"
    B, H_kv, gqa, D = args.batch, 8, 4, 128
    H_q = H_kv * gqa
    S = args.seq

    ks, vs, kq, vq, k_deq, v_deq = build_state(B, H_kv, S, D, args.bits, args.bits, device)
    q_fp16 = torch.randn(B, H_q, 1, D, device=device, dtype=torch.float16)

    k_rep = k_deq.repeat_interleave(gqa, dim=1)
    v_rep = v_deq.repeat_interleave(gqa, dim=1)

    # 1. FP16 SDPA on already-dense KV (best-case FP16 baseline)
    def sdpa():
        return F.scaled_dot_product_attention(q_fp16, k_rep, v_rep)

    # 2. Drop-in path: dequantize full KV each step + SDPA
    def dropin():
        k_full = kq.dequantize_with_norm(*ks.to_quantized_dict()).reshape(
            H_kv, B, S, D
        ).permute(1, 0, 2, 3).to(torch.float16)
        v_full = vq.dequantize_with_norm(*vs.to_quantized_dict()).reshape(
            H_kv, B, S, D
        ).permute(1, 0, 2, 3).to(torch.float16)
        k_full = k_full.repeat_interleave(gqa, dim=1)
        v_full = v_full.repeat_interleave(gqa, dim=1)
        return F.scaled_dot_product_attention(q_fp16, k_full, v_full)

    # 3. Fused Triton
    def fused():
        return fused_decompress_attention_triton(q_fp16, ks, vs, kq, vq, block_n=64)

    t_sdpa = timed(sdpa)
    t_drop = timed(dropin)
    t_fused = timed(fused)

    print(f"Shape: B={B}, H_q={H_q}, H_kv={H_kv}, S={S}, D={D}, b={args.bits}")
    print(f"  FP16 SDPA (dense):   {t_sdpa:.3f} ms  (1.00x baseline)")
    print(f"  Drop-in dequant+SDPA:{t_drop:.3f} ms  ({t_sdpa/t_drop:.2f}x)")
    print(f"  Triton fused:        {t_fused:.3f} ms  ({t_sdpa/t_fused:.2f}x)")


if __name__ == "__main__":
    main()

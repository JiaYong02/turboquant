"""Measure KV-cache compression ratio across bit-widths and outlier settings.

Reports bytes/token for keys and values at several configurations,
including single-bucket (``outlier_frac=0``) and split-bucket
(``outlier_frac>0``) modes. Compares against FP16 baseline (2*D bytes
per token for K + 2*D for V = 4*D bytes per head per token).

Usage::

    python scripts/measure_compression.py
    python scripts/measure_compression.py --bits 4 --head-dim 128
    python scripts/measure_compression.py --bits-avg 2.5 --outlier-frac 0.25

By default, sweeps a table of (bits, outlier_frac) combinations. The
``--bits-avg`` / ``--outlier-frac`` flags constrain to a single row.
"""

from __future__ import annotations

import argparse
import math

import torch

from turboquant.bit_packing import packed_dim
from turboquant.integrations.hf_cache import TurboQuantCacheLayer


def _measure(
    head_dim: int,
    num_heads: int,
    seq: int,
    key_bits: int,
    value_bits: int,
    outlier_frac: float,
    key_bits_hi: int | None,
    key_bits_lo: int | None,
    value_bits_hi: int | None,
    value_bits_lo: int | None,
) -> dict:
    """Materialize a cache layer, write `seq` tokens, and measure storage."""
    n_out = int(round(outlier_frac * head_dim)) if outlier_frac > 0 else 0
    kwargs = dict(
        layer_idx=0,
        num_kv_heads=num_heads,
        head_dim=head_dim,
        key_bits=key_bits,
        value_bits=value_bits,
        seed=42,
        device="cpu",
    )
    if n_out > 0:
        kwargs.update(
            n_out=n_out,
            key_bits_hi=key_bits_hi,
            key_bits_lo=key_bits_lo,
            value_bits_hi=value_bits_hi,
            value_bits_lo=value_bits_lo,
        )
    layer = TurboQuantCacheLayer(**kwargs)

    torch.manual_seed(0)
    k = torch.randn(1, num_heads, seq, head_dim)
    v = torch.randn(1, num_heads, seq, head_dim)
    layer.update(k, v)

    ks = layer._key_store
    vs = layer._value_store

    key_bytes = 0
    val_bytes = 0
    for attr in ("qjl_packed", "gamma", "norms", "mse_packed",
                 "mse_packed_hi", "mse_packed_lo"):
        t = getattr(ks, attr, None)
        if t is not None:
            key_bytes += t.nbytes
    for attr in ("idx_packed", "norms", "idx_packed_hi", "idx_packed_lo"):
        t = getattr(vs, attr, None)
        if t is not None:
            val_bytes += t.nbytes

    total = key_bytes + val_bytes
    # K + V each stored fp16: 2 tensors * H heads * S tokens * D dims * 2 bytes.
    fp16 = 2 * num_heads * seq * head_dim * 2
    return {
        "key_bytes": key_bytes,
        "val_bytes": val_bytes,
        "total": total,
        "fp16": fp16,
        "ratio": fp16 / total,
        "bytes_per_token_per_head": total / (num_heads * seq),
    }


def _theoretical_bytes_per_token(
    head_dim: int,
    key_bits: int,
    value_bits: int,
    outlier_frac: float,
    key_bits_hi: int,
    key_bits_lo: int,
    value_bits_hi: int,
    value_bits_lo: int,
) -> tuple[int, int]:
    """Analytical K and V bytes per token per head under tight packing."""
    n_out = int(round(outlier_frac * head_dim)) if outlier_frac > 0 else 0
    # K: 1-bit QJL over all D + fp16 gamma + fp16 norm.
    qjl = packed_dim(head_dim, 1)
    norm_k = 2
    gamma = 2
    if n_out > 0:
        mse_hi = packed_dim(n_out, key_bits_hi - 1)
        mse_lo = packed_dim(head_dim - n_out, key_bits_lo - 1)
        k_bytes = qjl + gamma + norm_k + mse_hi + mse_lo
    elif key_bits > 1:
        k_bytes = qjl + gamma + norm_k + packed_dim(head_dim, key_bits - 1)
    else:
        k_bytes = qjl + gamma + norm_k
    # V: full-bit MSE indices + fp16 norm.
    norm_v = 2
    if n_out > 0:
        v_hi = packed_dim(n_out, value_bits_hi)
        v_lo = packed_dim(head_dim - n_out, value_bits_lo)
        v_bytes = v_hi + v_lo + norm_v
    else:
        v_bytes = packed_dim(head_dim, value_bits) + norm_v
    return k_bytes, v_bytes


def _resolve_split(
    base_bits: int,
    hi: int | None,
    lo: int | None,
) -> tuple[int, int]:
    """Paper defaults: hi = b + 1, lo = b - 1."""
    return (hi if hi is not None else base_bits + 1,
            lo if lo is not None else base_bits - 1)


def _row(head_dim: int, num_heads: int, seq: int,
         kb: int, vb: int, frac: float,
         khi: int | None, klo: int | None,
         vhi: int | None, vlo: int | None) -> None:
    n_out = int(round(frac * head_dim)) if frac > 0 else 0
    if frac > 0:
        khi_r, klo_r = _resolve_split(kb, khi, klo)
        vhi_r, vlo_r = _resolve_split(vb, vhi, vlo)
        k_avg = (frac * khi_r + (1 - frac) * klo_r)
        v_avg = (frac * vhi_r + (1 - frac) * vlo_r)
        mode = f"split frac={frac:.2f} k=({khi_r},{klo_r}) v=({vhi_r},{vlo_r})"
    else:
        khi_r = klo_r = vhi_r = vlo_r = 0
        k_avg, v_avg = float(kb), float(vb)
        mode = "single"

    meas = _measure(
        head_dim=head_dim, num_heads=num_heads, seq=seq,
        key_bits=kb, value_bits=vb, outlier_frac=frac,
        key_bits_hi=khi_r or None, key_bits_lo=klo_r or None,
        value_bits_hi=vhi_r or None, value_bits_lo=vlo_r or None,
    )
    th_k, th_v = _theoretical_bytes_per_token(
        head_dim, kb, vb, frac, khi_r, klo_r, vhi_r, vlo_r,
    )
    th_total = th_k + th_v
    th_ratio = (4 * head_dim) / th_total

    print(
        f"  kb={kb} vb={vb} {mode:<40s} "
        f"k_avg={k_avg:.2f} v_avg={v_avg:.2f} | "
        f"B/tok/head={meas['bytes_per_token_per_head']:6.2f} "
        f"measured={meas['ratio']:5.2f}x "
        f"theory={th_ratio:5.2f}x (k={th_k}B v={th_v}B)"
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--head-dim", type=int, default=128)
    p.add_argument("--num-heads", type=int, default=8)
    p.add_argument("--seq", type=int, default=1024)
    p.add_argument("--bits", type=int, help="Single-mode bit-width")
    p.add_argument("--bits-avg", type=float,
                   help="Split-mode average bits (implies --outlier-frac)")
    p.add_argument("--outlier-frac", type=float, default=0.0)
    p.add_argument("--key-bits-hi", type=int)
    p.add_argument("--key-bits-lo", type=int)
    p.add_argument("--value-bits-hi", type=int)
    p.add_argument("--value-bits-lo", type=int)
    args = p.parse_args()

    D, H, S = args.head_dim, args.num_heads, args.seq
    fp16_btt = 4 * D  # bytes per token per head (K+V fp16)
    print(f"Config: D={D}  H={H}  S={S}  FP16 baseline = {fp16_btt} B/tok/head")

    if args.bits is not None and args.outlier_frac == 0.0:
        _row(D, H, S, args.bits, args.bits, 0.0,
             None, None, None, None)
        return

    if args.bits_avg is not None:
        # Infer base b from avg + frac: avg = frac*(b+1) + (1-frac)*(b-1)
        # = b + frac - (1-frac) = b + 2*frac - 1 → b = avg - 2*frac + 1.
        frac = args.outlier_frac or 0.25
        base = int(round(args.bits_avg - 2 * frac + 1))
        _row(D, H, S, base, base, frac,
             args.key_bits_hi, args.key_bits_lo,
             args.value_bits_hi, args.value_bits_lo)
        return

    # Default: sweep a small table.
    print("\nSingle-bucket:")
    for b in (4, 3, 2):
        _row(D, H, S, b, b, 0.0, None, None, None, None)

    print("\nSplit-bucket (paper defaults hi=b+1, lo=b-1):")
    for frac in (0.25,):
        for b in (4, 3):
            _row(D, H, S, b, b, frac, None, None, None, None)


if __name__ == "__main__":
    main()

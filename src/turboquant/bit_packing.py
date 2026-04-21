"""Bit packing utilities for TurboQuant compressed storage.

Packs b-bit integer indices and 1-bit sign values into compact uint8 byte
arrays using a bit-tight layout. Values are laid end-to-end across bytes
MSB-first; a single value may span two adjacent bytes.

Supported bit widths: 1–6. Storage cost is always `ceil(D * n_bits / 8)`
bytes, regardless of width. This matches the TurboQuant paper's
compression math for b ∈ {2, 3, 4, 5} and the outlier-channel split
configurations in Phase 14.

All functions operate on the last tensor dimension and preserve all leading
dimensions, so they work correctly on (..., D) tensors without reshaping.
"""

from __future__ import annotations

import math

import torch

_SUPPORTED_BITS = (1, 2, 3, 4, 5, 6)


def packed_dim(d: int, n_bits: int) -> int:
    """Bit-tight packed size: ceil(d * n_bits / 8)."""
    return math.ceil(d * n_bits / 8)


def _coord_layout(D: int, n_bits: int, device: torch.device) -> tuple[
    torch.Tensor, torch.Tensor, torch.Tensor
]:
    """Per-coord `(byte_lo, byte_hi, shift_down)` arrays in the 16-bit-window scheme.

    Each D-length coord owns bits `[i*n_bits, (i+1)*n_bits)` in the flat
    MSB-first bit stream. Within a 16-bit window `(byte_lo << 8) | byte_hi`
    the value sits at LSB offset `shift_down = 16 - n_bits - start_in_lo`.
    """
    i = torch.arange(D, device=device, dtype=torch.int64)
    bit_off = i * n_bits
    byte_lo = bit_off // 8
    start_in_lo = bit_off % 8
    shift_down = (16 - n_bits - start_in_lo).to(torch.int32)
    return byte_lo, (byte_lo + 1), shift_down


def pack_bits(values: torch.Tensor, n_bits: int) -> torch.Tensor:
    """Pack uint8 values (range 0..2^n_bits-1) along the last dimension.

    Bit-tight layout: value `i` occupies bits [i*n_bits, (i+1)*n_bits) in
    the flat MSB-first bit stream. Storage is `ceil(D*n_bits/8)` bytes.
    """
    if n_bits not in _SUPPORTED_BITS:
        raise ValueError(f"n_bits must be one of {_SUPPORTED_BITS}, got {n_bits}")

    D = values.shape[-1]
    prefix = values.shape[:-1]
    device = values.device
    if D == 0:
        return torch.empty(*prefix, 0, dtype=torch.uint8, device=device)

    packed_D = packed_dim(D, n_bits)
    byte_lo, byte_hi, shift_down = _coord_layout(D, n_bits, device)

    v = values.to(torch.int32)
    # Place each value at its 16-bit-window position, then split to two bytes.
    word = v << shift_down
    lo_contrib = (word >> 8) & 0xFF
    hi_contrib = word & 0xFF

    # One trailing pad byte so `byte_lo + 1` is always writeable.
    buf_shape = prefix + (packed_D + 1,)
    buf = torch.zeros(buf_shape, dtype=torch.int32, device=device)

    idx_shape = (1,) * len(prefix) + (D,)
    lo_idx = byte_lo.view(idx_shape).expand(prefix + (D,))
    hi_idx = byte_hi.view(idx_shape).expand(prefix + (D,))

    buf.scatter_add_(-1, lo_idx, lo_contrib)
    buf.scatter_add_(-1, hi_idx, hi_contrib)

    return (buf[..., :packed_D] & 0xFF).to(torch.uint8).contiguous()


def unpack_bits(packed: torch.Tensor, n_bits: int, n_values: int) -> torch.Tensor:
    """Unpack bit-tight uint8 stream to uint8 values of length `n_values`."""
    if n_bits not in _SUPPORTED_BITS:
        raise ValueError(f"n_bits must be one of {_SUPPORTED_BITS}, got {n_bits}")
    D = n_values
    prefix = packed.shape[:-1]
    device = packed.device
    if D == 0:
        return torch.empty(*prefix, 0, dtype=torch.uint8, device=device)

    byte_lo, byte_hi, shift_down = _coord_layout(D, n_bits, device)
    mask = (1 << n_bits) - 1

    packed32 = packed.to(torch.int32)
    # Pad one trailing zero byte so `byte_hi` is always a valid index.
    padded = torch.nn.functional.pad(packed32, (0, 1))

    idx_shape = (1,) * len(prefix) + (D,)
    lo_idx = byte_lo.view(idx_shape).expand(prefix + (D,))
    hi_idx = byte_hi.view(idx_shape).expand(prefix + (D,))

    lo_bytes = torch.gather(padded, -1, lo_idx)
    hi_bytes = torch.gather(padded, -1, hi_idx)

    word = (lo_bytes << 8) | hi_bytes
    values = (word >> shift_down) & mask
    return values.to(torch.uint8).contiguous()


def pack_signs(signs: torch.Tensor) -> torch.Tensor:
    """Pack int8 {-1, +1} sign tensor into 1-bit uint8 (8 signs per byte).

    Encoding: +1 -> bit 1, -1 -> bit 0. MSB-first within each byte.
    """
    bits = (signs > 0).to(torch.uint8)
    return pack_bits(bits, n_bits=1)


def unpack_signs(packed: torch.Tensor, n_values: int) -> torch.Tensor:
    """Unpack 1-bit-packed tensor back to int8 {-1, +1} values."""
    bits = unpack_bits(packed, n_bits=1, n_values=n_values)
    return (bits.to(torch.int8) * 2 - 1)

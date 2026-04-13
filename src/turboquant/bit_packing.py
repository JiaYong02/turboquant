"""Bit packing utilities for TurboQuant compressed storage.

Packs b-bit integer indices and 1-bit sign values into compact uint8 byte
arrays, reducing storage by floor(8/n_bits)x compared to one byte per value.

Supported bit widths: 1, 2, 3, 4.
  1-bit: 8 values per byte (0% waste)
  2-bit: 4 values per byte (0% waste)
  3-bit: 2 values per byte (25% waste — non-power-of-two)
  4-bit: 2 values per byte (0% waste)

All functions operate on the last tensor dimension and preserve all leading
dimensions, so they work correctly on (..., D) tensors without reshaping.
"""

from __future__ import annotations

import math

import torch


def _vpb(n_bits: int) -> int:
    """Values per byte for a given bit width."""
    return 8 // n_bits


def packed_dim(d: int, n_bits: int) -> int:
    """Number of uint8 bytes needed to store d values at n_bits each."""
    return math.ceil(d / _vpb(n_bits))


def pack_bits(values: torch.Tensor, n_bits: int) -> torch.Tensor:
    """Pack uint8 values (range 0..2^n_bits-1) along the last dimension.

    Args:
        values: Tensor of dtype uint8 and shape (..., D).
        n_bits: Bits per value; must be 1, 2, 3, or 4.

    Returns:
        Packed uint8 tensor of shape (..., ceil(D / vpb)) where
        vpb = 8 // n_bits. For n_bits=3, each byte holds 2 values
        using 6 bits (2 bits wasted per byte).
    """
    if n_bits not in (1, 2, 3, 4):
        raise ValueError(f"n_bits must be 1-4, got {n_bits}")
    D = values.shape[-1]
    prefix = values.shape[:-1]
    device = values.device
    values = values.to(torch.uint8)

    vpb = _vpb(n_bits)

    # Pad last dim to a multiple of vpb (zero-pad; padding values don't appear
    # in the unpacked output because unpack truncates to n_values).
    pad = (-D) % vpb
    if pad:
        pad_t = torch.zeros(*prefix, pad, dtype=torch.uint8, device=device)
        values = torch.cat([values, pad_t], dim=-1)

    # Reshape: (..., packed_D, vpb) — each slice of vpb values → one byte.
    packed_D = (D + pad) // vpb
    values = values.reshape(*prefix, packed_D, vpb)

    packed = torch.zeros(*prefix, packed_D, dtype=torch.uint8, device=device)
    for i in range(vpb):
        shift = (vpb - 1 - i) * n_bits
        # Upcast to int32 before shift to avoid undefined behaviour on small types,
        # then mask to 8 bits and cast back to uint8.
        packed = packed | (
            (values[..., i].to(torch.int32) << shift) & 0xFF
        ).to(torch.uint8)

    return packed


def unpack_bits(packed: torch.Tensor, n_bits: int, n_values: int) -> torch.Tensor:
    """Unpack bit-packed uint8 tensor back to uint8 values.

    Args:
        packed: uint8 tensor of shape (..., packed_D).
        n_bits: Bits per value; must be 1, 2, 3, or 4.
        n_values: Number of values to recover (original D before packing).

    Returns:
        uint8 tensor of shape (..., n_values).
    """
    if n_bits not in (1, 2, 3, 4):
        raise ValueError(f"n_bits must be 1-4, got {n_bits}")
    prefix = packed.shape[:-1]
    device = packed.device
    vpb = _vpb(n_bits)
    mask = torch.tensor((1 << n_bits) - 1, dtype=torch.uint8, device=device)

    total = packed.shape[-1] * vpb
    result = torch.zeros(*prefix, total, dtype=torch.uint8, device=device)

    for i in range(vpb):
        shift = (vpb - 1 - i) * n_bits
        # Extract position i from every byte simultaneously.
        # result[..., i::vpb] selects index i+j*vpb for j=0,1,...,packed_D-1,
        # which corresponds to position i within byte j.
        result[..., i::vpb] = (
            packed.to(torch.int32) >> shift
        ).to(torch.uint8) & mask

    return result[..., :n_values]


def pack_signs(signs: torch.Tensor) -> torch.Tensor:
    """Pack int8 {-1, +1} sign tensor into 1-bit uint8 (8 signs per byte).

    Encoding: +1 → bit 1, -1 → bit 0.

    Args:
        signs: Tensor of dtype int8 and shape (..., D), values in {-1, +1}.

    Returns:
        Packed uint8 tensor of shape (..., ceil(D / 8)).
    """
    bits = (signs > 0).to(torch.uint8)
    return pack_bits(bits, n_bits=1)


def unpack_signs(packed: torch.Tensor, n_values: int) -> torch.Tensor:
    """Unpack 1-bit-packed tensor back to int8 {-1, +1} values.

    Args:
        packed: uint8 tensor of shape (..., ceil(D/8)).
        n_values: Original number of signs D.

    Returns:
        int8 tensor of shape (..., n_values), values in {-1, +1}.
    """
    bits = unpack_bits(packed, n_bits=1, n_values=n_values)
    # bit 0 → -1, bit 1 → +1
    return (bits.to(torch.int8) * 2 - 1)

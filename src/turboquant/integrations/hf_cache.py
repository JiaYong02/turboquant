"""HuggingFace transformers KV cache integration.

Provides TurboQuantCache, a drop-in replacement for DynamicCache that
compresses keys with TurboQuantProd and values with TurboQuantMSE.

Storage layout
--------------
All heads for a layer are stored together in [H, B, S, ...] tensors.
Indices are bit-tight packed (see ``bit_packing.py``) so actual storage is
``ceil(D * n_bits / 8)`` bytes per vector, independent of ``n_bits``:

  Keys (TurboQuantProd at b bits):
    mse_packed : uint8 [H, B, S, ceil(D*(b-1)/8)]  — (b-1)-bit MSE indices
    qjl_packed : uint8 [H, B, S, ceil(D/8)]        — 1-bit QJL signs
    gamma      : f16   [H, B, S]                   — residual norms
    norms      : f16   [H, B, S]                   — input norms

  Values (TurboQuantMSE at b bits):
    idx_packed : uint8 [H, B, S, ceil(D*b/8)]      — b-bit MSE indices
    norms      : f16   [H, B, S]                   — input norms

In split-bucket mode (``n_out > 0``), ``mse_packed`` / ``idx_packed`` are
replaced by ``_hi`` / ``_lo`` pairs sized to the two buckets.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Any, Optional

import torch
from transformers.cache_utils import Cache, CacheLayerMixin
from transformers.configuration_utils import PretrainedConfig

from ..bit_packing import pack_bits, pack_signs, packed_dim, unpack_bits, unpack_signs
from ..quantizer_mse_torch import BatchedTurboQuantMSETorch
from ..quantizer_prod_torch import BatchedTurboQuantProdTorch


def _env_flag(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() not in ("0", "", "false", "no", "off")


@dataclass
class _CapturedGraph:
    """Long-lived state for one captured decode subgraph (per S_bucket)."""
    graph: Any                      # torch.cuda.CUDAGraph
    out_captured: torch.Tensor      # fp32 [B, H_q, 1, D], filled by replay
    s_total_static: torch.Tensor    # int32 0-d, update in-place before replay
    scale: float
    split_size: int
    num_splits_max: int


# ---------------------------------------------------------------------------
# Internal storage helpers
# ---------------------------------------------------------------------------


def _cat_or_init(
    existing: torch.Tensor | None, new: torch.Tensor, dim: int
) -> torch.Tensor:
    if existing is None:
        return new
    return torch.cat([existing, new], dim=dim)


class _BatchedKeyStore:
    """Compressed key storage for all H attention heads in a layer.

    Stores bit-packed tensors of shape [H, B, S, packed_D] to minimise VRAM.
    Sequence dimension (dim=2) grows by concatenation at each decode step.
    """

    def __init__(
        self,
        key_bits: int,
        head_dim: int,
        *,
        n_out: int = 0,
        bits_hi: int | None = None,
        bits_lo: int | None = None,
    ) -> None:
        if n_out < 0 or n_out > head_dim:
            raise ValueError(
                f"n_out must be in [0, head_dim]; got n_out={n_out}, "
                f"head_dim={head_dim}"
            )
        if n_out > 0:
            if bits_hi is None or bits_lo is None:
                raise ValueError("bits_hi and bits_lo are required when n_out > 0")
            # K-stage MSE uses (bits - 1); QJL supplies the final bit, so each
            # bucket needs at least 2 total bits to leave 1 bit for MSE.
            if bits_hi < 2 or bits_lo < 2:
                raise ValueError(
                    "bits_hi and bits_lo must be >= 2 for the K store "
                    f"(one bit goes to QJL); got bits_hi={bits_hi}, "
                    f"bits_lo={bits_lo}"
                )

        self.key_bits = key_bits
        self.head_dim = head_dim
        self.n_out = n_out
        self.bits_hi = bits_hi
        self.bits_lo = bits_lo
        self.qjl_packed: torch.Tensor | None = None   # uint8 [H, B, S, ceil(D/8)]
        self.gamma: torch.Tensor | None = None         # f32   [H, B, S]
        self.norms: torch.Tensor | None = None         # f32   [H, B, S]
        # Single-bucket mode (n_out == 0).
        self.mse_packed: torch.Tensor | None = None    # uint8 [H, B, S, packed_mse]
        # Split-bucket mode (n_out > 0). K-stage MSE uses (bits - 1) per bucket
        # because QJL supplies the final bit.
        self.mse_packed_hi: torch.Tensor | None = None
        self.mse_packed_lo: torch.Tensor | None = None

        # Slab-mode storage (Phase 11b, CUDA-graph capture). When active, the
        # packed tensors above are fixed-size [H, B, max_seq_len, ...] slabs and
        # `_seq_len` tracks the valid prefix. Pointer identity is preserved
        # across in-place appends so a captured graph can replay safely.
        self._max_seq_len: int | None = None
        self._slab_B: int | None = None
        self._seq_len: int = 0
        self._slab_active: bool = False
        self._overflow_warned: bool = False

    # ---- slab lifecycle --------------------------------------------------

    @property
    def slab_active(self) -> bool:
        return self._slab_active

    def preallocate(
        self,
        max_seq_len: int,
        B: int,
        H: int,
        device: torch.device,
    ) -> None:
        """Pre-allocate fixed-size packed slabs.

        Why: a captured CUDA graph bakes tensor pointers at capture time, so the
        growing-cat path invalidates every replay. A slab keeps the same
        storage for the life of the cache; in-place appends advance `_seq_len`
        but never reassign the tensors.
        """
        D = self.head_dim
        self._max_seq_len = max_seq_len
        self._slab_B = B
        self._seq_len = 0
        self.qjl_packed = torch.empty(
            (H, B, max_seq_len, packed_dim(D, 1)),
            dtype=torch.uint8,
            device=device,
        )
        # fp16 storage (Phase A): scalars per position, kernel upcasts on load.
        # Saves ~6 B/position/head of metadata vs fp32.
        self.gamma = torch.empty(
            (H, B, max_seq_len), dtype=torch.float16, device=device
        )
        self.norms = torch.empty(
            (H, B, max_seq_len), dtype=torch.float16, device=device
        )
        if self.n_out > 0:
            self.mse_packed = None
            self.mse_packed_hi = torch.empty(
                (H, B, max_seq_len, packed_dim(self.n_out, self.bits_hi - 1)),
                dtype=torch.uint8,
                device=device,
            )
            self.mse_packed_lo = torch.empty(
                (H, B, max_seq_len, packed_dim(D - self.n_out, self.bits_lo - 1)),
                dtype=torch.uint8,
                device=device,
            )
        else:
            mse_bits = self.key_bits - 1
            if mse_bits > 0:
                self.mse_packed = torch.empty(
                    (H, B, max_seq_len, packed_dim(D, mse_bits)),
                    dtype=torch.uint8,
                    device=device,
                )
            else:
                self.mse_packed = None
        self._slab_active = True
        self._overflow_warned = False

    def _demote_slab(self) -> None:
        """Drop to growable storage, keeping only the valid prefix.

        Called on slab overflow. Caller is expected to invalidate any captured
        graph relying on the pre-demote pointers.
        """
        if not self._slab_active:
            return
        S = self._seq_len
        self.qjl_packed = self.qjl_packed[:, :, :S, :].contiguous()
        self.gamma = self.gamma[:, :, :S].contiguous()
        self.norms = self.norms[:, :, :S].contiguous()
        if self.mse_packed is not None:
            self.mse_packed = self.mse_packed[:, :, :S, :].contiguous()
        if self.mse_packed_hi is not None:
            self.mse_packed_hi = self.mse_packed_hi[:, :, :S, :].contiguous()
        if self.mse_packed_lo is not None:
            self.mse_packed_lo = self.mse_packed_lo[:, :, :S, :].contiguous()
        self._slab_active = False
        self._max_seq_len = None
        self._slab_B = None

    def seq_length(self) -> int:
        # qjl_packed is always populated (QJL stays 1-bit full-D regardless of
        # split-bucket mode), so it's the canonical seq-length witness.
        if self._slab_active:
            return self._seq_len
        return 0 if self.qjl_packed is None else self.qjl_packed.shape[2]

    def append(
        self,
        q_dict: dict,
        norms_in: torch.Tensor,
        B: int,
        S_new: int,
    ) -> None:
        if self._slab_active:
            if self._seq_len + S_new <= self._max_seq_len:
                self._append_slab(q_dict, norms_in, B, S_new)
                return
            if not self._overflow_warned:
                warnings.warn(
                    f"TurboQuant KV slab overflow at key store "
                    f"(seq_len={self._seq_len} + {S_new} > "
                    f"max_seq_len={self._max_seq_len}). Falling back to "
                    f"cat-growth and disabling CUDA-graph capture for this "
                    f"layer.",
                    stacklevel=3,
                )
                self._overflow_warned = True
            self._demote_slab()

        H = q_dict["gamma"].shape[0]
        D = self.head_dim

        # qjl signs: [H, B*S_new, D] → pack (1-bit) → [H, B, S_new, ceil(D/8)]
        qjl_4d = q_dict["qjl"].reshape(H, B, S_new, D)
        self.qjl_packed = _cat_or_init(self.qjl_packed, pack_signs(qjl_4d), dim=2)

        # gamma/norms: narrow to fp16 at store time (Phase A). Kernel upcasts
        # on load; fp16 precision is ample for a per-position scalar.
        gamma_3d = q_dict["gamma"].reshape(H, B, S_new).to(torch.float16)
        self.gamma = _cat_or_init(self.gamma, gamma_3d, dim=2)

        norms_3d = norms_in.reshape(H, B, S_new).to(torch.float16)
        self.norms = _cat_or_init(self.norms, norms_3d, dim=2)

        # mse indices: [H, B*S_new, D] → pack at (key_bits-1) bits, or split
        # into hi/lo buckets when outlier split is active.
        if q_dict["mse"] is not None:
            if self.n_out > 0:
                idx_hi_4d = q_dict["mse"]["idx_hi"].reshape(H, B, S_new, self.n_out)
                idx_lo_4d = q_dict["mse"]["idx_lo"].reshape(
                    H, B, S_new, D - self.n_out
                )
                self.mse_packed_hi = _cat_or_init(
                    self.mse_packed_hi,
                    pack_bits(idx_hi_4d, n_bits=self.bits_hi - 1),
                    dim=2,
                )
                self.mse_packed_lo = _cat_or_init(
                    self.mse_packed_lo,
                    pack_bits(idx_lo_4d, n_bits=self.bits_lo - 1),
                    dim=2,
                )
            else:
                mse_bits = self.key_bits - 1
                mse_4d = q_dict["mse"]["idx"].reshape(H, B, S_new, D)
                self.mse_packed = _cat_or_init(
                    self.mse_packed, pack_bits(mse_4d, n_bits=mse_bits), dim=2
                )

    def _append_slab(
        self,
        q_dict: dict,
        norms_in: torch.Tensor,
        B: int,
        S_new: int,
    ) -> None:
        H = q_dict["gamma"].shape[0]
        D = self.head_dim
        sl = slice(self._seq_len, self._seq_len + S_new)

        qjl_4d = q_dict["qjl"].reshape(H, B, S_new, D)
        self.qjl_packed[:, :, sl, :].copy_(pack_signs(qjl_4d))

        gamma_3d = q_dict["gamma"].reshape(H, B, S_new)
        self.gamma[:, :, sl].copy_(gamma_3d)

        norms_3d = norms_in.reshape(H, B, S_new)
        self.norms[:, :, sl].copy_(norms_3d)

        if q_dict["mse"] is not None:
            if self.n_out > 0:
                idx_hi_4d = q_dict["mse"]["idx_hi"].reshape(H, B, S_new, self.n_out)
                idx_lo_4d = q_dict["mse"]["idx_lo"].reshape(
                    H, B, S_new, D - self.n_out
                )
                self.mse_packed_hi[:, :, sl, :].copy_(
                    pack_bits(idx_hi_4d, n_bits=self.bits_hi - 1)
                )
                self.mse_packed_lo[:, :, sl, :].copy_(
                    pack_bits(idx_lo_4d, n_bits=self.bits_lo - 1)
                )
            elif self.mse_packed is not None:
                mse_bits = self.key_bits - 1
                mse_4d = q_dict["mse"]["idx"].reshape(H, B, S_new, D)
                self.mse_packed[:, :, sl, :].copy_(pack_bits(mse_4d, n_bits=mse_bits))

        self._seq_len += S_new

    def to_quantized_dict(self) -> tuple[dict, torch.Tensor]:
        """Unpack all stored tokens and return (q_dict, norms).

        Returns tensors shaped [H, B*S, D] / [H, B*S] suitable for the
        batched quantizer's dequantize_with_norm().
        """
        H, B = self.qjl_packed.shape[:2]
        S = self._seq_len if self._slab_active else self.qjl_packed.shape[2]
        D = self.head_dim

        qjl_src = self.qjl_packed[:, :, :S, :] if self._slab_active else self.qjl_packed
        gamma_src = self.gamma[:, :, :S] if self._slab_active else self.gamma
        norms_src = self.norms[:, :, :S] if self._slab_active else self.norms

        qjl = unpack_signs(qjl_src, n_values=D).reshape(H, B * S, D)
        gamma = gamma_src.reshape(H, B * S)
        norms = norms_src.reshape(H, B * S)

        mse_q: dict | None = None
        if self.n_out > 0:
            hi_src = (
                self.mse_packed_hi[:, :, :S, :]
                if self._slab_active
                else self.mse_packed_hi
            )
            lo_src = (
                self.mse_packed_lo[:, :, :S, :]
                if self._slab_active
                else self.mse_packed_lo
            )
            idx_hi = unpack_bits(
                hi_src, n_bits=self.bits_hi - 1, n_values=self.n_out
            ).reshape(H, B * S, self.n_out)
            idx_lo = unpack_bits(
                lo_src, n_bits=self.bits_lo - 1, n_values=D - self.n_out
            ).reshape(H, B * S, D - self.n_out)
            mse_q = {"idx_hi": idx_hi, "idx_lo": idx_lo}
        elif self.mse_packed is not None:
            mse_bits = self.key_bits - 1
            mse_src = (
                self.mse_packed[:, :, :S, :] if self._slab_active else self.mse_packed
            )
            mse_idx = unpack_bits(
                mse_src, n_bits=mse_bits, n_values=D
            ).reshape(H, B * S, D)
            mse_q = {"idx": mse_idx}

        return {"mse": mse_q, "qjl": qjl, "gamma": gamma}, norms

    # ---- lifecycle helpers -----------------------------------------------
    #
    # All lifecycle ops break pointer identity of the underlying tensors
    # (they allocate new buffers). Callers that depend on stable pointers
    # (e.g. CUDA-graph capture) must invalidate their caches around these.

    def reorder(self, beam_idx: torch.LongTensor) -> None:
        """Reorder batch dimension for beam search (beam_idx selects B dim)."""
        if self.qjl_packed is None:
            return
        self.qjl_packed = self.qjl_packed.index_select(1, beam_idx).contiguous()
        self.gamma = self.gamma.index_select(1, beam_idx).contiguous()
        self.norms = self.norms.index_select(1, beam_idx).contiguous()
        if self.mse_packed is not None:
            self.mse_packed = self.mse_packed.index_select(1, beam_idx).contiguous()
        if self.mse_packed_hi is not None:
            self.mse_packed_hi = self.mse_packed_hi.index_select(1, beam_idx).contiguous()
            self.mse_packed_lo = self.mse_packed_lo.index_select(1, beam_idx).contiguous()
        if self._slab_active:
            self._slab_B = int(beam_idx.shape[0])

    def crop(self, max_length: int) -> None:
        if self._slab_active:
            self._seq_len = min(self._seq_len, max(0, max_length))
            return
        if self.qjl_packed is not None:
            self.qjl_packed = self.qjl_packed[:, :, :max_length, ...]
            self.gamma = self.gamma[:, :, :max_length]
            self.norms = self.norms[:, :, :max_length]
        if self.mse_packed is not None:
            self.mse_packed = self.mse_packed[:, :, :max_length, ...]
        if self.mse_packed_hi is not None:
            self.mse_packed_hi = self.mse_packed_hi[:, :, :max_length, ...]
            self.mse_packed_lo = self.mse_packed_lo[:, :, :max_length, ...]

    def batch_repeat_interleave(self, repeats: int) -> None:
        if self.qjl_packed is None:
            return
        self.qjl_packed = self.qjl_packed.repeat_interleave(repeats, dim=1)
        self.gamma = self.gamma.repeat_interleave(repeats, dim=1)
        self.norms = self.norms.repeat_interleave(repeats, dim=1)
        if self.mse_packed is not None:
            self.mse_packed = self.mse_packed.repeat_interleave(repeats, dim=1)
        if self.mse_packed_hi is not None:
            self.mse_packed_hi = self.mse_packed_hi.repeat_interleave(repeats, dim=1)
            self.mse_packed_lo = self.mse_packed_lo.repeat_interleave(repeats, dim=1)
        if self._slab_active:
            self._slab_B = int(self.qjl_packed.shape[1])

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        if self.qjl_packed is None:
            return
        self.qjl_packed = self.qjl_packed.index_select(1, indices).contiguous()
        self.gamma = self.gamma.index_select(1, indices).contiguous()
        self.norms = self.norms.index_select(1, indices).contiguous()
        if self.mse_packed is not None:
            self.mse_packed = self.mse_packed.index_select(1, indices).contiguous()
        if self.mse_packed_hi is not None:
            self.mse_packed_hi = self.mse_packed_hi.index_select(1, indices).contiguous()
            self.mse_packed_lo = self.mse_packed_lo.index_select(1, indices).contiguous()
        if self._slab_active:
            self._slab_B = int(indices.shape[0])


class _BatchedValueStore:
    """Compressed value storage for all H attention heads in a layer."""

    def __init__(
        self,
        value_bits: int,
        head_dim: int,
        *,
        n_out: int = 0,
        bits_hi: int | None = None,
        bits_lo: int | None = None,
    ) -> None:
        if n_out < 0 or n_out > head_dim:
            raise ValueError(
                f"n_out must be in [0, head_dim]; got n_out={n_out}, "
                f"head_dim={head_dim}"
            )
        if n_out > 0:
            if bits_hi is None or bits_lo is None:
                raise ValueError("bits_hi and bits_lo are required when n_out > 0")
            # V stage uses full bit-widths with no QJL stage; require >= 1.
            if bits_hi < 1 or bits_lo < 1:
                raise ValueError(
                    "bits_hi and bits_lo must be >= 1 for the V store; "
                    f"got bits_hi={bits_hi}, bits_lo={bits_lo}"
                )

        self.value_bits = value_bits
        self.head_dim = head_dim
        self.n_out = n_out
        self.bits_hi = bits_hi
        self.bits_lo = bits_lo
        # Single-bucket mode (n_out == 0).
        self.idx_packed: torch.Tensor | None = None   # uint8 [H, B, S, packed_D]
        # Split-bucket mode. V stage uses full bit-widths (no -1).
        self.idx_packed_hi: torch.Tensor | None = None
        self.idx_packed_lo: torch.Tensor | None = None
        self.norms: torch.Tensor | None = None         # f32   [H, B, S]

        self._max_seq_len: int | None = None
        self._slab_B: int | None = None
        self._seq_len: int = 0
        self._slab_active: bool = False
        self._overflow_warned: bool = False

    @property
    def slab_active(self) -> bool:
        return self._slab_active

    def preallocate(
        self,
        max_seq_len: int,
        B: int,
        H: int,
        device: torch.device,
    ) -> None:
        """See _BatchedKeyStore.preallocate."""
        D = self.head_dim
        self._max_seq_len = max_seq_len
        self._slab_B = B
        self._seq_len = 0
        if self.n_out > 0:
            self.idx_packed = None
            self.idx_packed_hi = torch.empty(
                (H, B, max_seq_len, packed_dim(self.n_out, self.bits_hi)),
                dtype=torch.uint8,
                device=device,
            )
            self.idx_packed_lo = torch.empty(
                (H, B, max_seq_len, packed_dim(D - self.n_out, self.bits_lo)),
                dtype=torch.uint8,
                device=device,
            )
        else:
            self.idx_packed = torch.empty(
                (H, B, max_seq_len, packed_dim(D, self.value_bits)),
                dtype=torch.uint8,
                device=device,
            )
        self.norms = torch.empty(
            (H, B, max_seq_len), dtype=torch.float16, device=device
        )
        self._slab_active = True
        self._overflow_warned = False

    def _demote_slab(self) -> None:
        if not self._slab_active:
            return
        S = self._seq_len
        if self.idx_packed is not None:
            self.idx_packed = self.idx_packed[:, :, :S, :].contiguous()
        if self.idx_packed_hi is not None:
            self.idx_packed_hi = self.idx_packed_hi[:, :, :S, :].contiguous()
            self.idx_packed_lo = self.idx_packed_lo[:, :, :S, :].contiguous()
        self.norms = self.norms[:, :, :S].contiguous()
        self._slab_active = False
        self._max_seq_len = None
        self._slab_B = None

    def seq_length(self) -> int:
        if self._slab_active:
            return self._seq_len
        if self.idx_packed is not None:
            return self.idx_packed.shape[2]
        if self.idx_packed_hi is not None:
            return self.idx_packed_hi.shape[2]
        return 0

    def append(
        self,
        q_dict: dict,
        norms_in: torch.Tensor,
        B: int,
        S_new: int,
    ) -> None:
        if self._slab_active:
            if self._seq_len + S_new <= self._max_seq_len:
                self._append_slab(q_dict, norms_in, B, S_new)
                return
            if not self._overflow_warned:
                warnings.warn(
                    f"TurboQuant KV slab overflow at value store "
                    f"(seq_len={self._seq_len} + {S_new} > "
                    f"max_seq_len={self._max_seq_len}). Falling back to "
                    f"cat-growth and disabling CUDA-graph capture for this "
                    f"layer.",
                    stacklevel=3,
                )
                self._overflow_warned = True
            self._demote_slab()

        H = norms_in.shape[0]
        D = self.head_dim

        if self.n_out > 0:
            idx_hi_4d = q_dict["idx_hi"].reshape(H, B, S_new, self.n_out)
            idx_lo_4d = q_dict["idx_lo"].reshape(H, B, S_new, D - self.n_out)
            self.idx_packed_hi = _cat_or_init(
                self.idx_packed_hi,
                pack_bits(idx_hi_4d, n_bits=self.bits_hi),
                dim=2,
            )
            self.idx_packed_lo = _cat_or_init(
                self.idx_packed_lo,
                pack_bits(idx_lo_4d, n_bits=self.bits_lo),
                dim=2,
            )
        else:
            idx_4d = q_dict["idx"].reshape(H, B, S_new, D)
            self.idx_packed = _cat_or_init(
                self.idx_packed, pack_bits(idx_4d, n_bits=self.value_bits), dim=2
            )

        norms_3d = norms_in.reshape(H, B, S_new).to(torch.float16)
        self.norms = _cat_or_init(self.norms, norms_3d, dim=2)

    def _append_slab(
        self,
        q_dict: dict,
        norms_in: torch.Tensor,
        B: int,
        S_new: int,
    ) -> None:
        H = norms_in.shape[0]
        D = self.head_dim
        sl = slice(self._seq_len, self._seq_len + S_new)

        if self.n_out > 0:
            idx_hi_4d = q_dict["idx_hi"].reshape(H, B, S_new, self.n_out)
            idx_lo_4d = q_dict["idx_lo"].reshape(H, B, S_new, D - self.n_out)
            self.idx_packed_hi[:, :, sl, :].copy_(
                pack_bits(idx_hi_4d, n_bits=self.bits_hi)
            )
            self.idx_packed_lo[:, :, sl, :].copy_(
                pack_bits(idx_lo_4d, n_bits=self.bits_lo)
            )
        else:
            idx_4d = q_dict["idx"].reshape(H, B, S_new, D)
            self.idx_packed[:, :, sl, :].copy_(pack_bits(idx_4d, n_bits=self.value_bits))

        norms_3d = norms_in.reshape(H, B, S_new)
        self.norms[:, :, sl].copy_(norms_3d)

        self._seq_len += S_new

    def to_quantized_dict(self) -> tuple[dict, torch.Tensor]:
        D = self.head_dim
        if self.n_out > 0:
            H, B = self.idx_packed_hi.shape[:2]
            S = self._seq_len if self._slab_active else self.idx_packed_hi.shape[2]
            hi_src = (
                self.idx_packed_hi[:, :, :S, :]
                if self._slab_active
                else self.idx_packed_hi
            )
            lo_src = (
                self.idx_packed_lo[:, :, :S, :]
                if self._slab_active
                else self.idx_packed_lo
            )
            norms_src = self.norms[:, :, :S] if self._slab_active else self.norms

            idx_hi = unpack_bits(
                hi_src, n_bits=self.bits_hi, n_values=self.n_out
            ).reshape(H, B * S, self.n_out)
            idx_lo = unpack_bits(
                lo_src, n_bits=self.bits_lo, n_values=D - self.n_out
            ).reshape(H, B * S, D - self.n_out)
            norms = norms_src.reshape(H, B * S)
            return {"idx_hi": idx_hi, "idx_lo": idx_lo}, norms

        H, B = self.idx_packed.shape[:2]
        S = self._seq_len if self._slab_active else self.idx_packed.shape[2]
        idx_src = self.idx_packed[:, :, :S, :] if self._slab_active else self.idx_packed
        norms_src = self.norms[:, :, :S] if self._slab_active else self.norms

        idx = unpack_bits(
            idx_src, n_bits=self.value_bits, n_values=D
        ).reshape(H, B * S, D)
        norms = norms_src.reshape(H, B * S)

        return {"idx": idx}, norms

    def _primary_packed(self) -> torch.Tensor | None:
        return self.idx_packed if self.idx_packed is not None else self.idx_packed_hi

    def reorder(self, beam_idx: torch.LongTensor) -> None:
        primary = self._primary_packed()
        if primary is None:
            return
        if self.idx_packed is not None:
            self.idx_packed = self.idx_packed.index_select(1, beam_idx).contiguous()
        if self.idx_packed_hi is not None:
            self.idx_packed_hi = self.idx_packed_hi.index_select(1, beam_idx).contiguous()
            self.idx_packed_lo = self.idx_packed_lo.index_select(1, beam_idx).contiguous()
        self.norms = self.norms.index_select(1, beam_idx).contiguous()
        if self._slab_active:
            self._slab_B = int(beam_idx.shape[0])

    def crop(self, max_length: int) -> None:
        if self._slab_active:
            self._seq_len = min(self._seq_len, max(0, max_length))
            return
        if self.idx_packed is not None:
            self.idx_packed = self.idx_packed[:, :, :max_length, ...]
        if self.idx_packed_hi is not None:
            self.idx_packed_hi = self.idx_packed_hi[:, :, :max_length, ...]
            self.idx_packed_lo = self.idx_packed_lo[:, :, :max_length, ...]
        if self.norms is not None:
            self.norms = self.norms[:, :, :max_length]

    def batch_repeat_interleave(self, repeats: int) -> None:
        primary = self._primary_packed()
        if primary is None:
            return
        if self.idx_packed is not None:
            self.idx_packed = self.idx_packed.repeat_interleave(repeats, dim=1)
        if self.idx_packed_hi is not None:
            self.idx_packed_hi = self.idx_packed_hi.repeat_interleave(repeats, dim=1)
            self.idx_packed_lo = self.idx_packed_lo.repeat_interleave(repeats, dim=1)
        self.norms = self.norms.repeat_interleave(repeats, dim=1)
        if self._slab_active:
            new_B = (
                self.idx_packed.shape[1]
                if self.idx_packed is not None
                else self.idx_packed_hi.shape[1]
            )
            self._slab_B = int(new_B)

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        primary = self._primary_packed()
        if primary is None:
            return
        if self.idx_packed is not None:
            self.idx_packed = self.idx_packed.index_select(1, indices).contiguous()
        if self.idx_packed_hi is not None:
            self.idx_packed_hi = self.idx_packed_hi.index_select(1, indices).contiguous()
            self.idx_packed_lo = self.idx_packed_lo.index_select(1, indices).contiguous()
        self.norms = self.norms.index_select(1, indices).contiguous()
        if self._slab_active:
            self._slab_B = int(indices.shape[0])


# ---------------------------------------------------------------------------
# Per-layer cache
# ---------------------------------------------------------------------------


class TurboQuantCacheLayer(CacheLayerMixin):
    """Per-layer KV cache with TurboQuant compression.

    Keys are quantized with TurboQuantProd (unbiased inner product).
    Values are quantized with TurboQuantMSE (MSE-optimal reconstruction).

    All H KV heads are processed in a single batched matmul call per layer,
    and indices are bit-packed to match the theoretical storage of b bits per
    coordinate.

    Args:
        layer_idx: Transformer layer index.
        num_kv_heads: Number of key-value attention heads.
        head_dim: Dimension per attention head.
        key_bits: Bits per coordinate for key quantization.
        value_bits: Bits per coordinate for value quantization.
        seed: Base random seed.
        device: Target device for quantizer matrices.
    """

    is_sliding = False

    def __init__(
        self,
        layer_idx: int,
        num_kv_heads: int,
        head_dim: int,
        key_bits: int = 4,
        value_bits: int = 4,
        seed: int = 42,
        device: torch.device | str | None = None,
        use_fused_kernel: bool = False,
        max_seq_len: int | None = None,
        use_cuda_graph: bool = False,
        graph_pool: Any = None,
        *,
        n_out: int = 0,
        key_bits_hi: int | None = None,
        key_bits_lo: int | None = None,
        value_bits_hi: int | None = None,
        value_bits_lo: int | None = None,
    ):
        super().__init__()
        self._layer_idx = layer_idx
        self._num_kv_heads = num_kv_heads
        self._head_dim = head_dim
        self._key_bits = key_bits
        self._value_bits = value_bits
        self._seed = seed
        self._target_device = device
        self._seq_length = 0
        self._use_fused_kernel = use_fused_kernel
        self._n_out = n_out
        self._key_bits_hi = key_bits_hi
        self._key_bits_lo = key_bits_lo
        self._value_bits_hi = value_bits_hi
        self._value_bits_lo = value_bits_lo

        # Phase 11b: CUDA-graph capture. Slab preallocation + graph capture
        # require max_seq_len; otherwise we stay on the cat-growth path.
        self._max_seq_len = max_seq_len
        self._use_cuda_graph = (
            use_cuda_graph and use_fused_kernel and max_seq_len is not None
        )
        self._graph_pool = graph_pool
        self._slab_preallocated = False
        self._graph_cache: dict[int, _CapturedGraph] = {}
        self._graph_buffers_ready = False
        self._graph_shape_key: tuple[Any, ...] | None = None
        self._q_static: torch.Tensor | None = None
        self._shared_scratch: Any = None

        seed_base = seed + layer_idx * 1000

        # Non-overlapping seed ranges so MSE rotation, QJL projection, and
        # value rotation matrices are generated from independent seeds.
        # Each range accommodates up to 100 heads before collision.
        mse_seeds = [seed_base + h for h in range(num_kv_heads)]
        qjl_seeds = [seed_base + 100 + h for h in range(num_kv_heads)]
        val_seeds = [seed_base + 500 + h for h in range(num_kv_heads)]

        self._key_quantizer = BatchedTurboQuantProdTorch(
            d=head_dim,
            b=key_bits,
            num_heads=num_kv_heads,
            mse_seeds=mse_seeds,
            qjl_seeds=qjl_seeds,
            device=device,
            n_out=n_out,
            bits_hi=key_bits_hi,
            bits_lo=key_bits_lo,
        )
        self._value_quantizer = BatchedTurboQuantMSETorch(
            d=head_dim,
            b=value_bits,
            num_heads=num_kv_heads,
            seeds=val_seeds,
            device=device,
            n_out=n_out,
            bits_hi=value_bits_hi,
            bits_lo=value_bits_lo,
        )

        self._key_store = _BatchedKeyStore(
            key_bits=key_bits,
            head_dim=head_dim,
            n_out=n_out,
            bits_hi=key_bits_hi,
            bits_lo=key_bits_lo,
        )
        self._value_store = _BatchedValueStore(
            value_bits=value_bits,
            head_dim=head_dim,
            n_out=n_out,
            bits_hi=value_bits_hi,
            bits_lo=value_bits_lo,
        )

        # Incremental dense fp32 cache — avoids O(N²) full-dequant each step.
        # Holds [B, H, S_total, D] in fp32; returned as model_dtype at update() exit.
        self._dense_keys: torch.Tensor | None = None
        self._dense_values: torch.Tensor | None = None

        # Lazily-computed per-Q-head rotations for the fused kernel. Populated
        # on first decode via _prepare_fused_state(); invalidated by reset().
        self._fused_state_ready: bool = False
        self._kv_for_q: torch.Tensor | None = None
        self._S_k_per_q: torch.Tensor | None = None
        self._Pi_k_per_q: torch.Tensor | None = None
        self._Pi_v_per_q: torch.Tensor | None = None
        self._key_cent_lut: torch.Tensor | None = None
        self._val_cent_lut: torch.Tensor | None = None
        self._fused_H_q: int | None = None

    def _prepare_fused_state(self, device: torch.device, H_q: int) -> None:
        """Precompute per-Q-head rotation tensors used by the fused kernel.

        These tensors depend only on quantizer init state and the (fixed)
        GQA ratio, so they are safe to compute once and reuse for every
        decode step on this layer.
        """
        if self._fused_state_ready and self._fused_H_q == H_q:
            return
        if H_q % self._num_kv_heads != 0:
            raise ValueError(
                f"H_q={H_q} not divisible by num_kv_heads={self._num_kv_heads}"
            )
        gqa_groups = H_q // self._num_kv_heads
        kv_for_q = torch.arange(H_q, device=device) // gqa_groups

        S_k = self._key_quantizer.qjl.S.to(device=device, dtype=torch.float32)
        Pi_v = self._value_quantizer.rotation.Pi.to(device=device, dtype=torch.float32)
        self._S_k_per_q = S_k.index_select(0, kv_for_q).contiguous()
        self._Pi_v_per_q = Pi_v.index_select(0, kv_for_q).contiguous()

        if self._key_bits - 1 > 0:
            Pi_k = self._key_quantizer.mse.rotation.Pi.to(
                device=device, dtype=torch.float32
            )
            self._Pi_k_per_q = Pi_k.index_select(0, kv_for_q).contiguous()
        else:
            self._Pi_k_per_q = None

        # Centroid LUTs are kernel-visible tensors; caching them here lets the
        # CUDA-graph driver bind a stable pointer at capture time.
        from ..kernels.decompress_attn import _key_centroid_lut, _val_centroid_lut
        self._key_cent_lut = (
            _key_centroid_lut(self._key_quantizer).to(device).contiguous()
        )
        self._val_cent_lut = (
            _val_centroid_lut(self._value_quantizer).to(device).contiguous()
        )

        self._kv_for_q = kv_for_q
        self._fused_H_q = H_q
        self._fused_state_ready = True

    # ------------------------------------------------------------------
    # CUDA-graph driver (Phase 11b)
    # ------------------------------------------------------------------

    def _drop_graph_cache(self) -> None:
        """Tear down captured graphs. Pointer-identity invariants no longer hold."""
        graphs = getattr(self, "_graph_cache", None)
        if graphs:
            graphs.clear()
        else:
            self._graph_cache = {}

    def _ensure_graph_buffers(
        self,
        device: torch.device,
        B: int,
        H_q: int,
        D: int,
    ) -> None:
        """Allocate long-lived q-input and scratch buffers sized for the
        worst-case bucket (S = max_seq_len). Idempotent across decode steps.
        Rebuilds if B / H_q / D / device changed since last allocation.
        """
        from ..kernels.decompress_attn import FusedAttnScratch, _pick_split_size

        shape_key = (B, H_q, D, device)
        if (
            getattr(self, "_graph_buffers_ready", False)
            and self._graph_shape_key == shape_key
        ):
            return

        # Shape change invalidates any captured graphs.
        self._drop_graph_cache()

        max_seq = self._max_seq_len or 1
        max_split_size = _pick_split_size(max_seq, B, H_q)
        # Worst-case split count across all buckets ≤ max_seq_len. Every
        # bucket's own num_splits ≤ this, because _pick_split_size is
        # non-decreasing in S_total and num_splits = ceil(S/split_size)
        # stays close to the SM-saturation target.
        max_num_splits_any_bucket = 0
        S = max_seq
        while S > 0:
            ss = _pick_split_size(S, B, H_q)
            ns = (S + ss - 1) // ss
            max_num_splits_any_bucket = max(max_num_splits_any_bucket, ns)
            S //= 2

        self._q_static = torch.empty(
            B, H_q, 1, D, dtype=torch.float32, device=device
        )
        self._shared_scratch = FusedAttnScratch(
            partials_acc=torch.empty(
                B, H_q, max_num_splits_any_bucket, D,
                dtype=torch.float32, device=device,
            ),
            partials_m=torch.empty(
                B, H_q, max_num_splits_any_bucket,
                dtype=torch.float32, device=device,
            ),
            partials_l=torch.empty(
                B, H_q, max_num_splits_any_bucket,
                dtype=torch.float32, device=device,
            ),
            out=torch.empty(
                B, H_q, D, dtype=torch.float32, device=device,
            ),
            num_splits_max=max_num_splits_any_bucket,
            split_size=max_split_size,
        )
        self._graph_shape_key = shape_key
        self._graph_buffers_ready = True
        self._graph_cache: dict[int, _CapturedGraph] = {}

    def _capture_bucket(
        self,
        S_bucket: int,
        scale: float,
        B: int,
        H_q: int,
        D: int,
        device: torch.device,
    ) -> _CapturedGraph:
        """Warm up and capture the fused decode for one S_bucket."""
        from ..kernels.decompress_attn import (
            FusedAttnScratch,
            _pick_split_size,
            fused_decompress_attention_triton,
        )

        split_size = _pick_split_size(S_bucket, B, H_q)
        num_splits_max = (S_bucket + split_size - 1) // split_size

        # Bucket-local scratch view. Shares backing storage with the shared
        # scratch buffer; only num_splits_max/split_size differ so the kernel
        # iterates the right number of splits.
        bucket_scratch = FusedAttnScratch(
            partials_acc=self._shared_scratch.partials_acc,
            partials_m=self._shared_scratch.partials_m,
            partials_l=self._shared_scratch.partials_l,
            out=self._shared_scratch.out,
            num_splits_max=num_splits_max,
            split_size=split_size,
        )

        s_total_static = torch.tensor(
            max(1, self._seq_length), dtype=torch.int32, device=device
        )

        def _run() -> torch.Tensor:
            return fused_decompress_attention_triton(
                self._q_static,
                self._key_store,
                self._value_store,
                self._key_quantizer,
                self._value_quantizer,
                scale=scale,
                pi_k_per_q=self._Pi_k_per_q,
                s_k_per_q=self._S_k_per_q,
                pi_v_per_q=self._Pi_v_per_q,
                kv_for_q=self._kv_for_q,
                key_cent_lut=self._key_cent_lut,
                val_cent_lut=self._val_cent_lut,
                s_total_tensor=s_total_static,
                scratch=bucket_scratch,
            )

        # Warmup on side stream, required by torch.cuda.graph. This also
        # lets Triton's autotuner specialise before capture.
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            _run()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self._graph_pool):
            out_captured = _run()

        return _CapturedGraph(
            graph=graph,
            out_captured=out_captured,
            s_total_static=s_total_static,
            scale=scale,
            split_size=split_size,
            num_splits_max=num_splits_max,
        )

    def _fused_graph_forward(
        self, query: torch.Tensor, scale: float
    ) -> torch.Tensor:
        """Replay (or capture-then-replay) the fused decode for ``query``.

        Returns ``[B, H_q, 1, D]`` in ``query.dtype``. Assumes the caller has
        already validated slab preallocation and S_q == 1.
        """
        from ..kernels.decompress_attn import _s_bucket

        B, H_q, _, D = query.shape
        device = query.device

        self._prepare_fused_state(device, H_q)
        self._ensure_graph_buffers(device, B, H_q, D)

        seq_len = self._seq_length
        S_bucket = _s_bucket(max(1, seq_len))

        cached = self._graph_cache.get(S_bucket)
        if cached is None or cached.scale != scale:
            cached = self._capture_bucket(S_bucket, scale, B, H_q, D, device)
            self._graph_cache[S_bucket] = cached

        # Replay: update device scalar and q_static, then launch the graph.
        # copy_ handles the fp16/bf16 → fp32 cast when needed.
        self._q_static.copy_(query)
        cached.s_total_static.fill_(max(1, seq_len))
        cached.graph.replay()

        out = cached.out_captured
        if out.dtype != query.dtype:
            out = out.to(query.dtype)
        else:
            out = out.clone()
        return out

    def lazy_initialization(self, key_states: torch.Tensor) -> None:
        self.dtype = key_states.dtype
        self.device = key_states.device
        self.keys = torch.tensor([], dtype=self.dtype, device=self.device)
        self.values = torch.tensor([], dtype=self.dtype, device=self.device)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize new K/V states, return all dequantized K/V for attention.

        Args:
            key_states: [batch, num_kv_heads, seq_new, head_dim]
            value_states: [batch, num_kv_heads, seq_new, head_dim]
            cache_kwargs: Unused (accepted for API compatibility).

        Returns:
            (all_keys, all_values) both [batch, num_kv_heads, total_seq, head_dim].
            Output dtype matches the input dtype for fp16/bf16/fp32. For fp8 inputs
            (itemsize=1), output is promoted to fp16 because vanilla PyTorch attention
            cannot consume fp8 tensors. Internal quantizer math always runs in fp32.

        Speed note: dequantization is O(S_new) per step (only new tokens), not
        O(S_total). Total work over N decode steps is O(N) instead of O(N²).
        The compressed store is still populated for memory_tracker reporting.
        """
        if self.keys is None:
            self.lazy_initialization(key_states)

        model_dtype = key_states.dtype
        B, H, S_new, D = key_states.shape

        # Lazily preallocate packed slabs once B is known. Graph capture needs
        # stable tensor pointers and cannot cooperate with cat-growth.
        if (
            self._use_cuda_graph
            and not self._slab_preallocated
            and self._max_seq_len is not None
        ):
            self._key_store.preallocate(
                self._max_seq_len, B, H, key_states.device
            )
            self._value_store.preallocate(
                self._max_seq_len, B, H, key_states.device
            )
            self._slab_preallocated = True

        # Reshape to [H, B*S_new, D] so all heads process in one batched call.
        k_batched = key_states.permute(1, 0, 2, 3).reshape(H, B * S_new, D)
        v_batched = value_states.permute(1, 0, 2, 3).reshape(H, B * S_new, D)

        # quantize_with_norm casts to fp32 internally; no pre-cast needed.
        q_k, norms_k = self._key_quantizer.quantize_with_norm(k_batched)
        q_v, norms_v = self._value_quantizer.quantize_with_norm(v_batched)

        # Pack into compressed store (for memory_tracker / future fused kernel).
        self._key_store.append(q_k, norms_k, B, S_new)
        self._value_store.append(q_v, norms_v, B, S_new)
        self._seq_length += S_new

        # Fused-kernel fast path: skip dense buffer growth on decode. The custom
        # attention hook reads directly from the compressed stores; what we
        # return here is ignored for decode steps. Prefill (S_new > 1) still
        # needs dense K/V because the fused kernel is decode-only. Split-bucket
        # mode (n_out > 0) is not wired in the fused kernel yet (Phase B-3);
        # fall back to the dense dequant path until that lands.
        if (
            self._use_fused_kernel
            and S_new == 1
            and self._seq_length > 1
            and self._n_out == 0
        ):
            empty = key_states.new_empty(0)
            return empty, empty

        # Dequantize only the NEW tokens (O(S_new), not O(S_total)).
        # q_k / q_v already have the right shapes [H, B*S_new, D] — pass directly.
        new_keys_flat = self._key_quantizer.dequantize_with_norm(q_k, norms_k)
        new_vals_flat = self._value_quantizer.dequantize_with_norm(q_v, norms_v)

        # Reshape from [H, B*S_new, D] → [B, H, S_new, D]
        new_keys = new_keys_flat.reshape(H, B, S_new, D).permute(1, 0, 2, 3)
        new_vals = new_vals_flat.reshape(H, B, S_new, D).permute(1, 0, 2, 3)

        # Append new tokens to dense fp32 buffers (one cheap cat per step).
        if self._dense_keys is None:
            self._dense_keys = new_keys
            self._dense_values = new_vals
        else:
            self._dense_keys = torch.cat([self._dense_keys, new_keys], dim=2)
            self._dense_values = torch.cat([self._dense_values, new_vals], dim=2)

        # FP8 guard: float8_e4m3fn / float8_e5m2 have itemsize=1.
        # Vanilla PyTorch attention matmuls cannot consume FP8 tensors, so
        # promote to FP16. Full FP8 support (Transformer Engine) is Phase 11f.
        safe_dtype = model_dtype if model_dtype.itemsize >= 2 else torch.float16
        return self._dense_keys.to(safe_dtype), self._dense_values.to(safe_dtype)

    def _dequantize_all_keys(
        self, batch_size: int, num_heads: int, head_dim: int
    ) -> torch.Tensor:
        """Dequantize all cached keys, return [B, H, S, D]."""
        q_dict, norms = self._key_store.to_quantized_dict()
        # flat: [H, B*S, D]
        flat = self._key_quantizer.dequantize_with_norm(q_dict, norms)
        # Reshape and permute to [B, H, S, D]
        return flat.reshape(num_heads, batch_size, self._seq_length, head_dim).permute(
            1, 0, 2, 3
        )

    def _dequantize_all_values(
        self, batch_size: int, num_heads: int, head_dim: int
    ) -> torch.Tensor:
        """Dequantize all cached values, return [B, H, S, D]."""
        q_dict, norms = self._value_store.to_quantized_dict()
        flat = self._value_quantizer.dequantize_with_norm(q_dict, norms)
        return flat.reshape(num_heads, batch_size, self._seq_length, head_dim).permute(
            1, 0, 2, 3
        )

    def get_seq_length(self) -> int:
        return self._seq_length

    def get_max_cache_shape(self) -> int:
        return -1

    def get_mask_sizes(self, cache_position) -> tuple[int, int]:
        # HF's Cache.get_mask_sizes passes a plain int query_length to the
        # per-layer method, while create_causal_mask may pass a cache_position
        # tensor. Accept both.
        if hasattr(cache_position, "shape"):
            query_length = cache_position.shape[0]
        else:
            query_length = int(cache_position)
        kv_length = query_length + self._seq_length
        return kv_length, 0

    def reorder_cache(self, beam_idx: torch.LongTensor) -> None:
        self._key_store.reorder(beam_idx)
        self._value_store.reorder(beam_idx)
        if self._dense_keys is not None:
            self._dense_keys = self._dense_keys[beam_idx]
            self._dense_values = self._dense_values[beam_idx]
        # Slab tensors were reassigned — captured graphs now hold stale
        # pointers. Drop and allow re-capture on next decode.
        self._drop_graph_cache()
        self._graph_buffers_ready = False

    def reset(self) -> None:
        self._seq_length = 0
        self._key_store = _BatchedKeyStore(
            key_bits=self._key_bits,
            head_dim=self._head_dim,
            n_out=self._n_out,
            bits_hi=self._key_bits_hi,
            bits_lo=self._key_bits_lo,
        )
        self._value_store = _BatchedValueStore(
            value_bits=self._value_bits,
            head_dim=self._head_dim,
            n_out=self._n_out,
            bits_hi=self._value_bits_hi,
            bits_lo=self._value_bits_lo,
        )
        self._slab_preallocated = False
        self._dense_keys = None
        self._dense_values = None
        self._fused_state_ready = False
        self._kv_for_q = None
        self._S_k_per_q = None
        self._Pi_k_per_q = None
        self._Pi_v_per_q = None
        self._key_cent_lut = None
        self._val_cent_lut = None
        self._fused_H_q = None
        self._drop_graph_cache()
        if self.keys is not None:
            self.keys = torch.tensor([], dtype=self.dtype, device=self.device)
            self.values = torch.tensor([], dtype=self.dtype, device=self.device)

    def crop(self, max_length: int) -> None:
        if max_length < 0:
            max_length = self._seq_length - abs(max_length)
        if self._seq_length <= max_length:
            return
        self._seq_length = max_length
        self._key_store.crop(max_length)
        self._value_store.crop(max_length)
        if self._dense_keys is not None:
            self._dense_keys = self._dense_keys[:, :, :max_length, :]
            self._dense_values = self._dense_values[:, :, :max_length, :]
        # Crop may cross a bucket boundary; captures keyed by the old bucket
        # no longer reflect the current valid range. The cheapest safe move
        # is to drop and re-capture on demand.
        self._drop_graph_cache()

    def batch_repeat_interleave(self, repeats: int) -> None:
        self._key_store.batch_repeat_interleave(repeats)
        self._value_store.batch_repeat_interleave(repeats)
        if self._dense_keys is not None:
            self._dense_keys = self._dense_keys.repeat_interleave(repeats, dim=0)
            self._dense_values = self._dense_values.repeat_interleave(repeats, dim=0)
        self._drop_graph_cache()
        self._graph_buffers_ready = False

    def batch_select_indices(self, indices: torch.Tensor) -> None:
        self._key_store.batch_select_indices(indices)
        self._value_store.batch_select_indices(indices)
        if self._dense_keys is not None:
            self._dense_keys = self._dense_keys[indices]
            self._dense_values = self._dense_values[indices]
        self._drop_graph_cache()
        self._graph_buffers_ready = False


# ---------------------------------------------------------------------------
# Full multi-layer cache
# ---------------------------------------------------------------------------


class TurboQuantCache(Cache):
    """HuggingFace-compatible KV cache with TurboQuant compression.

    Drop-in replacement for DynamicCache. Keys are quantized with
    TurboQuantProd (unbiased inner product preservation) and values
    with TurboQuantMSE (MSE-optimal reconstruction).

    Indices are bit-packed to achieve theoretical compression ratios:
    b=4 → ~3.7×, b=3 → ~4.7×, b=2 → ~6.7× vs FP16 DynamicCache.

    Args:
        config: HuggingFace model config (PretrainedConfig).
        key_bits: Bits per coordinate for key quantization (default 4).
        value_bits: Bits per coordinate for value quantization (default 4).
        per_layer_bits: Optional per-layer overrides, e.g.
            ``{0: {"key_bits": 3, "value_bits": 4}}``. When ``outlier_frac > 0``
            and no explicit bucket widths are provided, each layer's
            ``key_bits_hi/lo`` defaults to ``(kb + 1, kb - 1)`` derived from
            the layer's own (possibly overridden) ``key_bits`` — and similarly
            for values.
        seed: Base random seed for reproducibility.
        device: Target device for quantizer matrices.
        outlier_frac: Fraction of head-dim channels quantized at higher bits
            (paper heuristic, Phase B). ``0.0`` disables the split.
        key_bits_hi/key_bits_lo: Explicit outlier / regular bit widths for
            keys. If unset, defaults follow the paper formula
            ``hi = key_bits + 1``, ``lo = key_bits - 1``.
        value_bits_hi/value_bits_lo: Same for values.

    Example::

        from transformers import AutoModelForCausalLM
        from turboquant.integrations.hf_cache import TurboQuantCache

        model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
        cache = TurboQuantCache(model.config, key_bits=4, value_bits=4)
        outputs = model.generate(**inputs, past_key_values=cache, use_cache=True)
    """

    def __init__(
        self,
        config: PretrainedConfig,
        key_bits: int = 4,
        value_bits: int = 4,
        per_layer_bits: dict[int, dict[str, int]] | None = None,
        seed: int = 42,
        device: torch.device | str | None = None,
        use_fused_kernel: bool = False,
        max_seq_len: int | None = None,
        use_cuda_graph: bool | None = None,
        outlier_frac: float = 0.0,
        key_bits_hi: int | None = None,
        key_bits_lo: int | None = None,
        value_bits_hi: int | None = None,
        value_bits_lo: int | None = None,
    ):
        text_config = config.get_text_config(decoder=True)
        num_layers = text_config.num_hidden_layers
        num_kv_heads = getattr(
            text_config, "num_key_value_heads", text_config.num_attention_heads
        )
        head_dim = getattr(
            text_config,
            "head_dim",
            text_config.hidden_size // text_config.num_attention_heads,
        )

        per_layer_bits = per_layer_bits or {}

        if use_cuda_graph is None:
            use_cuda_graph = _env_flag("TURBOQUANT_CUDA_GRAPH", default=False)
        graph_enabled = bool(
            use_cuda_graph and use_fused_kernel and max_seq_len is not None
        )
        # Shared memory pool keeps per-layer captures cheap and predictable.
        graph_pool = (
            torch.cuda.graph_pool_handle()
            if graph_enabled and torch.cuda.is_available()
            else None
        )

        # Outlier split (Phase B): n_out channels quantized at higher bits,
        # remainder at lower bits. Defaults follow the paper formula when
        # explicit bucket widths aren't supplied (hi = b+1, lo = b-1).
        if not (0.0 <= outlier_frac < 1.0):
            raise ValueError(
                f"outlier_frac must be in [0, 1); got {outlier_frac}"
            )
        n_out = int(round(outlier_frac * head_dim)) if outlier_frac > 0 else 0

        def _resolve_bits(
            stage: str,
            base_bits: int,
            hi_override: int | None,
            lo_override: int | None,
            min_bits: int,
        ) -> tuple[int | None, int | None]:
            """Derive bucket widths from base bits + optional overrides.

            Returns (None, None) when split mode is off. Validates that the
            resolved widths are within the kernel's supported range.
            """
            if n_out == 0:
                return None, None
            hi = hi_override if hi_override is not None else base_bits + 1
            lo = lo_override if lo_override is not None else base_bits - 1
            # bit_packing supports n_bits in [1, 6]; K stage packs (bits - 1)
            # so the hi-stage ceiling is 7; V stage packs full bits => 6.
            max_bits = 7 if stage == "key" else 6
            if hi < min_bits or hi > max_bits:
                raise ValueError(
                    f"{stage}_bits_hi must be in [{min_bits}, {max_bits}]; "
                    f"got {hi} (base_bits={base_bits}, outlier_frac={outlier_frac})"
                )
            if lo < min_bits or lo > max_bits:
                raise ValueError(
                    f"{stage}_bits_lo must be in [{min_bits}, {max_bits}]; "
                    f"got {lo} (base_bits={base_bits}, outlier_frac={outlier_frac})"
                )
            return hi, lo

        layers = []
        for i in range(num_layers):
            layer_cfg = per_layer_bits.get(i, {})
            kb = layer_cfg.get("key_bits", key_bits)
            vb = layer_cfg.get("value_bits", value_bits)
            # Per-layer overrides can also specify bucket widths directly;
            # otherwise derive from this layer's own kb/vb so overrides stay
            # internally consistent.
            kbh_i, kbl_i = _resolve_bits(
                "key",
                kb,
                layer_cfg.get("key_bits_hi", key_bits_hi),
                layer_cfg.get("key_bits_lo", key_bits_lo),
                min_bits=2,  # K stage: QJL needs 1 bit, MSE needs >= 1
            )
            vbh_i, vbl_i = _resolve_bits(
                "value",
                vb,
                layer_cfg.get("value_bits_hi", value_bits_hi),
                layer_cfg.get("value_bits_lo", value_bits_lo),
                min_bits=1,  # V stage: MSE only
            )
            layers.append(
                TurboQuantCacheLayer(
                    layer_idx=i,
                    num_kv_heads=num_kv_heads,
                    head_dim=head_dim,
                    key_bits=kb,
                    value_bits=vb,
                    seed=seed,
                    device=device,
                    use_fused_kernel=use_fused_kernel,
                    max_seq_len=max_seq_len,
                    use_cuda_graph=graph_enabled,
                    graph_pool=graph_pool,
                    n_out=n_out,
                    key_bits_hi=kbh_i,
                    key_bits_lo=kbl_i,
                    value_bits_hi=vbh_i,
                    value_bits_lo=vbl_i,
                )
            )

        super().__init__(layers=layers)

        self._key_bits = key_bits
        self._value_bits = value_bits
        self._num_layers = num_layers
        self._num_kv_heads = num_kv_heads
        self._head_dim = head_dim
        self._use_fused_kernel = use_fused_kernel
        self._max_seq_len = max_seq_len
        self._use_cuda_graph = graph_enabled
        self._outlier_frac = outlier_frac
        self._n_out = n_out

    def bind(self):
        """Context manager: route the fused attention hook to this cache."""
        from .attention import bind_cache

        return bind_cache(self)

"""PyTorch QJL (Quantized Johnson-Lindenstrauss) 1-bit quantizer.

Port of qjl.py — random sign projections for unbiased inner product estimation.
"""

import math

import torch


class QJLTorch:
    """1-bit quantizer using random Gaussian projections (PyTorch).

    Quantizes r to sign(S @ r) where S is a d x d random Gaussian matrix.
    Dequantization: sqrt(pi/2)/d * gamma * S^T @ signs.

    Args:
        d: Dimension of input vectors.
        seed: Random seed for reproducibility.
        device: Target device for the projection matrix.
    """

    def __init__(
        self,
        d: int,
        seed: int | None = None,
        device: torch.device | str | None = None,
    ):
        gen = torch.Generator(device="cpu")
        if seed is not None:
            gen.manual_seed(seed)
        self.S = torch.randn(d, d, generator=gen, dtype=torch.float32)
        if device is not None:
            self.S = self.S.to(device)
        self.d = d
        self._scale = math.sqrt(math.pi / 2) / d

    def quantize(self, r: torch.Tensor) -> torch.Tensor:
        """Compute sign(r @ S^T), mapping zeros to +1.

        Args:
            r: Input tensor of shape (d,) or (n, d).

        Returns:
            Sign tensor, dtype int8, values in {-1, +1}.
        """
        projected = r @ self.S.T
        signs = torch.sign(projected).to(torch.int8)
        signs[signs == 0] = 1
        return signs

    def dequantize(
        self, signs: torch.Tensor, gamma: float | torch.Tensor
    ) -> torch.Tensor:
        """Reconstruct vector(s) from sign bits and norm.

        Args:
            signs: Sign tensor of shape (d,) or (n, d), values in {-1, +1}.
            gamma: Scalar norm or tensor of norms of shape (n,).

        Returns:
            Reconstructed tensor, same shape as signs.
        """
        signs_f = signs.to(torch.float32)
        if signs_f.dim() == 1:
            return self._scale * gamma * (self.S.T @ signs_f)
        else:
            gamma_t = torch.as_tensor(
                gamma, dtype=torch.float32, device=signs.device
            )
            return self._scale * gamma_t.unsqueeze(-1) * (signs_f @ self.S)


class BatchedQJLTorch:
    """Stacked QJL projection matrices for H attention heads.

    Holds H independent d×d Gaussian projection matrices as a single
    [H, d, d] tensor to enable batched GPU matmuls via torch.bmm.

    Args:
        d: Dimension of input vectors.
        num_heads: Number of heads (H).
        seeds: Sequence of H integer seeds, one per head.
        device: Target device.
    """

    def __init__(
        self,
        d: int,
        num_heads: int,
        seeds: list[int],
        device: torch.device | str | None = None,
    ):
        if len(seeds) != num_heads:
            raise ValueError(f"Expected {num_heads} seeds, got {len(seeds)}")
        S_list = [QJLTorch(d, seed=s, device=device).S for s in seeds]
        self.S = torch.stack(S_list, dim=0)  # [H, D, D]
        self.d = d
        self.num_heads = num_heads
        self._scale = math.sqrt(math.pi / 2) / d

    def quantize(self, r: torch.Tensor) -> torch.Tensor:
        """Compute sign(r @ S^T) for all heads simultaneously.

        Args:
            r: Tensor of shape (H, N, D).

        Returns:
            Sign tensor of dtype int8 and shape (H, N, D), values in {-1, +1}.
        """
        projected = torch.bmm(r, self.S.transpose(-1, -2))  # [H, N, D]
        signs = torch.sign(projected).to(torch.int8)
        signs[signs == 0] = 1
        return signs

    def dequantize(
        self, signs: torch.Tensor, gamma: torch.Tensor
    ) -> torch.Tensor:
        """Reconstruct residual estimate for all heads simultaneously.

        Args:
            signs: int8 tensor of shape (H, N, D), values in {-1, +1}.
            gamma: float32 tensor of shape (H, N), residual norms.

        Returns:
            Reconstructed tensor of shape (H, N, D).
        """
        signs_f = signs.to(torch.float32)
        # signs_f @ S is equivalent to S^T @ signs^T per row vector convention.
        result = torch.bmm(signs_f, self.S)  # [H, N, D]
        return self._scale * gamma.unsqueeze(-1) * result

"""Lightweight autoencoder and Gaussian smoothing used by CTCal.

The autoencoder (paper Sec. 3.2, supplementary Fig. 1) projects a single-channel
attention map to a 16-d code. The semantic-level loss compares codes; a
reconstruction proxy task on the same autoencoder is what stops it from collapsing
to a constant encoding.

The supplementary gives two variants, 16x16 input for SD 2.1 and 64x64 for SD 3
(higher compression). FLUX at 512px packs latents 2x2, giving a 32x32 patch grid,
so the encoder is built from the input resolution and keeps the 1024-d flattened
bottleneck the supplementary specifies.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GaussianSmoothing(nn.Module):
    """Depthwise Gaussian blur over attention maps.

    Maps are smoothed before comparison so the loss keys on the spatial layout of
    the response rather than single-pixel spikes.
    """

    def __init__(self, kernel_size: int = 3, sigma: float = 0.5):
        super().__init__()
        self.kernel_size = kernel_size
        self.pad = kernel_size // 2

        grid = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2
        kernel_1d = torch.exp(-(grid**2) / (2 * sigma**2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        self.register_buffer("weight", kernel_2d[None, None], persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, 1, H, W) -> (N, 1, H, W)."""
        x = F.pad(x, (self.pad,) * 4, mode="reflect")
        return F.conv2d(x, self.weight.to(x.dtype))


class Reshape(nn.Module):
    def __init__(self, c: int, h: int, w: int):
        super().__init__()
        self.c, self.h, self.w = c, h, w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.reshape(-1, self.c, self.h, self.w)


class AttentionAutoencoder(nn.Module):
    """Encoder/decoder over one attention map, following supplementary Fig. 1.

    `n_down` strided stages take the map to a 4x4 spatial grid, so the flattened
    bottleneck is 64*4*4 = 1024 for any input resolution -- the FC(1024, 16) the
    supplementary specifies.
    """

    def __init__(self, resolution: int = 32, latent_dim: int = 16):
        super().__init__()
        n_down = int(math.log2(resolution // 4))
        if 4 * 2**n_down != resolution:
            raise ValueError(f"resolution {resolution} must be 4 * a power of two")
        self.resolution = resolution
        self.n_down = n_down

        enc: list[nn.Module] = [
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.LeakyReLU(0.2),
        ]
        for _ in range(n_down - 1):
            enc += [nn.Conv2d(64, 64, 3, 2, 1), nn.LeakyReLU(0.2)]
        enc += [nn.Conv2d(64, 64, 3, 1, 1), nn.Flatten(), nn.Linear(1024, latent_dim)]
        self.encoder = nn.Sequential(*enc)

        dec: list[nn.Module] = [nn.Linear(latent_dim, 1024), Reshape(64, 4, 4)]
        for _ in range(n_down - 1):
            dec += [
                nn.Conv2d(64, 64, 3, 1, 1),
                nn.LeakyReLU(0.2),
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            ]
        dec += [
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(32, 1, 3, 1, 1),
            nn.Sigmoid(),
        ]
        self.decoder = nn.Sequential(*dec)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        return z, self.decoder(z)

"""Image and proprio encoders for the factored model.

WorldViT: pixels to world latent. EgoMLP: proprio to ego latent.
Two separate encoders so object and robot info stay split.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TransformerBlock(nn.Module):
    """Transformer block: attention + MLP."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 2.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        # full attention over all patches (no mask)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class WorldViT(nn.Module):
    """Small ViT: image (B,C,H,W) -> world latent (B, out_dim)."""

    def __init__(
        self,
        img_size: int = 64,
        in_chans: int = 3,
        patch_size: int = 8,
        embed_dim: int = 192,
        depth: int = 4,
        num_heads: int = 6,
        mlp_ratio: float = 2.0,
        out_dim: int = 192,
        dropout: float = 0.0,
        head_norm: str = "none",
    ):
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError(f"img_size ({img_size}) must be divisible by patch_size ({patch_size}).")
        if head_norm not in ("batchnorm", "none"):
            raise ValueError(f"head_norm must be 'batchnorm' or 'none', got {head_norm!r}.")
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # +1 for CLS token
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        # CLS -> world latent (optional BatchNorm on head)
        head_layers: list[nn.Module] = [nn.Linear(embed_dim, out_dim)]
        if head_norm == "batchnorm":
            head_layers.append(nn.BatchNorm1d(out_dim))
        self.head = nn.Sequential(*head_layers)
        self.out_dim = out_dim

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, H, W) float image in [0, 1] -> (B, out_dim)."""
        b = x.shape[0]
        x = self.patch_embed(x)  # (B, D, H/p, W/p)
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)  # (B, N+1, D)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.head(x[:, 0])  # CLS token


class EgoMLP(nn.Module):
    """MLP: proprio (B, in_dim) -> ego latent (B, out_dim)."""

    def __init__(self, in_dim: int, out_dim: int = 32, hidden_dim: int = 128, depth: int = 2):
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()]
            d = hidden_dim
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)
        self.out_dim = out_dim
        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, in_dim) -> (B, out_dim)."""
        return self.net(x)


def _init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)

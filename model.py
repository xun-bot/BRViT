"""Core BRViT architecture for SEED (three emotion classes)."""

import torch
import torch.nn as nn

from seed_data import CHANNEL_MAP_62, SEED_CHANNEL_ORDER


BRAIN_REGIONS = {
    "Frontal": {"rows": [0, 1, 2, 3], "cols": list(range(9))},
    "Temporal": {"rows": [3, 4, 5], "cols": [0, 1, 7, 8]},
    "Central": {"rows": [3, 4, 5], "cols": [2, 3, 4, 5, 6]},
    "Parietal": {"rows": [5, 6, 7], "cols": [1, 2, 3, 4, 5, 6, 7]},
    "Occipital": {"rows": [7, 8], "cols": list(range(9))},
}


def build_region_masks(map_size=9):
    masks = []
    electrode_mask = torch.zeros(map_size, map_size)
    for name in SEED_CHANNEL_ORDER:
        row, col = CHANNEL_MAP_62[name]
        electrode_mask[row, col] = 1.0
    for region in BRAIN_REGIONS.values():
        mask = torch.zeros(map_size, map_size)
        for row in region["rows"]:
            for col in region["cols"]:
                mask[row, col] = electrode_mask[row, col]
        masks.append(mask / mask.sum().clamp_min(1.0))
    return torch.stack(masks)


class FrequencySpatialAttention(nn.Module):
    def __init__(self, n_bands=5, reduction=2):
        super().__init__()
        hidden = max(n_bands // reduction, 1)
        self.pool_avg = nn.AdaptiveAvgPool2d(1)
        self.pool_max = nn.AdaptiveMaxPool2d(1)
        self.band_mlp = nn.Sequential(
            nn.Linear(n_bands, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, n_bands)
        )
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, x):
        batch, bands, _, _ = x.shape
        avg = self.band_mlp(self.pool_avg(x).view(batch, bands))
        maximum = self.band_mlp(self.pool_max(x).view(batch, bands))
        x = x * torch.sigmoid(avg + maximum).view(batch, bands, 1, 1)
        spatial = torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], dim=1)
        return x * torch.sigmoid(self.spatial_conv(spatial))


class BrainRegionPooling(nn.Module):
    def __init__(self, n_bands=5, embed_dim=64):
        super().__init__()
        self.register_buffer("region_masks", build_region_masks())
        self.projection = nn.Linear(n_bands, embed_dim)
        self.normalization = nn.LayerNorm(embed_dim)

    def forward(self, x):
        masks = self.region_masks[None, :, None, :, :]
        features = (x[:, None, :, :, :] * masks).sum(dim=(-2, -1))
        return self.normalization(self.projection(features))


class PatchRegionCrossAttention(nn.Module):
    def __init__(self, embed_dim=64, n_heads=4, dropout=0.5):
        super().__init__()
        self.patch_norm = nn.LayerNorm(embed_dim)
        self.region_norm = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim, n_heads, dropout=dropout, batch_first=True
        )

    def forward(self, patch_tokens, region_tokens):
        update, _ = self.attention(
            self.region_norm(region_tokens),
            self.patch_norm(patch_tokens),
            self.patch_norm(patch_tokens),
            need_weights=False,
        )
        return torch.cat([patch_tokens, region_tokens + update], dim=1)


class BRViT(nn.Module):
    def __init__(self, n_classes=3, embed_dim=64, n_heads=4, n_layers=1,
                 patch_size=3, patch_stride=2, dropout=0.5):
        super().__init__()
        self.n_patches = ((9 - patch_size) // patch_stride + 1) ** 2
        self.fsam = FrequencySpatialAttention(n_bands=5, reduction=2)
        self.patch_embedding = nn.Conv2d(
            5, embed_dim, kernel_size=patch_size, stride=patch_stride
        )
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.n_patches, embed_dim)
        )
        self.region_pooling = BrainRegionPooling(5, embed_dim)
        self.cross_attention = PatchRegionCrossAttention(embed_dim, n_heads, dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.output_norm = nn.LayerNorm(embed_dim)
        self.feature_norm = nn.LayerNorm(embed_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim * 2, n_classes)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(self, x, return_features=False):
        x = self.fsam(x)
        patch_tokens = self.patch_embedding(x).flatten(2).transpose(1, 2)
        patch_tokens = patch_tokens + self.position_embedding
        region_tokens = self.region_pooling(x)
        tokens = self.output_norm(
            self.encoder(self.cross_attention(patch_tokens, region_tokens))
        )
        patch_feature = tokens[:, :self.n_patches].mean(dim=1)
        region_feature = tokens[:, self.n_patches:].mean(dim=1)
        features = self.feature_norm(torch.cat([patch_feature, region_feature], dim=-1))
        logits = self.classifier(self.dropout(features))
        return (logits, features) if return_features else logits

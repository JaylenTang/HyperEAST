
import torch
import torch.nn as nn
import numpy as np

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class SpatialOperation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.op = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(),
            nn.Conv1d(dim, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x should be [Batch, Channels, Length] for Conv1d
        return x * self.op(x)

class ChannelOperation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.op = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(dim, dim, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x should be [Batch, Channels, Length] for Conv1d
        return x * self.op(x)

class AdditiveTokenMixer(nn.Module):
    def __init__(self, dim, attn_bias=False, proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.qkv = nn.Linear(dim, 3 * dim, bias=attn_bias)
        # 合并空间和通道操作
        self.mixed_op = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim),  # 保持空间信息
            nn.ReLU(),
            nn.Conv1d(dim, dim, kernel_size=1),  # 融合通道信息
            nn.Sigmoid()
        )
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, C).permute(2, 0, 1, 3)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 减少维度转换，直接在通道维上操作
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # 应用混合操作
        qk = self.mixed_op(q + k)  # 直接融合q和k后处理

        # 生成注意力图并应用于v
        attn = torch.tanh(qk) * v
        attn = attn.transpose(1, 2)  # 重置维度

        x = self.proj(attn)
        x = self.proj_drop(x)
        return x



class Attention(nn.Module):
    def __init__(self, dim, n_heads, qkv_bias=True, attn_p=0., proj_p=0.):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = (self.head_dim ** -0.5)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, x):
        B, N, _ = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, Batch, n_heads, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Compute dot-product attention
        dots = (q @ k.transpose(-2, -1)) * self.scale
        attn = dots.softmax(dim=-1)
        attn = self.attn_drop(attn)

        weighted_avg = attn @ v
        weighted_avg = weighted_avg.transpose(1, 2).reshape(B, N, self.dim)

        x = self.proj(weighted_avg)
        x = self.proj_drop(x)
        return x

class Transformer(nn.Module):
    def __init__(self, dim, depth, n_heads, mlp_ratio, qkv_bias, attn_p, proj_p, use_catm=True):
        super().__init__()
        print(f"Transformer initialized with use_catm={use_catm}")
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            attn_layer = AdditiveTokenMixer(dim, attn_bias=qkv_bias, proj_drop=attn_p) if use_catm else Attention(dim, n_heads=n_heads, qkv_bias=qkv_bias, attn_p=attn_p, proj_p=proj_p)
            self.layers.append(nn.ModuleList([
                PreNorm(dim, attn_layer),
                PreNorm(dim, FeedForward(dim, int(dim * mlp_ratio), dropout=proj_p))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)

class PatchEmbed3D(nn.Module):
    def __init__(self, img_size, patch_size, embed_dim, stride):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.stride = stride

        # Calculate the number of patches along each dimension
        self.num_patches_x = (img_size[2] - patch_size[2]) // stride[2] + 1
        self.num_patches_y = (img_size[1] - patch_size[1]) // stride[1] + 1
        self.num_patches_t = (img_size[0] - patch_size[0]) // stride[0] + 1
        self.num_patches = self.num_patches_t * self.num_patches_x * self.num_patches_y

        self.proj = nn.Conv3d(1, embed_dim, kernel_size=patch_size, stride=stride, padding=0)

    def forward(self, x):
        B, T, H, W = x.shape
        assert H == self.img_size[1] and W == self.img_size[2] and T == self.img_size[0], \
            f"Input image size ({T}*{H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]}*{self.img_size[2]})."

        x = x.reshape(B, 1, T, H, W)  # Add channel dimension
        x = self.proj(x)  # Apply the convolution
        x = x.flatten(2)  # Flatten the patch dimensions
        x = x.transpose(1, 2)  # [B, num_patches, embed_dim]

        return x

class FactoFormer(nn.Module):
    def __init__(self, img_size, spatial_patch, spectral_patch, n_classes, spatial_embed_dim, spectral_embed_dim, bands, depth, n_heads,
                 qkv_bias, attn_p, proj_p):

        super().__init__()

        self.spatial_patch_embed = PatchEmbed3D(img_size=img_size, patch_size=spatial_patch, embed_dim=spatial_embed_dim, stride=spatial_patch)
        self.spectral_patch_embed = PatchEmbed3D(img_size=img_size, patch_size=spectral_patch, embed_dim=spectral_embed_dim, stride=spectral_patch)

        num_patches_spatial = self.spatial_patch_embed.num_patches
        num_patches_spectral = bands

        self.spatial_pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches_spatial, spatial_embed_dim))
        self.spectral_pos_embed = nn.Parameter(torch.zeros(1, 1 + num_patches_spectral, spectral_embed_dim))

        self.spatial_cls_token = nn.Parameter(torch.zeros(1, 1, spatial_embed_dim))
        self.spectral_cls_token = nn.Parameter(torch.zeros(1, 1, spectral_embed_dim))

        # Specify that only the spatial transformer uses CATM
        self.spatial_transformer = Transformer(spatial_embed_dim, depth, n_heads, 8, qkv_bias, attn_p, proj_p, use_catm=False)
        self.spectral_transformer = Transformer(spectral_embed_dim, depth, n_heads, 4, qkv_bias, attn_p, proj_p, use_catm=True)

        dim = spectral_embed_dim + spatial_embed_dim
        self.lin = nn.Linear(dim, dim)
        self.pos_drop = nn.Dropout(p=proj_p)
        self.norm_spatial = nn.LayerNorm(spatial_embed_dim, eps=1e-6)
        self.norm_spectral = nn.LayerNorm(spectral_embed_dim, eps=1e-6)
        self.head = nn.Linear(dim, n_classes)

    def forward(self, x):
        n_samples = x.shape[0]
        x1 = self.spatial_patch_embed(x)
        x2 = self.spectral_patch_embed(x)

        cls_token_spatial = self.spatial_cls_token.expand(n_samples, -1, -1)
        cls_token_spectral = self.spectral_cls_token.expand(n_samples, -1, -1)

        x1 = torch.cat((cls_token_spatial, x1), dim=1)
        x2 = torch.cat((cls_token_spectral, x2), dim=1)

        x1 = x1 + self.spatial_pos_embed
        x2 = x2 + self.spectral_pos_embed

        x1 = self.pos_drop(x1)
        x2 = self.pos_drop(x2)

        x1 = self.spatial_transformer(x1)
        x1 = self.norm_spatial(x1)

        x2 = self.spectral_transformer(x2)
        x2 = self.norm_spectral(x2)

        cls_token_spatial = x1[:, 0]
        cls_token_spectral = x2[:, 0]

        out_x = torch.cat((cls_token_spatial, cls_token_spectral), 1)

        out_x = self.lin(out_x)
        out_x = self.head(out_x)

        return out_x


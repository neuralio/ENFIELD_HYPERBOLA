import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    """
    Splits the input image into patches and projects them into a higher-dimensional space.
    """
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, embed_dim, num_patches_h, num_patches_w)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x

class SelfAttention(nn.Module):
    """
    Self-attention mechanism to capture global dependencies.
    """
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, N, C = x.shape
        q = self.query(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, N, C)
        out = self.out(out)
        return out

class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, dropout_rate):
        super().__init__()
        self.layers = nn.ModuleList([
            ResidualTransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=embed_dim * 4,
                dropout=dropout_rate,
                activation="gelu",
                batch_first=True
            ) for _ in range(num_layers)
        ])
        self.attention = ResidualSelfAttention(embed_dim, num_heads)

    def forward(self, x):
        # Apply residual self-attention
        x = self.attention(x)
        
        # Store the input for residual connection
        identity = x
        
        # Apply transformer layers with progressive depth
        for i, layer in enumerate(self.layers):
            # Adjust layer depth dynamically
            layer_scale = 1.0 - (i / len(self.layers))
            x = layer(x) * layer_scale + identity
        
        return x

class ResidualSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.attention = SelfAttention(embed_dim, num_heads)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # Residual connection for self-attention
        return self.layer_norm(x + self.attention(x))

class ResidualTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def forward(self, x):
        # Add residual connections to standard transformer layer
        x = super().forward(x)
        return x
    

class SegmentationHead(nn.Module):
    """
    Decoder to reconstruct the spatial resolution and produce a feature map.
    Instead of projecting to num_classes, it projects to feature_dim channels.
    """
    def __init__(self, embed_dim, feature_dim, img_size, patch_size):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        self.decoder = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(embed_dim, feature_dim, kernel_size=1)
        )

    def forward(self, x):
        B, N, D = x.shape
        H = W = int(N ** 0.5)
        x = x.transpose(1, 2).view(B, D, H, W)
        x = F.interpolate(x, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False)
        x = self.decoder(x)
        return x

class SegmenterViT(nn.Module):
    """
    Vision Transformer for feature map extraction with attention mechanisms.
    """
    def __init__(self, img_size, patch_size, in_channels, embed_dim, num_heads, num_layers, dropout_rate):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.transformer_encoder = TransformerEncoder(embed_dim, num_heads, num_layers, dropout_rate)
        self.segmentation_head = SegmentationHead(embed_dim, embed_dim, img_size, patch_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, embed_dim))

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.transformer_encoder(x)
        x = self.segmentation_head(x)
        return x

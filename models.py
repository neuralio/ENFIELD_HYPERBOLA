import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.nn import HeteroConv, GATConv
from segmenter import SegmenterViT

# Squeeze-and-Excitation Block
class SEBlock(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction_ratio, channels),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

# Enhanced Residual Block with Group Normalization and SE Block
class EnhancedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, groups=8, dropout_rate=0.2, stride=1):
        super(EnhancedResidualBlock, self).__init__()
        
        # First convolution with Group Normalization
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
            nn.GroupNorm(min(groups, out_channels), out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate)
        )
        
        # Second convolution with Group Normalization
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(min(groups, out_channels), out_channels)
        )
        
        # Squeeze-and-Excitation block for channel recalibration
        self.se_block = SEBlock(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.GroupNorm(min(groups, out_channels), out_channels)
            )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        
        # Apply SE attention
        out = self.se_block(out)
        
        # Add residual connection
        out += self.shortcut(residual)
        out = self.relu(out)
        return out

# Enhanced UNet with deeper architecture
class EnhancedUNet(nn.Module):
    def __init__(self, in_channels, base_channels=16, out_channels=16, dropout_rate=0.2, groups=8):
        super(EnhancedUNet, self).__init__()

        # Encoder - now with 4 levels instead of 2
        self.down1 = EnhancedResidualBlock(in_channels, base_channels, groups, dropout_rate)
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = EnhancedResidualBlock(base_channels, base_channels*2, groups, dropout_rate)
        self.pool2 = nn.MaxPool2d(2)

        self.down3 = EnhancedResidualBlock(base_channels*2, base_channels*4, groups, dropout_rate)
        self.pool3 = nn.MaxPool2d(2)

        # Bridge
        self.bridge = EnhancedResidualBlock(base_channels*4, base_channels*4, groups, dropout_rate)

        # Decoder - with transposed convolutions and more levels
        self.uptrans3 = nn.ConvTranspose2d(base_channels * 4, base_channels * 4, 2, stride=2)
        self.up3 = nn.Sequential(
            nn.Conv2d(base_channels * 8, base_channels * 4, 3, padding=1),
            nn.GroupNorm(min(groups, base_channels * 4), base_channels * 4),
            nn.ReLU(),
            nn.Dropout2d(p=dropout_rate),
            nn.Conv2d(base_channels * 4, base_channels * 4, 3, padding=1),
            nn.GroupNorm(min(groups, base_channels * 4), base_channels * 4),
            nn.ReLU()
        )

        self.uptrans2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, stride=2)
        self.up2 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 2, 3, padding=1),
            nn.GroupNorm(min(groups, base_channels * 2), base_channels * 2),
            nn.ReLU(),
            nn.Dropout2d(p=dropout_rate),
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1),
            nn.GroupNorm(min(groups, base_channels * 2), base_channels * 2),
            nn.ReLU()
        )

        self.uptrans1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, stride=2)
        self.up1 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, 3, padding=1),  
            nn.GroupNorm(min(groups, base_channels), base_channels),
            nn.ReLU(),
            nn.Dropout2d(p=dropout_rate),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            nn.GroupNorm(min(groups, base_channels), base_channels),
            nn.ReLU()
        )

        # Output layer
        self.final = nn.Conv2d(base_channels, out_channels, kernel_size=1)

        # Deep supervision outputs
        self.deep_supervision = True
        if self.deep_supervision:
            self.ds_out3 = nn.Conv2d(base_channels*4, out_channels, kernel_size=1)
            self.ds_out2 = nn.Conv2d(base_channels*2, out_channels, kernel_size=1)

    def forward(self, x):
        # Store original input size for potential upsampling
        input_size = x.size()[2:]

        # Encoder
        x1 = self.down1(x)         # (B, base_channels, H, W)
        x1p = self.pool1(x1)       # (B, base_channels, H/2, W/2)

        x2 = self.down2(x1p)       # (B, base_channels*2, H/2, W/2)
        x2p = self.pool2(x2)       # (B, base_channels*2, H/4, W/4)

        x3 = self.down3(x2p)       # (B, base_channels*4, H/4, W/4)
        x3p = self.pool3(x3)       # (B, base_channels*4, H/8, W/8)

        # Bridge
        bridge = self.bridge(x3p)  # (B, base_channels*4, H/8, W/8)

        # Decoder
        x3u = self.uptrans3(bridge)   # (B, base_channels*4, H/4, W/4)
        x3cat = torch.cat([x3u, x3], dim=1)  # (B, base_channels*8, H/4, W/4)
        x3d = self.up3(x3cat)      # (B, base_channels*4, H/4, W/4)

        # Deep supervision output 3 
        if self.deep_supervision:
            out3 = self.ds_out3(x3d)
            out3 = F.interpolate(out3, size=input_size, mode='bilinear', align_corners=False)

        x2u = self.uptrans2(x3d)   # (B, base_channels*2, H/2, W/2)
        x2cat = torch.cat([x2u, x2], dim=1)  # (B, base_channels*4, H/2, W/2)
        x2d = self.up2(x2cat)      # (B, base_channels*2, H/2, W/2)

        # Deep supervision output 2 
        if self.deep_supervision:
            out2 = self.ds_out2(x2d)
            out2 = F.interpolate(out2, size=input_size, mode='bilinear', align_corners=False)

        x1u = self.uptrans1(x2d)   # (B, base_channels, H, W)
        x1cat = torch.cat([x1u, x1], dim=1)  # (B, base_channels*2, H, W)
        x1d = self.up1(x1cat)      # (B, base_channels, H, W)

        # Final output
        main_out = self.final(x1d)  # (B, out_channels, H, W)

        # Handle size mismatch if present
        #if main_out.size()[2:] != input_size:
        main_out = F.interpolate(main_out, size=input_size, mode='bilinear', align_corners=False)

        if self.deep_supervision:
            return main_out, out2, out3
        else:
            return main_out

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.5, stride=1):
        super().__init__()
        # Strided convolution for downsampling
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=dropout_rate) if dropout_rate > 0 else None
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        
        # Add residual connection
        out += self.shortcut(residual)
        out = self.relu(out)
        return out
    
# CNN Branch for Segmentation (FCN-style)
class CNNBranch(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CNNBranch, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, out_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
    def forward(self, x):
        return self.net(x)  # (B, out_channels, patch_size, patch_size)
    
class SelfAttention(nn.Module):
    """
    A simple self-attention mechanism.
    """
    def __init__(self, embed_dim, num_heads):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.to_qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.to_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        # x: (B, N, embed_dim)
        B, N, D = x.shape
        qkv = self.to_qkv(x)  # (B, N, 3*embed_dim)
        q, k, v = torch.chunk(qkv, 3, dim=-1)  # each: (B, N, embed_dim)
        # reshape for multi-head: (B, N, num_heads, head_dim) and then transpose to (B, num_heads, N, head_dim)
        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, N)
        attn = F.softmax(attn_scores, dim=-1)  # (B, num_heads, N, N)
        out = torch.matmul(attn, v)  # (B, num_heads, N, head_dim)
        
        # Merge heads
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        out = self.to_out(out)  # (B, N, embed_dim)
        return out
    
# ViT Branch for Segmentation
class SegmentationViT(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, embed_dim, num_layers, num_heads, dropout_rate):
        """
        Splits the input patch into sub-patches, processes them with a transformer encoder, and then reassembles a feature map.
        """
        super(SegmentationViT, self).__init__()
        assert image_size % patch_size == 0, "image_size must be divisible by patch_size"
        self.patch_size = patch_size  # size of sub-patches (e.g., 4)
        self.grid_size = image_size // patch_size  # number of sub-patches per side
        self.num_patches = self.grid_size ** 2
        self.patch_dim = in_channels * patch_size * patch_size
        
        self.proj = nn.Linear(self.patch_dim, embed_dim)
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
                                                    batch_first=True, dropout=dropout_rate)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Additional self-attention block
        #self.extra_attention = SelfAttention(embed_dim, num_heads)

        # A simple decoder to reassemble the spatial grid
        self.decoder_conv = nn.Conv2d(embed_dim, embed_dim, kernel_size=1)
    
    def forward(self, x):
        # x: (B, in_channels, patch_size_full, patch_size_full) where patch_size_full is the patch from the dataset.
        B, C, H, W = x.shape  # H = W = full patch size (e.g., 48)
        # We'll split the patch into sub-patches of size self.patch_size
        grid = H // self.patch_size
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
            # patches: (B, C, grid, grid, patch_size, patch_size)
            patches = patches.contiguous().view(B, C, -1, self.patch_size, self.patch_size)  # (B, C, N, p, p)
            patches = patches.permute(0, 2, 1, 3, 4).contiguous()  # (B, N, C, p, p)
            patches = patches.view(B, self.num_patches, -1)  # (B, N, patch_dim)
            
            tokens = self.proj(patches)  # (B, N, embed_dim)
            tokens = tokens + self.pos_embedding
            tokens = self.transformer(tokens)  # (B, N, embed_dim)
            
            # Apply extra self-attention:
            # tokens_attn = self.extra_attention(tokens)  # (B, N, embed_dim)
            # # Combine the transformer output and the extra self-attention output.
            # tokens = (tokens + tokens_attn) / 2.0

            # Reassemble tokens into spatial grid
            tokens = tokens.view(B, grid, grid, -1)  # (B, grid, grid, embed_dim)
            tokens = tokens.permute(0, 3, 1, 2).contiguous()  # (B, embed_dim, grid, grid)
            upsampled = F.interpolate(tokens, size=(H, W), mode='bilinear', align_corners=False)
            out = self.decoder_conv(upsampled)  # (B, embed_dim, H, W)
        return out

def create_patch_graph(side_length):
    """
    Creates a 2D grid graph for a square patch.
    
    Args:
        side_length: Size of one side of the square patch
        
    Returns:
        edge_index: Tensor of shape [2, num_edges] containing the edge indices
    """
    num_nodes = side_length * side_length
    
    # Create a 2D grid
    grid = torch.zeros(num_nodes, 2)
    for i in range(num_nodes):
        grid[i, 0] = i % side_length   # x-coordinate
        grid[i, 1] = i // side_length  # y-coordinate
    
    # Add edges for 8 neighbors (including diagonals)
    edge_list = []
    for i in range(num_nodes):
        x, y = grid[i]
        
        # Check 8 neighboring pixels
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue  # Skip self-loop
                
                nx, ny = x + dx, y + dy
                if 0 <= nx < side_length and 0 <= ny < side_length:
                    j = int(ny * side_length + nx)
                    edge_list.append([i, j])
    
    edge_index = torch.tensor(edge_list, dtype=torch.long).t()
    return edge_index

class HeterogeneousPatchGNN(nn.Module):
    def __init__(self, num_channels, hidden_channels, out_channels, num_heads, dropout_rate):
        super(HeterogeneousPatchGNN, self).__init__()

        self.num_channels = num_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.dropout_rate = dropout_rate

        # Spatial convolutions (within each channel)
        spatial_convs = {
            (f'ch_{ch}', 'spatial', f'ch_{ch}'): GATConv(
                in_channels=1,
                out_channels=hidden_channels,
                heads=num_heads,
                concat=False,
                add_self_loops=False
            ) for ch in range(num_channels)
        }

        # Inter-channel convolutions
        inter_channel_convs = {
            (f'ch_{src_ch}', f'inter_ch_{src_ch}_{dst_ch}', f'ch_{dst_ch}'): GATConv(
                1,
                hidden_channels,
                heads=num_heads,
                concat=False,
                add_self_loops=False
            ) for src_ch in range(num_channels) for dst_ch in range(num_channels) if src_ch != dst_ch
        }

        self.conv1 = HeteroConv({**spatial_convs, **inter_channel_convs}, aggr='mean')

        # Second convolution
        spatial_convs2 = {
            key: GATConv(hidden_channels, hidden_channels, heads=num_heads, concat=False, add_self_loops=False)
            for key in spatial_convs.keys()
        }
        inter_channel_convs2 = {
            key: GATConv(hidden_channels, hidden_channels, heads=num_heads, concat=False, add_self_loops=False)
            for key in inter_channel_convs.keys()
        }

        self.conv2 = HeteroConv({**spatial_convs2, **inter_channel_convs2}, aggr='mean')

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x_dict, edge_index_dict):
        # First GNN layer
        hidden = self.conv1(x_dict, edge_index_dict)
        hidden = {k: F.relu(self.dropout(v)) for k, v in hidden.items()}

        # Second GNN layer
        hidden = self.conv2(hidden, edge_index_dict)
        hidden = {k: F.relu(self.dropout(v)) for k, v in hidden.items()}

        # Concatenate all node features
        combined = torch.cat([hidden[k] for k in sorted(hidden.keys())], dim=-1)
        return combined
    
class HybridModel_1(nn.Module):
    def __init__(self, 
                 num_classes,
                 patch_size, 
                 in_channels,
                 dropout_rate):
        super(HybridModel_1, self).__init__()
        self.patch_size = patch_size
        
        # CNN Branch
        self.cnn_branch = EnhancedUNet(in_channels, 
                                       base_channels=16, 
                                       out_channels=16, 
                                       dropout_rate=dropout_rate)
        self.cnn_branch.deep_supervision = False  # Disable deep supervision

        # ViT Branch
        self.vit_patch_size = patch_size//8
        self.vit_branch = SegmenterViT(
            img_size=patch_size,
            patch_size=self.vit_patch_size,
            in_channels=in_channels,
            embed_dim=16,
            num_heads=4,
            num_layers=6,
            dropout_rate=dropout_rate
        )
        
        # Output channels from each branch
        cnn_out_channels = 16  # EnhancedUNet output
        vit_out_channels = 16  # SegmenterViT embed_dim
        
        # Replace the feature alignment and fusion attention with SpatialAttentionFusion
        self.fusion = SpatialAttentionFusion(
            input_channels=[cnn_out_channels, vit_out_channels],
            fusion_channels=16
        )
        
        # Dropout layer - fix this to be a layer, not a rate
        self.dropout = nn.Dropout(dropout_rate)
        
        # Final classifier
        self.classifier = nn.Conv2d(16, num_classes, kernel_size=1)
       
    def forward(self, x):
        B, C, H, W = x.shape

        # Get features from CNN and ViT branches
        cnn_feat = self.cnn_branch(x)  
        vit_feat = self.vit_branch(x)  
        
        # Use the SpatialAttentionFusion module directly
        attended_features = self.fusion([cnn_feat, vit_feat])
        
        # Apply dropout for regularization
        attended_features = self.dropout(attended_features)
        
        # Final classification
        main_output = self.classifier(attended_features)

        return main_output

class HybridModel_2(nn.Module):
    def __init__(self, 
                 num_classes,
                 patch_size, 
                 in_channels,
                 dropout_rate):
        super(HybridModel_2, self).__init__()
        self.patch_size = patch_size
        
        # CNN Branch
        self.cnn_branch = CNNBranch(in_channels, out_channels=16)

        self.vit_patch_size = patch_size//8
        self.vit_branch = SegmenterViT(
            img_size=patch_size,
            patch_size=self.vit_patch_size,
            in_channels=in_channels,
            embed_dim=32,
            num_heads=4,
            num_layers=6,
            dropout_rate=dropout_rate
        )
        
        # Output channels from each branch
        cnn_out_channels = 16
        vit_out_channels = 32  # SegmenterViT embed_dim
        
        # Feature alignment layers (normalize each branch to 32 channels)
        self.feature_alignment = nn.ModuleList([
            nn.Conv2d(cnn_out_channels, 32, kernel_size=1),
            nn.Conv2d(vit_out_channels, 32, kernel_size=1)
        ])
        
        # Spatial attention mechanism for fusion
        # This properly computes pixel-wise attention weights across branches
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 2, kernel_size=1)  # 2 branches
        )
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout_rate)
        
        # Final classifier
        self.classifier = nn.Conv2d(32, num_classes, kernel_size=1)
       
    def forward(self, x):
        B, C, H, W = x.shape

        # Get features from CNN and ViT branches
        cnn_feat = self.cnn_branch(x)  
        vit_feat = self.vit_branch(x)  
        
        # Align features to common channel dimension
        aligned_cnn = self.feature_alignment[0](cnn_feat)  
        aligned_vit = self.feature_alignment[1](vit_feat)  
        
        # Compute spatial attention weights from the combined features
        # This approach uses the average of both feature maps to generate attention
        feature_for_attention = (aligned_cnn + aligned_vit) / 2
        
        # Generate spatial attention weights 
        attention_weights = self.spatial_attention(feature_for_attention)
        attention_weights = F.softmax(attention_weights, dim=1)
        
        # Apply spatial attention to each branch
        # Reshape for proper broadcasting
        attention_weights = attention_weights.unsqueeze(2)
        
        # Stack aligned features
        stacked_features = torch.stack([aligned_cnn, aligned_vit], dim=1)
        
        # Apply attention weights to features
        # This performs element-wise multiplication and sum across branches
        attended_features = (stacked_features * attention_weights).sum(dim=1) 
        
        # Apply dropout for regularization
        attended_features = self.dropout(attended_features)
        
        # Final classification
        main_output = self.classifier(attended_features)

        return main_output
      

class HybridModel_3(nn.Module):
    def __init__(self, 
                 num_classes,
                 patch_size, 
                 in_channels,
                 dropout_rate):
        super(HybridModel_3, self).__init__()
        self.patch_size = patch_size
        
        # CNN Branch
        self.cnn_branch = EnhancedUNet(in_channels, 
                                       base_channels=16, 
                                       out_channels=16, 
                                       dropout_rate=dropout_rate)
        self.cnn_branch.deep_supervision = False  # Disable deep supervision
        
        # Dropout layer 
        self.dropout = nn.Dropout(dropout_rate)
        
        # Final classifier
        self.fusion_conv = nn.Conv2d(16, 16, kernel_size=1) 
        self.classifier = nn.Conv2d(16, num_classes, kernel_size=1)
       
    def forward(self, x): 
        B, C, H, W = x.shape
        # Get features from CNN
        cnn_feat = self.cnn_branch(x)  
        
        # Final classification
        fused = self.fusion_conv(cnn_feat)  
        logits = self.classifier(fused)  
        return logits
    

# Hybrid Model for Patch-based Pixel-wise Segmentation
class HybridModel_4(nn.Module):
    def __init__(self, 
                 num_classes,
                 patch_size, 
                 in_channels,
                 dropout_rate):
        super(HybridModel_4, self).__init__()
        self.patch_size = patch_size
        # CNN Branch: FCN style for patches
        self.cnn_branch = EnhancedUNet(in_channels, 
                                       base_channels=16, 
                                       out_channels=16, 
                                       dropout_rate=dropout_rate)
        self.cnn_branch.deep_supervision = False  # Disable deep supervision
      
        self.vit_patch_size = patch_size//8

        self.vit_branch = SegmenterViT(
            img_size=patch_size,
            patch_size=self.vit_patch_size,
            in_channels=in_channels,
            embed_dim=16,
            num_heads=4,
            num_layers=6,
            dropout_rate=dropout_rate
        )
        
        self.gnn_branch = HeterogeneousPatchGNN(
            num_channels=in_channels,      
            hidden_channels=16,
            out_channels=16,
            num_heads=4,
            dropout_rate=dropout_rate
        )

        cnn_out_channels = 16  # from EnhancedUNet (fixed)
        vit_out_channels = 16  # SegmenterViT embed_dim (fixed)
        gnn_out_channels = 16 * in_channels  # GNN depends on num_channels  
                        
        self.dropout_rate = nn.Dropout(dropout_rate)
                
        # Assuming cnn_out_channels, vit_out_channels, and gnn_out_channels are known
        self.fusion = SpatialAttentionFusion(
            input_channels=[cnn_out_channels, vit_out_channels, gnn_out_channels],
            fusion_channels=16 
        )

        self.classifier = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x, hetero_graph):
        B, C, H, W = x.shape
        
        cnn_feat = self.cnn_branch(x)
        vit_feat = self.vit_branch(x)
            
        # Ensure graph data is on the correct device
        hetero_graph = hetero_graph.to(x.device)
        
        # Assign features dynamically
        for ch in range(C):
            ch_key = f'ch_{ch}'
            hetero_graph[ch_key].x = x[:, ch, :, :].reshape(B * H * W, 1)
                
        # GNN forward
        gnn_feat_flat = self.gnn_branch(hetero_graph.x_dict, hetero_graph.edge_index_dict)
        gnn_feat = gnn_feat_flat.view(B, H, W, -1).permute(0, 3, 1, 2)
        
        # Use attention fusion instead of concatenation
        # Assuming you've initialized self.fusion as an instance of SpatialAttentionFusion
        fused = self.fusion([cnn_feat, vit_feat, gnn_feat])
        
        # No need for additional fusion_conv as the attention module handles this
        main_output = self.classifier(fused)
        
        return main_output

class HyperspectralGraphDataset(Dataset):
    def __init__(self, patch_dataset):
        """
        A wrapper dataset that takes an existing patch dataset and 
        returns items flexibly based on the dataset's structure.
        """
        self.patch_dataset = patch_dataset
    
    def __len__(self):
        return len(self.patch_dataset)
    
    def __getitem__(self, idx):
        # Get the item from the original dataset
        item = self.patch_dataset[idx]
        #print(f"Debug: len(item) = {len(item)}, item = {item}")
        # Handle different dataset return structures
        if len(item) == 3:
            # Standard case: patch_tensor, label_tensor, img_idx
            patch_tensor, label_tensor, img_idx = item
            #print("Returning 3 elements: patch_tensor, label_tensor, img_idx")
            return patch_tensor, label_tensor, img_idx 
        elif len(item) == 4:
            # Case with additional hetero_graph
            patch_tensor, label_tensor, img_idx, hetero_graph=item
            #print("Returning 4 elements: patch_tensor, label_tensor, img_idx, hetero_graph")
            return patch_tensor, label_tensor, img_idx, hetero_graph
        else:
            raise ValueError(f"Unexpected item structure with {len(item)} elements")
        
# --- Attention-based Fusion Module ---
class SpatialAttentionFusion(nn.Module):
    def __init__(self, input_channels, fusion_channels):
        """
        Args:
            input_channels: list of integers, the channel dimensions of each branch's output
            fusion_channels: output channel dimension after fusion
        """
        super(SpatialAttentionFusion, self).__init__()
        
        # Project each branch to a common channel dimension first
        self.projections = nn.ModuleList([
            nn.Conv2d(in_channels, fusion_channels, kernel_size=1) 
            for in_channels in input_channels
        ])
        
        # Attention mechanism for spatial-aware weighting
        self.attention = nn.Sequential(
            nn.Conv2d(fusion_channels, fusion_channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(fusion_channels // 4, len(input_channels), kernel_size=1)
        )
    
    def forward(self, features):
        """
        Args:
            features: list of tensors, each of shape (B, C_i, H, W)
        Returns:
            fused: Tensor of shape (B, fusion_channels, H, W)
        """
        # Project each branch to common dimension
        projected = [proj(f) for proj, f in zip(self.projections, features)]
        
        # Use the first projection as reference for attention computation
        # This assumes all feature maps have the same spatial dimensions
        reference = projected[0]
        
        # Compute attention weights (B, num_branches, H, W)
        attn_weights = self.attention(reference)
        attn_weights = torch.softmax(attn_weights, dim=1)
        
        # Apply attention weights - need to expand dims for broadcasting
        fused = torch.zeros_like(reference)
        for i, feat in enumerate(projected):
            # Extract the attention map for this branch: (B, 1, H, W)
            branch_attention = attn_weights[:, i:i+1, :, :]
            fused += feat * branch_attention
            
        return fused
    
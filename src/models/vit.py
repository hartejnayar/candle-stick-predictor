import torch
import torch.nn as nn
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PatchEmbedding(nn.Module):
    """Splits an image into non-overlapping patches and projects them to embedding dimension."""
    
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_channels: int = 3, embed_dim: int = 128):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        
        # Use a convolutional layer to extract and project patches efficiently
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input shape: (Batch, Channels, Height, Width)
        x = self.proj(x)                  # (Batch, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2)                  # (Batch, embed_dim, n_patches)
        x = x.transpose(1, 2)             # (Batch, n_patches, embed_dim)
        return x


class CandlestickVisionTransformer(nn.Module):
    """
    Vision Transformer tailored for learning classical candlestick chart patterns 
    and predicting short-horizon market direction.
    """
    
    def __init__(
        self, 
        img_size: int = 224, 
        patch_size: int = 16, 
        in_channels: int = 3, 
        num_classes: int = 3, 
        embed_dim: int = 128, 
        depth: int = 4, 
        num_heads: int = 4, 
        mlp_ratio: float = 4.0, 
        dropout: float = 0.1
    ):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.n_patches
        
        # Learnable classification token and positional encoding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=dropout)
        
        # Multi-Head Self-Attention Transformer Encoder Layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads, 
            dim_feedforward=int(embed_dim * mlp_ratio), 
            dropout=dropout, 
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # Final normalization and multi-class classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        
        # Weight initialization
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        logging.info(f"Initialized ViT with embed_dim={embed_dim}, depth={depth}, num_heads={num_heads}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x)           # (B, n_patches, embed_dim)
        
        # Prepend learnable [CLS] token to the patch sequence
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1) # (B, n_patches + 1, embed_dim)
        
        # Add positional embeddings
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Process through transformer encoder blocks (self-attention)
        x = self.transformer_encoder(x)
        
        # Extract the classification token representation
        x = self.norm(x[:, 0])
        x = self.head(x)                  # (B, num_classes)
        return x
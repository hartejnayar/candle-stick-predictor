import torch
import torch.nn as nn

class MultimodalFusionModel(nn.Module):
    """Fuses Vision Transformer (ViT) visual image embeddings with Multi-Layer Perceptron (MLP) numerical technical indicator embeddings."""
    
    def __init__(self, num_numerical_features: int, num_classes: int = 3, embedding_dim: int = 128):
        super(MultimodalFusionModel, self).__init__()
        self.numerical_mlp = nn.Sequential(
            nn.Linear(num_numerical_features, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, embedding_dim),
            nn.ReLU()
        )
        self.visual_projection = nn.Sequential(
            nn.Linear(3 * 224 * 224, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim),
            nn.ReLU()
        )
        self.fusion_head = nn.Sequential(
            nn.Linear(embedding_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, images: torch.Tensor, numericals: torch.Tensor) -> torch.Tensor:
        if images.dim() == 4:
            batch_size = images.size(0)
            images_flat = images.view(batch_size, -1)
        else:
            images_flat = images
            
        vis_emb = self.visual_projection(images_flat)
        num_emb = self.numerical_mlp(numericals)
        combined = torch.cat((vis_emb, num_emb), dim=1)
        return self.fusion_head(combined)
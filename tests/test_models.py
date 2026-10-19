import torch
from src.models.fusion_model import MultimodalFusionModel

def test_fusion_model_forward():
    model = MultimodalFusionModel(num_numerical_features=10, num_classes=3)
    dummy_images = torch.randn(2, 3, 224, 224)
    dummy_numerics = torch.randn(2, 10)
    logits = model(dummy_images, dummy_numerics)
    assert logits.shape == (2, 3)
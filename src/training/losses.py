import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance in financial direction forecasting."""
    
    def __init__(self, alpha: float = 0.25, gamma: int = 2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss
        return torch.mean(focal_loss)
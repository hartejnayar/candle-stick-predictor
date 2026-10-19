import numpy as np

class ModelExplainer:
    """Generates feature attribution and attention insights."""
    
    @staticmethod
    def get_feature_importance_weights(feature_names: list, model_weights: np.ndarray) -> dict:
        """Maps model weights back to feature names."""
        return dict(zip(feature_names, map(float, np.abs(model_weights))))
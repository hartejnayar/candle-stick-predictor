import numpy as np

class BaseStrategy:
    """Base trading strategy template."""
    def generate_signals(self, predictions: np.ndarray) -> np.ndarray:
        positions = np.zeros_like(predictions)
        positions[predictions == 2] = 1
        positions[predictions == 0] = -1
        return positions
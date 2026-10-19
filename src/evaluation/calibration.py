import numpy as np

def expected_calibration_error(confidences: np.ndarray, accuracies: np.ndarray, n_bins: int = 10) -> float:
    """Computes Expected Calibration Error (ECE) for model confidence reliability."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(confidences)
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i+1]
        in_bin = np.logical_and(confidences > bin_lower, confidences <= bin_upper)
        bin_size = np.sum(in_bin)
        
        if bin_size > 0:
            accuracy = np.mean(accuracies[in_bin])
            confidence = np.mean(confidences[in_bin])
            ece += (bin_size / n) * abs(accuracy - confidence)
            
    return float(ece)
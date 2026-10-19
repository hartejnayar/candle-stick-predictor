import torch
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def save_model_checkpoint(model: torch.nn.Module, filepath: str) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(model.state_dict(), filepath)
    logging.info(f"Model checkpoint saved successfully to {filepath}")

def load_model_checkpoint(model: torch.nn.Module, filepath: str) -> torch.nn.Module:
    if os.path.exists(filepath):
        model.load_state_dict(torch.load(filepath))
        logging.info(f"Model checkpoint loaded successfully from {filepath}")
    return model
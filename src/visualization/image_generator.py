import pandas as pd
import numpy as np
from PIL import Image

def generate_chart_tensor_image(df: pd.DataFrame) -> Image.Image:
    arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(arr)
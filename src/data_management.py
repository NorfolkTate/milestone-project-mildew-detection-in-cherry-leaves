from typing import Iterable, Tuple
import numpy as np
from PIL import Image
from pathlib import Path

DATA_ROOT = Path("inputs/dataset/dataset_mini")

def load_image_file(file) -> np.ndarray:
    return np.array(Image.open(file).convert("RGB"))

def load_image_path(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))

def resize_input_image(img_rgb: np.ndarray, size: Tuple[int, int] = (128, 128)) -> np.ndarray:
    return np.array(Image.fromarray(img_rgb).resize(size))

def preprocess_input(img_rgb_resized: np.ndarray) -> np.ndarray:
    x = img_rgb_resized.astype("float32") / 255.0
    return np.expand_dims(x, axis=0)

def iter_image_paths(folder: Path) -> Iterable[Path]:
    if not folder.exists():
        return []
    for p in folder.rglob("*"):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            yield p

# code explained by medium.com and ref. in readme
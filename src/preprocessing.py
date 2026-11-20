"""Pad-to-square hand X-ray preprocessing. GPU-ready: returns CPU tensors,
caller moves batches to cuda in train/eval loop."""
import cv2
import numpy as np
import torch

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_hand(path, size=512):
    """Grayscale pad-to-square then resize. Keeps hand proportions."""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"missing image: {path}")
    h, w = img.shape
    s = max(h, w)
    pad = np.zeros((s, s), np.uint8)
    pad[(s - h) // 2:(s - h) // 2 + h, (s - w) // 2:(s - w) // 2 + w] = img
    interp = cv2.INTER_AREA if s > size else cv2.INTER_LINEAR
    return cv2.resize(pad, (size, size), interpolation=interp)


def to_model_tensor(gray_square):
    """uint8 (S,S) -> float32 torch (3,S,S) with ImageNet norm."""
    rgb = np.stack([gray_square] * 3, axis=-1).astype(np.float32) / 255.0
    rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(rgb).permute(2, 0, 1).contiguous()

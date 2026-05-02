"""Pseudo-sketch from RGB (opencv dodge pipeline, sketch-to-image image2sketch style)."""

from __future__ import annotations

import cv2
import numpy as np
import torch


def _dodge(image: np.ndarray, mask_inv_blur: np.ndarray) -> np.ndarray:
    return cv2.divide(image, 255 - mask_inv_blur, scale=256.0)


def photo_bgr_uint8_to_sketch_gray(image_bgr_u8: np.ndarray, blur_ksize: int = 21) -> np.ndarray:
    gray = cv2.cvtColor(image_bgr_u8, cv2.COLOR_BGR2GRAY)
    inv = 255 - gray
    k = blur_ksize | 1
    blur = cv2.GaussianBlur(inv, (k, k), sigmaX=0)
    dodge = np.clip(_dodge(gray.astype(np.float32), blur.astype(np.float32)), 0, 255)
    return dodge.astype(np.uint8)


def sketch_gray_uint8_to_tensor01(sk_uint8: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(sk_uint8).float().unsqueeze(0) / 255.0

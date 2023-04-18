import albumentations as A
import numpy as np


class Dropout(A.PixelDropout):
    def apply_to_bbox(self, bbox, **params):
        return bbox

    def apply_to_keypoint(self, keypoint, **params):
        return keypoint

    def apply_to_mask(self, img: np.ndarray, **params) -> np.ndarray:
        return img


class CoarseDropout(A.CoarseDropout):
    def apply_to_bbox(self, bbox, **params):
        return bbox

    def apply_to_keypoint(self, keypoint, **params):
        return keypoint

    def apply_to_mask(self, img: np.ndarray, **params) -> np.ndarray:
        return img

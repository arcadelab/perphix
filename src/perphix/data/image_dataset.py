"""A very simple dataset class for loading individual images from a dataset container.

This is useful for training models for spatial tasks, such as segmentation, where
the output should be independent of the image sequence.

"""
from __future__ import annotations
import time
from typing import Any, Optional, Union, TypeVar
from datetime import datetime
from pathlib import Path
from PIL import Image
import cv2
import pycocotools.mask as mask_util

import logging
import numpy as np


from .dataset import PerphixContainer, PerphixDataset
from ..augment import build_augmentation
from .. import utils

log = logging.getLogger(__name__)


class PerphixImageDataset(PerphixContainer):
    """Dataset for individual images in the whole dataset.

    This class is useful for training models for spatial tasks, such as segmentation, where
    the output should be independent of the image sequence.

    Users may wish to use this as a model for more complicated dataset, for instance with an
    augmentation pipeline.

    """

    def __init__(
        self,
        datasets: list[PerphixDataset],
        train: bool = True,
        image_size: int = 256,
        fliph: bool = False,
    ):
        """Initialize the dataset.

        Args:
            datasets (list[PerphixDataset]): The datasets to use.
            train (bool, optional): Whether this is a training dataset. Defaults to True.
            image_size (int, optional): The size of the images to return. Defaults to 256.
            fliph (bool, optional): Whether to flip the images horizontally. This is not an augmentation
                strategy but rather a correction if the images were not properly flipped. Defaults to False.

        """
        super().__init__(datasets)

        self.train = train
        self.image_size = image_size
        self.fliph = fliph

    def __getitem__(self, index: int) -> tuple[dict[str, Any], dict[str, Any]]:
        """Get the image and label corresponding to the given index.

        Args:
            index (int): The index into the images.

        Returns:
            tuple[dict[str, Any], np.ndarray]: The image dictionary and label.

        """
        image_dict, annotation_dicts, dataset, image_id = self.get_image_full(index)

        image: np.ndarray = cv2.imread(str(image_dict["path"]))
        category_ids, keypoints, masks, bboxes = self.decode_annotations(
            image_dict, annotation_dicts
        )
        transform = build_augmentation(
            is_train=self.train, resize=self.image_size, use_keypoint=True
        )

        transformed = transform(
            image=image,
            bboxes=bboxes,
            category_ids=category_ids,
            keypoints=keypoints,
            masks=masks,
        )

        image = transformed["image"]
        bboxes = transformed["bboxes"]
        category_ids = transformed["category_ids"]
        keypoints = transformed["keypoints"]
        masks = transformed["masks"]

        H, W = image.shape[:2]

        # Make segmentation mask
        segs = np.zeros((len(self.categories), H, W), dtype=bool)
        for mask, cat_id in zip(masks, category_ids):
            label = dataset.label_from_category_id[cat_id]
            segs[label] = np.logical_or(segs[label], mask)

        # Make keypoint heatmaps
        keypoints = np.array(keypoints).reshape(-1, 2)
        heatmaps = np.zeros((len(dataset.label_from_keypoint), H, W), dtype=np.float32)
        out_keypoints = np.ones((len(dataset.label_from_keypoint), 2), dtype=np.float32) * -1
        for label, (x, y) in zip(dataset.label_from_keypoint, keypoints):
            if not (0 <= x < W and 0 <= y < H):
                continue
            out_keypoints[label] = (x, y)
            heatmaps[label] = utils.heatmap(x, y, min(H, W) / 64, (H, W))

        if self.fliph:
            # Not for augmentation.
            image = image[:, ::-1]
            segs = segs[:, :, ::-1]
            heatmaps = heatmaps[:, :, ::-1]
            out_keypoints[:, 0] = W - out_keypoints[:, 0]

        # Cast to types/shapes expected by pytorch.
        image = image.transpose(2, 0, 1).astype(np.float32)
        segs = segs.astype(np.float32)
        heatmaps = heatmaps.astype(np.float32)
        keypoints = out_keypoints.astype(np.float32)

        inputs = dict(
            image=image,
            image_id=int(image_id),
        )

        targets = dict(
            segs=segs,
            heatmap=heatmaps,
            keypoints=keypoints,
        )

        return inputs, targets

    def __len__(self) -> int:
        """Get the number of images in the dataset.

        Returns:
            int: The number of images in the dataset.

        """
        return self.num_images

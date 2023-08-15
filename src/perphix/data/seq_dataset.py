from __future__ import annotations
import time
from typing import Any, Optional, Union, TypeVar
from datetime import datetime
from pathlib import Path
import sys
import torch
from PIL import Image

import logging
import numpy as np
from deepdrr.utils.data_utils import load_json, save_json
from rich.progress import track
import pandas as pd
from collections import Counter
from deepdrr.utils import listify
from functools import reduce
import cv2
from pycocotools import mask as mask_util

from .dataset import PerphixContainer, PerphixDataset
from ..augment import build_augmentation
from .base import PerphixBase
from .. import utils

log = logging.getLogger(__name__)


class PerphixSequenceDataset(PerphixContainer):
    """Dataset for procedures in the whole dataset."""

    def __init__(
        self,
        datasets: list[PerphixDataset],
        seq_len: int = 300,
        train: bool = True,
        image_size: int = 256,
        fliph: bool = False,
        overlap: float = 0,
        triplets: bool = False,
    ):
        """Create a new dataset for procedures in the whole dataset.

        Args:
            datasets: List of datasets to use.
            seq_len: Length of the sequences to use.
            train: Whether to use the training or validation set.
            image_size: Size of the images to use.
            fliph: Whether to flip the images horizontally.
            overlap: Amount of overlap between sequences, as a fraction of seq_len.
            triplets: Whether to provide 3-frame inputs instead of 1-frame inputs.
                The "current" frame is the last frame.

        """
        super().__init__(datasets)
        self.seq_len = seq_len
        self.train = train
        self.image_size = image_size
        self.fliph = fliph
        self.overlap = 0 if train else int(self.seq_len * overlap)
        self.triplets = triplets

        self.process_batches()

    def process_batches(self):
        self.samples = []  # sample_idx -> (procedure_idx, start_idx)
        step = self.seq_len - self.overlap
        for dataset in self.datasets:
            for first_frame_id, image_ids in dataset.procedures.items():
                procedure_idx = dataset.procedure_idx_from_first_frame_id[first_frame_id]
                for i in range(0, len(image_ids), step):
                    self.samples.append((procedure_idx, i))

    def __getitem__(self, index: int) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Get the embeddings for a procedure.

        Does not perform any truncation or padding.

        If the procedure is cached in the cache directory, it will be loaded from there.

        Args:
            procedure_idx: Index of the procedure to get the embeddings for.
            train: Whether to get the embeddings for the training or validation set.

        Returns:
            instance_features: Numpy array of shape (S, E) where S is the number of images in the
                procedure/sequence and E is the dimensionality of the embedding.
            labels: Numpy array of shape (S, num_supercategories) containing the labels for the procedure.

        """
        procedure_idx, start_idx = self.samples[index]

        (
            image_dicts,
            annotation_dicts,  # (S, num_annotations[s])
            labels,  # (S, num_supercategories)
            dataset,
            procedure_idx_in_dataset,
        ) = self.get_procedure_full(procedure_idx)

        image_indices = np.arange(len(image_dicts), dtype=int)
        seq_len = len(image_dicts)

        src_key_padding_mask = np.zeros(self.seq_len, dtype=bool)
        if seq_len > self.seq_len:
            start_idx = min(start_idx, seq_len - self.seq_len)
            image_dicts = image_dicts[start_idx : start_idx + self.seq_len]
            annotation_dicts = annotation_dicts[start_idx : start_idx + self.seq_len]
            image_indices = image_indices[start_idx : start_idx + self.seq_len]
            labels = labels[start_idx : start_idx + self.seq_len]

        seq_len = len(image_dicts)
        if seq_len < self.seq_len:
            # Pad with last image and update the mask
            image_dicts = image_dicts + [
                image_dicts[-1] for _ in range(self.seq_len - len(image_dicts))
            ]
            annotation_dicts = annotation_dicts + [
                [] for _ in range(self.seq_len - len(annotation_dicts))
            ]
            image_indices = np.concatenate(
                [image_indices, np.zeros(self.seq_len - len(image_indices), int)], axis=0
            )
            labels = np.concatenate(
                [labels, np.zeros((self.seq_len - len(labels), labels.shape[1]))], axis=0
            )
            src_key_padding_mask[seq_len:] = True

        # Get the annotation masks and keypoint list
        # Augment all the images in the sequence with the same augmentation
        transform = build_augmentation(
            is_train=self.train, resize=self.image_size, use_keypoint=True
        )

        seq_images = []
        seq_masks = []
        seq_heatmaps = []
        seq_keypoints = []
        for anno_dicts, image_dict in zip(annotation_dicts, image_dicts):
            image = np.array(cv2.imread(str(image_dict["path"])))
            category_ids, keypoints, masks, bboxes = self.decode_annotations(image_dict, anno_dicts)
            transformed = transform(
                image=image,
                masks=masks,
                keypoints=keypoints,
                category_ids=category_ids,
                bboxes=bboxes,
            )
            image = transformed["image"]
            masks = transformed["masks"]
            keypoints = transformed["keypoints"]
            category_ids = transformed["category_ids"]

            # Make and append the image for the sequence
            H, W = image.shape[:2]
            seq_images.append(image)

            # Make and append the masks for the sequence
            seq_mask = np.zeros((self.num_categories, H, W), dtype=bool)
            for mask, cat_id in zip(masks, category_ids):
                label = dataset.label_from_category_id[cat_id]
                seq_mask[label] = np.logical_or(seq_mask[label], mask)
            seq_masks.append(seq_mask.astype(np.float32))

            # Make and append the heatmaps for the sequence
            keypoints = np.array(keypoints).reshape(-1, 2)
            seq_heatmap = np.zeros((len(dataset.label_from_keypoint), H, W), dtype=np.float32)
            out_keypoints = np.ones((len(dataset.label_from_keypoint), 2), dtype=np.float32) * -1
            for label, (x, y) in enumerate(keypoints):
                if not (0 <= x < W and 0 <= y < H):
                    continue

                out_keypoints[label] = (x, y)
                seq_heatmap[label] = utils.heatmap(x, y, H / 64, (H, W))

            seq_keypoints.append(out_keypoints)
            seq_heatmaps.append(seq_heatmap)

        seq_images = np.array(seq_images)  # (seq_len, H, W, 3)
        seq_masks = np.array(seq_masks)  # (seq_len, num_classes, H, W)
        seq_heatmaps = np.array(seq_heatmaps)  # (seq_len, num_keypoints, H, W)
        seq_keypoints = np.array(seq_keypoints)  # (seq_len, num_keypoints, 2)

        if self.fliph:
            seq_images = seq_images[:, :, ::-1, :].copy()
            seq_masks = seq_masks[:, :, :, ::-1].copy()
            seq_heatmaps = seq_heatmaps[:, :, :, ::-1].copy()

            H, W = seq_images.shape[1:3]
            seq_keypoints[:, :, 0] = W - seq_keypoints[:, :, 0]

        if self.triplets:
            for s in range(seq_len):
                if s > 1:
                    prev_prev_image = seq_images[s - 2, :, :, 2].copy()
                else:
                    prev_prev_image = np.zeros_like(seq_images[0, :, :, 0])

                if s > 0:
                    prev_image = seq_images[s - 1, :, :, 2].copy()
                else:
                    prev_image = np.zeros_like(seq_images[0, :, :, 0])

                seq_images[s, :, :, 0] = prev_prev_image
                seq_images[s, :, :, 1] = prev_image

        # Cast to tensors. Maybe not necessary.
        images = torch.tensor(seq_images).permute(0, 3, 1, 2).float()
        masks = torch.tensor(seq_masks).float()
        heatmaps = torch.tensor(seq_heatmaps).float()
        labels = torch.tensor(labels).long()
        src_key_padding_mask = torch.tensor(src_key_padding_mask).bool()
        keypoints = torch.tensor(seq_keypoints).float()

        inputs = dict(
            images=images,
            src_key_padding_mask=src_key_padding_mask,
            procedure_idx=torch.tensor(procedure_idx_in_dataset).long(),
            image_ids=np.array([image_dict["id"] for image_dict in image_dicts]),
        )

        targets = dict(
            labels=labels,
            masks=masks,
            heatmaps=heatmaps,
            keypoints=keypoints,
        )

        # log.debug(f"inputs: {inputs}")
        # log.debug(f"targets: {targets}")

        return inputs, targets

    def __len__(self):
        return len(self.samples)

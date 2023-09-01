"""Contains the PerphixDataset and PerphixContainer classes for reading saved datasets.

PerphixDataset is the main class for reading saved datasets. It is used to load an existing dataset
from a single annotation file.

PerphixContainer is a container for multiple PerphixDatasets. It is used to merge multiple datasets
into one.

Note that neither of these datasets are used for training or inference. They are used to load
existing datasets for analysis or visualization. They may be subclassed to create new datasets
with __getitem__ and __len__ methods for training or inference, depending on the goal.

"""
from __future__ import annotations
import time
from typing import Any, Optional, Union, TypeVar
from datetime import datetime
from pathlib import Path
import sys
from PIL import Image
import cv2
import seaborn as sns
from deepdrr.utils import image_utils, data_utils
from hydra.utils import get_original_cwd

import logging
import numpy as np
from rich.progress import track
import csv
import json
import pandas as pd
from collections import Counter
from functools import reduce
import operator

from .base import PerphixBase
from ..utils import load_json, save_json, vis_utils

log = logging.getLogger(__name__)


def one_hot(x: np.ndarray, num_classes: int) -> np.ndarray:
    """One-hot encode an array.

    Args:
        x (np.ndarray): (n,) Array to encode.
        num_classes (int): Number of classes.

    Returns:
        np.ndarray: (n, num_classes) One-hot encoded array.

    """
    return np.eye(num_classes, dtype=int)[x]


def count_consecutive_elements(x: list[Any]) -> list[int]:
    """Count the number of consecutive elements in a list.

    Returns:
        List[int]: List of counts.

    """
    for i, v in enumerate(x):
        if i == 0:
            counts = [1]
        elif v == x[i - 1]:
            counts[-1] += 1
        else:
            counts.append(1)

    return counts


def get_seq_lengths(first_frame_ids: list[int]) -> np.ndarray:
    """Get the sequence lengths from the first frame ids.

    Args:
        first_frame_ids (list[int]): List of first frame ids.

    Returns:
        list[int]: List of sequence lengths, for each frame id.

    """
    counts = count_consecutive_elements(first_frame_ids)
    seq_lengths = np.zeros(len(first_frame_ids), dtype=int)
    total = 0
    for c in counts:
        seq_lengths[total : total + c] = c
        total += c

    return seq_lengths


T = TypeVar("T")


def merge(*datasets: PerphixDataset) -> PerphixContainer:
    """Merge multiple datasets into one."""
    if len(datasets) == 1:
        return PerphixContainer([datasets[0]])
    return reduce(operator.or_, datasets)


class PerphixDataset(PerphixBase):
    """Loads an existing dataset from a single annotation file.

    In general, PerphixDataset is preferred over PerphixInstance, as it allows for merging multiple
    datasets into one.

    Attributes:
        coco (dict[str, Any]): The complete dict containing the dataset annotations.
        image_dir (Path): Path to the directory containing the images.
        name (Optional[str], optional): Name of the dataset.
        info (dict[str, Any]): The info dict.
        licenses (list[dict[str, Any]]): List of license dicts.
        categories (dict[int, [dict[str, Any]]]): Mapping from category id to category dict.
        super_categories (dict[int, set(int)]): Mapping from super category id to set of category ids in that supercategory.
        images (dict[int, dict[str, Any]]): Mapping from image id to image dict.
        segmentations (dict[int, list[dict[str, Any]]]): Mapping from image id to list of segmentations in that image.
        sequence_categories (dict[int, list[int]]): Mapping from sequence category id to the dict for that sequence category.
        sequence_super_categories (dict[str, set(int)]): Mapping from sequence super category name to set of sequence category ids in that supercategory.
        sequences (dict[int, dict[int, int]]): Maping from image_id to a dict mapping from the sequence super category to the sequence id.
        procedures (dict[int, list[int]]): Mapping from the first_frame_id to the list of image ids in that procedure sequence.
        first_frame_ids (list[int]): List of first frame ids.

    """

    # try fetching logos
    try:
        _jhu_logo_path = data_utils.download(
            "https://benjamindkilleen.com/files/jhu_logo_black_bg.png",
            root="/tmp",
            filename="jhu_logo_black_bg.png",
        )
        _arcade_logo_path = data_utils.download(
            "https://benjamindkilleen.com/files/arcade_logo_black.png",
            root="/tmp",
            filename="arcade_logo_black.png",
        )
    except Exception as e:
        log.exception(e)
        _jhu_logo_path = None
        _arcade_logo_path = None

    def __init__(
        self,
        annotation: dict[str, Any],
        image_dir: Path,
        name: Optional[str] = None,
        palette: str = "hls",  # "Spectral"
    ):
        """The dataset.

        Args:
            annotation (dict[str, Any]): The complete dict containing the dataset annotations.
            image_dir (Path): Path to the directory containing the images.
            name (Optional[str], optional): Name of the dataset.
        """
        self.annotation = annotation
        self.image_dir = Path(image_dir).expanduser()
        self.name = self.image_dir.name if name is None else name

        self.process_info()
        self.process_licenses()
        self.process_categories()
        self.process_images()
        self.process_segmentations()
        self.process_sequence_categories()
        self.process_sequences()

        self.keypoint_colors = np.array(sns.color_palette(palette, self.num_keypoints))
        self.keypoint_colors = self.keypoint_colors[
            np.random.permutation(len(self.keypoint_colors))
        ]

        segmentation_colors = np.array(sns.color_palette(palette, len(self.categories)))
        segmentation_colors = segmentation_colors[np.random.permutation(len(segmentation_colors))]
        self.segmentation_colors = dict()
        for i, cat in enumerate(self.categories.values()):
            self.segmentation_colors[cat["id"]] = segmentation_colors[i]

        sequence_colors = np.array(sns.color_palette(palette, len(self.seq_categories)))
        sequence_colors = sequence_colors[np.random.permutation(len(sequence_colors))]
        self.sequence_colors = dict()
        for i, cat in enumerate(self.seq_categories):
            self.sequence_colors[cat["id"]] = sequence_colors[i]

        # TODO: check if sequence IDs consist of consecutive labels starting at 1, and if they do not, create a mapping
        # The 0 label is reserved for "no sequence" of that type.

    def save(self, path: Optional[Path] = None):
        """Save the annotation file.

        If the path is not provided, the annotation file is saved as `self.image_dir.parent / "annotations" / f"{self.name}.json"`.

        If that file already exists, a timestamp is appended to the file name.

        Args:
            path (Optional[Path], optional): Path to save the annotation file. Defaults to None.
        """

        if path is None:
            path = self.image_dir.parent / "annotations" / f"{self.name}.json"

        if path.exists():
            log.warning(f"File {path} already exists. Appending timestamp to file name.")
            path = (
                path.parent / f"{path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{path.suffix}"
            )

        log.info(f"Saving dataset to {path}...")
        t = time.time()
        with open(path, "w") as f:
            json.dump(self.annotation, f)
        log.info(f"Saved {path} in {time.time() - t:.2f} seconds.")

    def add_image(
            self, 
            image_path: Path, 
            case_name: str = "",
            standard_view_angles: dict[str, float] = {}, frame_id: int = -1, first_frame_id: int = 0,):
        """Add an image to the dataset."""
        image_id = len(self.images)
        image = cv2.imread(str(image_path))

        image_info = {
            "id": image_id,
            "file_name": image_path.name,
            "height": image.shape[0],
            "width": image.shape[1],
            "date_captured": datetime.now().isoformat(),
            "license": 0,
            "frame_id": frame_id,
            "seq_length": -1,  # has to be fixed later
            "first_frame_id": first_frame_id,
            "case_name": case_name, # name of the corresponding CT, for obtaining ground truth?

        }

        raise NotImplementedError("TODO: add image to dataset. Also add annotations.")

        if image_id in self.images:
                log.error("Skipping duplicate image id: {}".format(image_info))
            else:
                image_info["path"] = str(self.image_dir / image_info["file_name"])
                self.images[image_id] = image_info

        self.annotation["images"].append(
            {
                "license": 0,
                "file_name": image_path.name,
                "height": image.shape[0],
                "width": image.shape[1],
                "date_captured": datetime.datetime.now().isoformat(),
                "id": image_id,  # to be changed
                "frame_id": image_id,
                "seq_length": -1,  # to be changed
                "first_frame_id": 0,  # to be changed
                "case_name": case_name, # name of the corresponding CT, for obtaining ground truth?
                "standard_view_angles": standard_view_angles,
            }
        )

    def add_image_sequence(self, first_frame_id: int, seq_length: int):
        # Adds an image sequence, modifying the necessary images.

        raise NotImplementedError
    
    def add_sequence(self):
        # Adds a sequence annotation

        raise NotImplementedError

    def as_keypoints(self) -> PerphixDataset:
        return PerphixDataset(self.pelvis_only(self.annotation), self.image_dir, self.name)

    def __or__(self, other: Union[PerphixDataset, PerphixContainer]) -> PerphixContainer:
        """Merge two datasets. The second dataset is appended to the first one.

        This has to return a whole new class to contain a set of these things and check all the
        internals, without re-indexing everything. The container maintains the ID offsets without
        altering the contained datasets.

        The image_ids are shifted by the number of images in the first dataset.

        """
        if isinstance(other, PerphixDataset):
            return PerphixContainer([self, other])
        elif isinstance(other, PerphixContainer):
            return PerphixContainer([self] + other.datasets)
        else:
            raise TypeError("Can only merge with another dataset or container.")

    @classmethod
    def load(cls, annotation_path: Path, image_dir: Path, name: Optional[str] = None, **kwargs):
        # log.info(f"Loading dataset from {annotation_path}...")
        t = time.time()
        coco = load_json(Path(annotation_path).expanduser())
        log.info(f"Loaded {annotation_path} in {time.time() - t:.2f} seconds.")
        return cls(coco, image_dir, name, **kwargs)

    @property
    def num_procedures(self) -> int:
        """Number of procedures in the dataset.

        Returns:
            int: Number of procedures in the dataset.

        """
        return len(self.procedures)

    @property
    def num_images(self) -> int:
        """Number of images in the dataset.

        Returns:
            int: Number of images in the dataset.

        """
        return len(self.images)

    def get_procedure_image_ids(self, procedure_idx: int) -> list[int]:
        """Get the image ids for a given procedure index.

        Args:
            procedure_idx (int): The procedure index.

        Returns:
            list[int]: List of image ids for the procedure.

        """
        return self.procedures[self.first_frame_ids[procedure_idx]]

    def get_procedure(self, procedure_idx: int) -> tuple[list[dict[str, Any]], np.ndarray]:
        """Get the procedure for a given first frame id.

        Args:
            procedure_idx (int): The procedure index. Overrides the first frame id if provided.

        Returns:
            list[dict]: The S-list of image dicts for the procedure.
            np.ndarray: (S, num_supercategories) labels for the procedure.

        """
        first_frame_id = self.first_frame_ids[procedure_idx]

        image_dicts = []
        labels = []
        # labels = np.zeros((procedure_length, len(self.sequence_super_categories)), dtype=int)
        for image_id in self.procedures[first_frame_id]:
            if image_id not in self.images or image_id not in self.sequences:
                log.warning(f"Image {image_id} not annotated with a sequence. Skipping...")
                continue
            image_dict = self.images[image_id]

            # Mapping from supercat_name to seq_catid
            sequences = self.sequences[image_id]
            label = self.get_label_from_sequences(sequences)

            image_dicts.append(image_dict)
            labels.append(label)

        return image_dicts, np.array(labels)

    def get_procedure_annotations(self, procedure_idx: int) -> list[list[dict[str, Any]]]:
        """Get the annotations for a given procedure.

        Should be used in concert with get_procedure().

        Args:
            procedure_idx (int): The procedure index.

        Returns:
            list[list[dict]]: (S, num_annotations[s]) List of annotations for each image in the procedure.
        """
        first_frame_id = self.first_frame_ids[procedure_idx]

        annotations = []
        for image_id in self.procedures[first_frame_id]:
            if image_id not in self.segmentations:
                segmentations = []
            else:
                segmentations = self.segmentations[image_id]
            annotations.append(segmentations)
        return annotations

    def get_label_from_sequences(self, sequences: dict[str, int]) -> np.ndarray:
        """Map the sequence category IDs to an [M,] array of labels.

        Args:
            sequences (dict[str, int]): Mapping from sequence supercategory name to sequence category id.

        Returns:
            np.ndarray: (num_supercategories,) array of labels, in the order of self.sequence_super_categories.

        """
        labels = np.zeros(len(self.sequence_super_categories), dtype=int)
        for i, seq_super_category in enumerate(self.sequence_super_categories):
            if seq_super_category not in sequences:
                log.info(
                    f"Sequence super category '{seq_super_category}' not found in sequences {sequences}."
                )
                labels[i] = 0
            elif (
                sequences[seq_super_category]
                not in self.label_from_seq_category_id[seq_super_category]
            ):
                log.error(
                    f"Sequence category {sequences[seq_super_category]} not found in sequence category to label mapping."
                )
                labels[i] = 0
            else:
                labels[i] = self.label_from_seq_category_id[seq_super_category][
                    sequences[seq_super_category]
                ]

        return labels

    def get_sequences_from_labels(self, labels: np.ndarray) -> dict[str, int]:
        """Map the [M,] array of labels to a dict of sequence category IDs.

        Args:
            labels (np.ndarray): [M,] array of labels, in the order of self.sequence_super_categories.

        Returns:
            dict[str, int]: Mapping from sequence super category name to sequence category id.

        """

        sequences = {}
        for i, seq_super_category in enumerate(self.sequence_super_categories):
            if labels[i] == 0:
                pass
            elif labels[i] not in self.seq_category_id_from_label[seq_super_category]:
                log.error(f"Label {labels[i]} not found in label to sequence category mapping.")
            else:
                sequences[seq_super_category] = self.seq_category_id_from_label[seq_super_category][
                    labels[i]
                ]
        return sequences

    def get_sequence_names(self, sequences: dict[str, int]) -> dict[str, str]:
        """Get the sequence names for a given sequence mapping.

        Args:
            sequences (dict[str, int]): Mapping from sequence super category name to sequence category id.

        Returns:
            dict[str, str]: Mapping from sequence super category name to sequence name.

        """
        sequence_names = {}
        for seq_super_category, seq_category_id in sequences.items():
            sequence_names[seq_super_category] = self.sequence_categories[seq_category_id]["name"]
        return sequence_names

    def get_sequence_names_from_labels(self, labels: np.ndarray) -> dict[str, str]:
        """Get the sequence names for a given label array.

        Args:
            labels (np.ndarray): [M,] array of labels, in the order of self.sequence_super_categories.

        Returns:
            dict[str, str]: Mapping from sequence super category name to sequence name.

        """
        sequences = self.get_sequences_from_labels(labels)
        return self.get_sequence_names(sequences)

    def get_sequence_counts(self) -> dict[str, np.ndarray]:
        """Get the counts for each sequence super category.

        Returns:
            dict[str, ndarray]: Mapping from sequence super category name to counts for each of the classes in that label.
        """
        counts: dict[str, np.ndarray] = {}
        for procedure_idx in range(self.num_procedures):
            image_dicts, labels = self.get_procedure(procedure_idx)
            # labels: (S, num_supercategories)
            labels = labels.transpose(1, 0)  # (num_supercategories, S)

            for i, seq_super_category in enumerate(self.sequence_super_categories):
                num_classes = len(self.sequence_super_categories[seq_super_category]) + 1  # for bg
                if seq_super_category not in counts:
                    counts[seq_super_category] = np.zeros(num_classes, dtype=int)
                counts[seq_super_category] += one_hot(labels[i], num_classes).sum(0)

        return counts

    def process_info(self):
        self.info: dict[str, Any] = self.annotation["info"]

    def process_licenses(self):
        self.licenses: dict[str, Any] = self.annotation["licenses"]

    def process_categories(self):
        self.categories: dict[int, dict[str, Any]] = {}
        self.super_categories: dict[str, set(int)] = {}
        self.label_from_category_id: dict[int, int] = {}
        self.category_id_from_label: dict[int, int] = {}
        self.label_from_keypoint: dict[str, int] = {}
        self.keypoint_from_label: dict[int, str] = {}
        for category in self.annotation["categories"]:
            cat_id = category["id"]
            super_category = category["supercategory"]

            # Add category to the categories dict
            if cat_id not in self.categories:
                self.categories[cat_id] = category
            else:
                log.error("Skipping duplicate category id: {}".format(category))

            # Add category to super_categories dict
            if super_category not in self.super_categories:
                self.super_categories[super_category] = {
                    cat_id
                }  # Create a new set with the category id
            else:
                self.super_categories[super_category] |= {cat_id}  # Add category id to the set

            # Add category to label_from_category_id dict
            if cat_id not in self.label_from_category_id:
                label = len(self.label_from_category_id)
                self.label_from_category_id[cat_id] = label
                self.category_id_from_label[label] = cat_id
            else:
                log.error("Skipping duplicate category id: {}".format(category))

            if category["name"] == "pelvis":
                # Process keypoints
                for i, keypoint in enumerate(category["keypoints"]):
                    self.label_from_keypoint[keypoint] = i
                    self.keypoint_from_label[i] = keypoint

    def process_images(self):
        self.image_ids: dict[int, int] = {}  # image_idx -> image_id
        self.images: dict[int, list[dict[str, Any]]] = {}  # image_id -> image
        self.procedures: dict[int, list[int]] = {}  # first_frame_id -> image_ids
        self.first_frame_ids: list[int] = []  # procedure_idx -> first_frame_id
        self.procedure_idx_from_first_frame_id: dict[
            int, int
        ] = {}  # first_frame_id -> procedure_idx
        for image_idx, image_info in enumerate(self.annotation["images"]):
            image_id = int(image_info["id"])
            self.image_ids[image_idx] = image_id

            if image_id in self.images:
                log.error("Skipping duplicate image id: {}".format(image_info))
            else:
                image_info["path"] = str(self.image_dir / image_info["file_name"])
                self.images[image_id] = image_info

            first_frame_id = image_info["first_frame_id"]
            if first_frame_id not in self.procedures:
                self.procedures[first_frame_id] = []
                procedure_idx = len(self.first_frame_ids)
                self.first_frame_ids.append(first_frame_id)
                self.procedure_idx_from_first_frame_id[first_frame_id] = procedure_idx
            self.procedures[first_frame_id].append(image_id)

    def process_segmentations(self):
        self.segmentations: dict[int, list[dict[str, Any]]] = {}
        for segmentation in self.annotation["annotations"]:
            image_id = segmentation["image_id"]
            if image_id not in self.segmentations:
                self.segmentations[image_id] = []
            self.segmentations[image_id].append(segmentation)

    def process_sequence_categories(self):
        # Relies on ordered dicts for the order of supercategories
        self.sequence_categories: dict[int, list[int]] = {}  # seq_category_id -> dict
        # supercategory -> set of seq_category_ids
        self.sequence_super_categories: dict[str, set(int)] = {}

        # Each value is a mapping from sequence category id to 1-indexed label space (for feeding
        # into a network), and vice versa. The key is the sequence supercategory.
        self.label_from_seq_category_id: dict[str, dict[int, int]] = {}
        self.seq_category_id_from_label: dict[str, dict[int, int]] = {}

        for seq_category in self.annotation["seq_categories"]:
            seq_category_id = seq_category["id"]
            seq_super_category = seq_category["supercategory"]

            # Add category to the categories dict
            if seq_category_id not in self.sequence_categories:
                self.sequence_categories[seq_category_id] = seq_category
            else:
                log.error("Skipping duplicate sequence category id: {}".format(seq_category))

            # Add category to super_categories dict
            if seq_super_category not in self.sequence_super_categories:
                self.sequence_super_categories[seq_super_category] = {seq_category_id}
            else:
                self.sequence_super_categories[seq_super_category] |= {seq_category_id}

            if seq_super_category not in self.label_from_seq_category_id:
                self.label_from_seq_category_id[seq_super_category] = {}
                self.seq_category_id_from_label[seq_super_category] = {}

            # Add category to the label dicts
            if seq_category_id not in self.label_from_seq_category_id[seq_super_category]:
                # So when we add the first category, the label is 1
                label = len(self.sequence_super_categories[seq_super_category])
                self.label_from_seq_category_id[seq_super_category][seq_category_id] = label
                self.seq_category_id_from_label[seq_super_category][label] = seq_category_id
            else:
                log.error(
                    "Skipping duplicate sequence category id in supercategory: {}".format(
                        seq_category
                    )
                )

        # log.debug(f"sequence_super_categories: {self.sequence_super_categories}")
        # log.debug(f"sequence_categories: {self.sequence_categories}")
        # log.debug(f"{self.name} sequence_category_to_label: {self.label_from_seq_category_id}")
        # log.debug(f"{self.name} sequence_category_from_label: {self.seq_category_id_from_label}")

    def process_sequences(self):
        """Make a mapping from image_id to the dict of sequence categories it belongs to, for each sequence supercategory."""
        self.sequences: dict[int, dict[str, int]] = {}
        for sequence in self.annotation["sequences"]:
            first_frame_id = int(sequence["first_frame_id"])
            if first_frame_id not in self.images:
                log.error(
                    f"Skipping sequence {sequence} with first_frame_id {first_frame_id}, {type(first_frame_id)} not in images"
                )
                log.debug(f"images: {self.images.keys()}")
                continue

            seq_super_category = self.sequence_categories[sequence["seq_category_id"]][
                "supercategory"
            ]

            name = self.get_sequence_name(sequence["seq_category_id"])
            image_id = sequence["first_frame_id"]
            if sequence["seq_length"] == 0:
                log.error(
                    f"Skipping sequence {seq_super_category}={name} with first_frame_id {image_id} and length 0"
                )
                continue

            length = 0
            while length < sequence["seq_length"]:
                if image_id not in self.images:
                    log.error(
                        f"Skipping sequence {sequence['id']} with frame_id that is not in images: {sequence} (might be fine)"
                    )
                    image_id += 1
                    continue

                if image_id not in self.sequences:
                    self.sequences[image_id] = {}

                if seq_super_category in self.sequences[image_id]:
                    log.error(
                        f"Skipping double sequence annotation of {image_id} in '{seq_super_category}', which is already"
                        f" '{self.get_sequence_name(self.sequences[image_id][seq_super_category])}', not "
                        f"'{self.get_sequence_name(sequence['seq_category_id'])}'"
                    )
                else:
                    self.sequences[image_id][seq_super_category] = sequence["seq_category_id"]
                image_id += 1
                length += 1

    @classmethod
    def get_annotation_path(
        self, image_dir: Path, annotations_dir: Optional[Path], name: Optional[str]
    ):
        if annotations_dir is None:
            annotations_dir = image_dir / "annotations"
        else:
            annotations_dir = Path(annotations_dir).expanduser()
        if name is None:
            name = image_dir.name
        return annotations_dir / f"{name}.json"

    @classmethod
    def from_image_dir(
        cls,
        image_dir: Path,
        image_names: Optional[list[str]] = None,
        name: Optional[str] = None,
        sequences: list[dict[str, Any]] = [],
        image_ids: Optional[list[int]] = None,
        first_frame_ids: Optional[list[int]] = None,
        annotations_dir: Optional[Path] = None,
        extension: str = "png",
        date_captured: Optional[str] = None,
        use_previous: bool = False,
    ):
        """Create a dataset PelvicWorkflowsReal instance from the image directory.

        TODO: add support for reading DICOM images, translating to PNG, and storing date captured
        from the DICOM metadata, including time.

        Args:
            image_dir (Path): Path to the image directory.
            image_names (list, optional): List of image names to use from image_dir.
                None takes all the images with the extension. Defaults to None.
            annotations_dir (Optional[Path], optional): Path to the annotations directory. Defaults to None.
                In this case, the annotations directory defaults to image_dir.parent / "annotations".
            sequences (list, optional): List of sequences to include in the dataset. Defaults to [].
            procedure_lengths (Optional[list[int]], optional): List of procedure lengths. Defaults to None.
            extension (str, optional): Image file extension. Defaults to "png".
            date_captured (Optional[str], optional): Date captured. Defaults to None.
            image_ids (Optional[list[int]], optional): List of image ids. Defaults to None.
            use_previous: If True, use the previous annotation file if it exists. Defaults to False.


        """
        image_dir = Path(image_dir).expanduser()

        if annotations_dir is None:
            annotations_dir = image_dir.parent / "annotations"
        annotations_dir = Path(annotations_dir).expanduser()

        if not annotations_dir.exists():
            annotations_dir.mkdir(parents=True)
        if name is None:
            name = image_dir.name

        annotation_path = annotations_dir / f"{name}.json"

        if annotation_path.exists() and use_previous:
            log.info(f"Loading previous annotation file: {annotation_path}...")
            return cls.load(annotation_path, image_dir=image_dir, name=name)

        if date_captured is None:
            date_captured = datetime.now().strftime("%Y-%m-%d")

        if image_names is None:
            image_paths = sorted(list(image_dir.glob(f"*.{extension}")))
        else:
            image_paths = [image_dir / image_name for image_name in image_names]

        if image_ids is None:
            image_ids = list(range(len(image_paths)))
        else:
            image_ids = [int(image_id) for image_id in image_ids]

        if first_frame_ids is None:
            first_frame_ids = [0] * len(image_paths)

        first_frame_ids.append(len(image_paths))
        seq_lengths = get_seq_lengths(first_frame_ids)

        annotation = cls.get_base_annotation()
        for i, image_path in enumerate(track(image_paths, description="Reading images...")):
            img = np.array(Image.open(image_path))
            image = {
                "license": 3,
                "file_name": image_path.name,
                "height": img.shape[0],
                "width": img.shape[1],
                "date_captured": date_captured,
                "id": image_ids[i],
                "first_frame_id": first_frame_ids[i],
                "frame_id": image_ids[i] - first_frame_ids[i],
                "seq_length": seq_lengths[i],
            }
            annotation["images"].append(image)

        annotation["sequences"] = sequences

        log.info(f"Saving annotation to {annotation_path}...")
        save_json(annotation_path, annotation)

        return cls(annotation, image_dir=image_dir, name=name)

    @classmethod
    def from_csv(
        cls,
        image_dir: Path,
        csv_path: Path,
        annotations_dir: Optional[Path] = None,
        extension: str = "png",
        name: Optional[str] = None,
        use_previous: bool = False,
    ):
        """Create a dataset PelvicWorkflowsReal instance from a CSV file for a SINGLE procedure.

        Args:
            image_dir (Path): Path to the image directory.
            csv_path (Path): Path to the CSV file.
            annotations_dir (Optional[Path], optional): Path to the annotations directory. Defaults to None.
                In this case, the annotations directory defaults to image_dir.parent / "annotations".

        """
        image_dir = Path(image_dir).expanduser()

        annotation_path = cls.get_annotation_path(image_dir, annotations_dir, name)
        if annotation_path.exists() and use_previous:
            log.info(f"Loading annotation from {annotation_path}...")
            return cls.load(annotation_path, image_dir=image_dir, name=name)

        df = pd.read_csv(csv_path)
        image_paths = sorted(list(image_dir.glob(f"*.{extension}")))
        sequences: list[dict] = []
        prev_seq_category_names = None
        seq_lengths = [0, 0, 0, 0]
        image_ids = []
        image_names = []
        for image_path in image_paths:
            # TODO: more robust way to get image_id from image_path
            image_id = int(image_path.stem.split()[0])

            # get the row in the df with this image_id
            row = df.loc[df["Frame Number"] == image_id]
            if row.empty:
                log.info(f"Could not find row in CSV for image {image_id} with path {image_path}")
                continue

            image_ids.append(image_id)
            image_names.append(image_path.name)
            seq_category_names = cls.process_row(row)

            if prev_seq_category_names is None:
                prev_seq_category_names = seq_category_names
                first_frame_ids = [image_id, image_id, image_id, image_id]
                seq_lengths = [1, 1, 1, 1]
                continue

            for i, supercategory in enumerate(["task", "activity", "acquisition", "frame"]):
                if seq_category_names[i] != prev_seq_category_names[i]:
                    seq = {
                        "id": len(sequences),
                        "seq_length": seq_lengths[i],
                        "first_frame_id": first_frame_ids[i],
                        "seq_category_id": cls.get_sequence_catid(
                            supercategory, prev_seq_category_names[i]
                        ),
                    }
                    sequences.append(seq)
                    first_frame_ids[i] = image_id
                    seq_lengths[i] = 0

                seq_lengths[i] += 1
                prev_seq_category_names[i] = seq_category_names[i]

        # add the last sequence
        for i, supercategory in enumerate(["task", "activity", "acquisition", "frame"]):
            seq = {
                "id": len(sequences),
                "seq_length": seq_lengths[i],
                "first_frame_id": first_frame_ids[i],
                "seq_category_id": cls.get_sequence_catid(supercategory, seq_category_names[i]),
            }
            sequences.append(seq)

        return cls.from_image_dir(
            image_dir,
            image_names=image_names,
            name=name,
            sequences=sequences,
            image_ids=image_ids,
            annotations_dir=annotations_dir,
            extension=extension,
            use_previous=False,
        )

    @classmethod
    def process_row(self, row: pd.DataFrame) -> list[str]:
        """Get the sequence category names from the given row."""
        task_column_names = ["Task", "task", "Corridor", "corridor"]
        activity_column_names = ["Activity", "activity"]
        acquisition_column_names = ["Acquisition", "acquisition", "View", "view"]
        frame_column_names = ["Frame", "frame"]

        for column_name in task_column_names:
            if column_name in row.columns:
                task = row[column_name].values[0]
                break
        else:
            raise ValueError(f"Could not find task column in row {row}")

        for column_name in activity_column_names:
            if column_name in row.columns:
                activity = row[column_name].values[0]
                break
        else:
            raise ValueError(f"Could not find activity column in row {row}")

        for column_name in acquisition_column_names:
            if column_name in row.columns:
                acquisition = row[column_name].values[0]
                break
        else:
            raise ValueError(f"Could not find acquisition column in row {row}")

        for column_name in frame_column_names:
            if column_name in row.columns:
                frame = row[column_name].values[0]
                break
        else:
            raise ValueError(f"Could not find frame column in row {row}")

        if acquisition == "oo":
            acquisition = "oblique_left" if "right" in task else "oblique_right"
            # acquisition = "oblique_left" if "left" in task else "oblique_right"
        elif acquisition == "io":
            # flipped
            # acquisition = "oblique_right" if "left" in task else "oblique_left"
            acquisition = "oblique_right" if "right" in task else "oblique_left"
        elif acquisition == "oblique_inlet":
            acquisition = "inlet"

        return [task, activity, acquisition, frame]

    def visualize_image(
        self,
        image_id: int,
        scale: float = 1.5,
        show_annotations: bool = True,
        show_phases: bool = True,
        show_logos: bool = True,
    ) -> np.ndarray:
        """Get a visualisation of the image corresponding to the given image index.

        Tile the image in a 2x2 grid, with the original in the upper left.
        - Keypoints are shown in the upper right.
        - Anatomy/tool masks are shown in the lower left.
        - Corridor masks are shown in the lower right.
        - Underneath the images, include text for the phase. Gray out the ones that are not active?

        Args:
            image_id (int): Image id in the dataset.
            scale (float): The scale to use for the images.
            show_annotations (bool): Whether to show the annotations.
            show_phases (bool): Whether to show the phases on the right panel.
                Set to false if sequences is empty.

        Returns:
            image_vis: (H, W, 3) uint8 image with annotations shown.

        """
        image_info = self.images[image_id]
        annos = self.segmentations[image_id]
        category_ids, keypoints, masks, bboxes = self.decode_annotations(image_info, annos)

        image = np.array(cv2.imread(str(image_info["path"])))

        frame_num = image_info["id"] - image_info["first_frame_id"]

        # TODO: scale the image and annotations

        if show_annotations:
            keypoints_vis = vis_utils.draw_keypoints(
                image, keypoints, names=self.keypoint_pretty_names, colors=self.keypoint_colors
            )

            corridor_category_ids, corridor_masks, corridor_colors = [], [], []
            anatomy_category_ids, anatomy_masks, anatomy_colors = [], [], []
            for category_id, mask in zip(category_ids, masks):
                if category_id in [9]:
                    # Skip pelvis
                    continue
                elif category_id in self.super_categories["corridor"]:
                    corridor_category_ids.append(category_id)
                    corridor_masks.append(mask)
                    corridor_colors.append(self.segmentation_colors[category_id])
                else:
                    anatomy_category_ids.append(category_id)
                    anatomy_masks.append(mask)
                    anatomy_colors.append(self.segmentation_colors[category_id])

            corridor_masks = np.array(corridor_masks)
            anatomy_masks = np.array(anatomy_masks)
            corridor_names = [
                self.get_annotation_pretty_name(catid) for catid in corridor_category_ids
            ]
            anatomy_names = [
                self.get_annotation_pretty_name(catid) for catid in anatomy_category_ids
            ]

            corridor_vis = vis_utils.draw_masks(
                image,
                corridor_masks,
                names=corridor_names,
                colors=corridor_colors,
            )

            anatomy_vis = vis_utils.draw_masks(
                image,
                anatomy_masks,
                names=anatomy_names,
                colors=anatomy_colors,
            )

            image_vis = np.concatenate(
                [
                    np.concatenate([image, keypoints_vis], axis=1),
                    np.concatenate([anatomy_vis, corridor_vis], axis=1),
                ],
                axis=0,
            )

        else:
            image_vis = cv2.resize(image, (image.shape[0] * 2, image.shape[1] * 2))

        image_vis = cv2.resize(image_vis, (0, 0), fx=scale, fy=scale)

        side_panel = np.zeros_like(image_vis)

        h, w = image_vis.shape[:2]
        step = h // 10
        text_scale = 1
        thickness = 2
        sep = w // 2
        offset = step // 2
        text_color = (255, 255, 255)

        # Title
        side_panel = cv2.putText(
            side_panel,
            f"Pelphix Sim",
            (offset, step * 2 - step // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            2 * text_scale,
            text_color,
            thickness,
            cv2.LINE_AA,
        )

        def put_phase(
            side_panel_: np.ndarray, label: str, phase: str, row: int, color: list[int]
        ) -> np.ndarray:
            side_panel_ = cv2.putText(
                side_panel_,
                f"{label}",
                (offset, step * row - step // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                text_scale,
                text_color,
                thickness,
                cv2.LINE_AA,
            )
            side_panel_ = cv2.putText(
                side_panel_,
                phase.capitalize().replace("_", " "),
                (sep, step * row - step // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                text_scale,
                color,
                thickness,
                cv2.LINE_AA,
            )
            return side_panel_

        if show_phases and image_id in self.sequences:
            sequences = self.sequences[image_id]
            seq_names = self.get_sequence_names(sequences)
            side_panel = put_phase(side_panel, f"{frame_num:03d}", "", 4, text_color)
            side_panel = put_phase(
                side_panel,
                "Corridor:",
                seq_names["task"],
                5,
                (self.sequence_colors[sequences["task"]] * 255).tolist(),
            )
            side_panel = put_phase(
                side_panel,
                "Activity:",
                seq_names["activity"],
                6,
                (self.sequence_colors[sequences["activity"]] * 255).tolist(),
            )
            side_panel = put_phase(
                side_panel,
                "View:",
                seq_names["acquisition"],
                7,
                (self.sequence_colors[sequences["acquisition"]] * 255).tolist(),
            )
            side_panel = put_phase(
                side_panel,
                "Frame:",
                seq_names["frame"],
                8,
                (self.sequence_colors[sequences["frame"]] * 255).tolist(),
            )

        margin = h // 20
        if show_logos and self._arcade_logo_path is not None:
            arcade_logo = cv2.imread(str(self._arcade_logo_path))
            arcade_logo = cv2.cvtColor(arcade_logo, cv2.COLOR_BGR2RGB)
            lh, lw = arcade_logo.shape[:2]
            arcade_h = h // 12
            arcade_w = int(arcade_h * lw / lh)
            arcade_logo = cv2.resize(arcade_logo, (arcade_w, arcade_h))
            x = w - arcade_w - margin
            y = h - arcade_h - margin
            side_panel[y : y + arcade_h, x : x + arcade_w] = arcade_logo

        if show_logos and self._jhu_logo_path is not None:
            jhu_logo = cv2.imread(str(self._jhu_logo_path))
            jhu_logo = cv2.cvtColor(jhu_logo, cv2.COLOR_BGR2RGB)
            lh, lw = jhu_logo.shape[:2]
            jhu_h = arcade_h
            jhu_w = int(jhu_h * lw / lh)
            jhu_logo = cv2.resize(jhu_logo, (jhu_w, jhu_h))
            x = w - arcade_w - margin - jhu_w - margin
            y = h - jhu_h - margin
            side_panel[y : y + jhu_h, x : x + jhu_w] = jhu_logo

        return np.concatenate([image_vis, side_panel], axis=1)

    def visualize_procedure(self, procedure_idx: int, **kwargs) -> np.ndarray:
        """Get the visualizations for every image in the procedure.

        Args:
            procedure_idx (int): The procedure index.

        Returns:
            image_vis: (S, H, W, 3) uint8 images with annotations shown.

        """
        image_ids = self.get_procedure_image_ids(procedure_idx)
        frames = []
        for i, image_id in enumerate(image_ids):
            log.debug(f"Visualizing image {i} / {len(image_ids)}")
            frames.append(self.visualize_image(image_id, **kwargs))

            if i == 0:
                image_utils.save(f"vis_{i}.png", frames[-1])

        frames = np.stack(frames, axis=0)
        return frames


class PerphixContainer(PerphixBase):
    """Container around multiple PelvicWorkflows datasets."""

    def __init__(self, datasets: list[PerphixDataset]):
        self.datasets = datasets
        self.cumulative_num_images = np.cumsum([d.num_images for d in datasets], dtype=int)
        self.cumulative_num_procedures = np.cumsum([d.num_procedures for d in datasets], dtype=int)
        # TODO: go through and check that all the mappings are the same.

    def __or__(self, other: Union[PerphixContainer, PerphixDataset]):
        if isinstance(other, PerphixContainer):
            return PerphixContainer(self.datasets + other.datasets)
        elif isinstance(other, PerphixDataset):
            return PerphixContainer(self.datasets + [other])
        else:
            raise ValueError(f"Cannot add {type(other)} to PelvicWorkflowsContainer.")

    def get_dataset(self, procedure_idx: int) -> PerphixDataset:
        """Get the dataset corresponding to the given procedure index.

        Args:
            procedure_idx (int): The procedure index.

        Returns:
            PelvicWorkflowsDataset: The dataset corresponding to the given procedure index.
        """
        # Smallest index of a dataset such that the last procedures index in that dataset is >= than the procedure index.
        dataset_idx: int = min(np.argwhere(procedure_idx < self.cumulative_num_procedures)[0])
        return self.datasets[dataset_idx]

    def get_procedure_idx_in_dataset(self, procedure_idx: int) -> int:
        """Get the procedure index in the dataset corresponding to the given procedure index.

        Args:
            procedure_idx (int): The procedure index.

        Returns:
            int: The procedure index in the dataset corresponding to the given procedure index.
        """
        dataset_idx: int = min(np.argwhere(procedure_idx < self.cumulative_num_procedures))
        # Have to subtract the cumulative number of procedures in the previous datasets to get the index in the current
        # dataset.
        if dataset_idx == 0:
            procedure_idx_in_dataset = procedure_idx
        else:
            procedure_idx_in_dataset = (
                procedure_idx - self.cumulative_num_procedures[dataset_idx - 1]
            )
        return procedure_idx_in_dataset

    def get_dataset_from_image_idx(self, image_idx: int) -> PerphixDataset:
        """Get the dataset corresponding to the given image index.

        Args:
            image_idx (int): The image index.

        Returns:
            PelvicWorkflowsDataset: The dataset corresponding to the given image index.
        """
        # Smallest index of a dataset such that the last image index in that dataset is >= than the image index.
        dataset_idx: int = min(np.argwhere(image_idx < self.cumulative_num_images)[0])
        return self.datasets[dataset_idx]

    def get_image_id(self, image_idx: int) -> int:
        """Get the image_id in the dataset corresponding to the given image index.

        Args:
            image_idx (int): The image index.

        Returns:
            int: The image index in the dataset corresponding to the given image index.
        """
        dataset_idx: int = min(np.argwhere(image_idx < self.cumulative_num_images)[0])
        # Have to subtract the cumulative number of images in the previous datasets to get the index in the current
        # dataset.
        if dataset_idx == 0:
            image_idx_in_dataset = image_idx
        else:
            image_idx_in_dataset = image_idx - self.cumulative_num_images[dataset_idx - 1]

        image_id = self.datasets[dataset_idx].image_ids[image_idx_in_dataset]
        return image_id

    def get_procedure_image_ids(self, procedure_idx) -> list[int]:
        """Get the image indices (in the container) for the given procedure index."""
        dataset_idx: int = int(min(np.argwhere(procedure_idx < self.cumulative_num_procedures)))
        # log.debug(f"Getting procedure {procedure_idx} from dataset {dataset_idx}.")
        dataset = self.datasets[dataset_idx]
        if dataset_idx == 0:
            procedure_idx_in_dataset = procedure_idx
        else:
            procedure_idx_in_dataset = (
                procedure_idx - self.cumulative_num_procedures[dataset_idx - 1]
            )
        image_ids = dataset.get_procedure_image_ids(procedure_idx_in_dataset)
        return image_ids

    def get_procedure_info(
        self, procedure_idx: int
    ) -> tuple[list[dict[str, Any]], np.ndarray, PerphixDataset, int]:
        """Get the procedure corresponding to the given procedure index.

        Args:
            procedure_idx (int): The procedure index.

        Returns:
            images: The image dictionaries in the annotation file.
            labels: The labels for the procedure.
            dataset_name: The name of the dataset.
            procedure_idx_in_dataset: The procedure index in the dataset.

        """
        dataset_idx: int = int(min(np.argwhere(procedure_idx < self.cumulative_num_procedures)))
        # log.debug(f"Getting procedure {procedure_idx} from dataset {dataset_idx}.")
        dataset = self.datasets[dataset_idx]
        # Have to subtract the cumulative number of procedures in the previous datasets to get the index in the current
        # dataset.
        if dataset_idx == 0:
            procedure_idx_in_dataset = procedure_idx
        else:
            procedure_idx_in_dataset = (
                procedure_idx - self.cumulative_num_procedures[dataset_idx - 1]
            )
        images, labels = dataset.get_procedure(procedure_idx_in_dataset)
        return images, labels, dataset, procedure_idx_in_dataset

    def get_image_info(self, image_idx: int) -> tuple[dict[str, Any], PerphixDataset, int]:
        """Get the info for the given image index.

        Args:
            image_idx (int): The image index in the container.

        Returns:
            image_info: The image info dictionary.
            dataset_name: The name of the dataset.
            image_id: The image id in the dataset (not the image_idx).
        """

        dataset_idx: int = int(min(np.argwhere(image_idx < self.cumulative_num_images)))
        dataset = self.datasets[dataset_idx]
        if dataset_idx == 0:
            image_idx_in_dataset = image_idx
        else:
            image_idx_in_dataset = image_idx - self.cumulative_num_images[dataset_idx - 1]
        image_id = dataset.image_ids[image_idx_in_dataset]
        image_info = dataset.images[image_id]
        return image_info, dataset, image_id

    def get_procedure(self, procedure_idx: int) -> tuple[list[dict[str, Any]], np.ndarray]:
        """Get the procedure corresponding to the given procedure index.

        Args:
            procedure_idx (int): The procedure index.

        Returns:
            images: The image dictionaries in the annotation file.
            labels: The labels for the procedure.
        """
        images, labels, _, _ = self.get_procedure_info(procedure_idx)
        return images, labels

    def get_image(self, image_idx: int) -> dict[str, Any]:
        """Get the image corresponding to the given image index.

        Args:
            image_idx (int): The image index.

        Returns:
            image: The image dictionary.
        """
        image_info, _, _ = self.get_image_info(image_idx)
        return image_info

    def get_image_full(
        self, image_idx: int
    ) -> tuple[dict[str, Any], list[dict], PerphixDataset, int]:
        """Get the image corresponding to the given image index.

        Args:
            image_idx (int): The image index.

        Returns:
            image_info: The image dictionary.
            anns: The list of annotations for the image.
        """
        image_info, dataset, image_id = self.get_image_info(image_idx)
        image_id = image_info["id"]
        if image_id not in dataset.segmentations:
            anns = []
        else:
            anns = dataset.segmentations[image_id]
        return image_info, anns, dataset, image_id

    def get_procedure_full(
        self, procedure_idx: int
    ) -> tuple[list[dict[str, Any]], list[list[dict]], np.ndarray, PerphixDataset, int]:
        """Get the procedure corresponding to the given procedure index.

        Args:
            procedure_idx (int): The procedure index.

        Returns:
            images: The image dictionaries in the annotation file.
            annotations: list of annotations for each image.
            labels: The labels for the procedure.
            dataset_name: The name of the dataset.
            procedure_idx_in_dataset: The procedure index in the dataset.
        """
        dataset = self.get_dataset(procedure_idx)
        images, labels, dataset, procedure_idx_in_dataset = self.get_procedure_info(procedure_idx)
        annotations = dataset.get_procedure_annotations(procedure_idx_in_dataset)
        return images, annotations, labels, dataset, procedure_idx_in_dataset

    def get_label_from_sequences(
        self, sequences: dict[str, int], procedure_idx: Optional[int] = None
    ) -> np.ndarray:
        """Map the sequence category IDs to an [M,] array of labels.

        Args:
            sequences (dict[str, int]): Mapping from sequence super category name to sequence category id.
            procedure_idx (Optional[int], optional): The procedure index. Defaults to None (use the first dataset
                in the container).

        Returns:
            np.ndarray: [M,] array of labels, in the order of self.sequence_super_categories.

        """
        if procedure_idx is None:
            dataset = self.datasets[0]
        else:
            dataset = self.get_dataset(procedure_idx)
        return dataset.get_label_from_sequences(sequences)

    def get_sequences_from_labels(
        self, labels: np.ndarray, procedure_idx: Optional[int] = None
    ) -> dict[str, int]:
        """Map the [M,] array of labels to a dict of sequence category IDs.

        Args:
            labels (np.ndarray): [M,] array of labels, in the order of self.sequence_super_categories.

        Returns:
            dict[str, int]: Mapping from sequence super category name to sequence category id.

        """
        if procedure_idx is None:
            dataset = self.datasets[0]
        else:
            dataset = self.get_dataset(procedure_idx)
        return dataset.get_sequences_from_labels(labels)

    def get_sequence_names(
        self, sequences: dict[str, int], procedure_idx: Optional[int] = None
    ) -> dict[str, str]:
        """Get the sequence names for a given sequence mapping.

        Args:
            sequences (dict[str, int]): Mapping from sequence super category name to sequence category id.

        Returns:
            dict[str, str]: Mapping from sequence super category name to sequence name.

        """
        if procedure_idx is None:
            dataset = self.datasets[0]
        else:
            dataset = self.get_dataset(procedure_idx)

        return dataset.get_sequence_names(sequences)

    def get_sequence_names_from_labels(
        self, labels: np.ndarray, procedure_idx: Optional[int] = None
    ) -> dict[str, str]:
        """Get the sequence names for a given label array.

        Args:
            labels (np.ndarray): [M,] array of labels, in the order of self.sequence_super_categories.

        Returns:
            dict[str, str]: Mapping from sequence super category name to sequence name.

        """
        if procedure_idx is None:
            dataset = self.datasets[0]
        else:
            dataset = self.get_dataset(procedure_idx)
        return dataset.get_sequence_names_from_labels(labels)

    def get_sequence_counts(self) -> dict[str, np.ndarray]:
        """Get the counts for each sequence super category.

        Returns:
            dict[str, ndarray]: The counts for all the datasets.
        """

        counts: dict[str, np.ndarray] = {}
        for dataset in self.datasets:
            for sequence_super_category, sequence_counts in dataset.get_sequence_counts().items():
                if sequence_super_category not in counts:
                    counts[sequence_super_category] = np.zeros_like(sequence_counts)
                counts[sequence_super_category] += sequence_counts

        return counts

    def visualize_image(self, image_idx: int, **kwargs) -> np.ndarray:
        """Get a visualisation of the image corresponding to the given image index.

        Args:
            image_idx (int): Image id in the dataset.
            scale (float): The scale to use for the images.

        Returns:
            image_vis: (H, W, 3) uint8 image with annotations shown.

        """
        dataset = self.get_dataset_from_image_idx(image_idx)
        image_id = self.get_image_id(image_idx)
        return dataset.visualize_image(image_id, **kwargs)

    def visualize_procedure(self, procedure_idx: int, **kwargs) -> np.ndarray:
        """Get the visualizations for every image in the procedure.

        Args:
            procedure_idx (int): The procedure index.

        Returns:
            image_vis: (S, H, W, 3) uint8 images with annotations shown.

        """
        dataset = self.get_dataset(procedure_idx)
        procedure_idx_in_dataset = self.get_procedure_idx_in_dataset(procedure_idx)
        log.debug(f"Visualizing procedure {procedure_idx} in dataset {dataset.name}")
        return dataset.visualize_procedure(procedure_idx_in_dataset, **kwargs)

    @property
    def num_procedures(self):
        return sum([d.num_procedures for d in self.datasets])

    @property
    def num_images(self):
        return sum([d.num_images for d in self.datasets])

    @property
    def name(self):
        return "+".join([d.name for d in self.datasets])

    @classmethod
    def from_configs(
        cls,
        configs: list[dict[str, Any]],
        **kwargs,
    ):
        datasets = []
        for config in configs:
            dataset = getattr(PerphixDataset, config["loader"])(**config["config"])
            datasets.append(dataset)
        return cls(datasets, **kwargs)

    @classmethod
    def load(
        cls,
        annotation_path: Union[Path, list[Path]],
        image_dir: Union[Path, list[Path]],
        name: None | str | list[str] = None,
        **kwargs,
    ):
        """Load a dataset or collection of datasets.

        Args:
            annotation_path (Union[Path, list[Path]]): Path to the annotation file or list of paths to annotation files.
            image_dir (Union[Path, list[Path]]): Path to the image directory or list of paths to image directories.
            name (Union[None, str, list[str]], optional): Name of the dataset or list of names of the datasets. Defaults to None.

        """
        if isinstance(annotation_path, list) and isinstance(image_dir, list):
            datasets = []
            for i, (ann_path, img_dir) in track(
                enumerate(zip(annotation_path, image_dir)),
                total=len(annotation_path),
                description="Loading datasets...",
            ):
                datasets.append(
                    PerphixDataset.load(ann_path, img_dir, name=None if name is None else name[i])
                )
            return cls(datasets, **kwargs)
        else:
            dataset = PerphixDataset.load(annotation_path, image_dir, name=name)
            return cls([dataset], **kwargs)

    def save(self, save_dir: Path):
        for dataset in self.datasets:
            dataset.save(save_dir / f"{dataset.name}.json")

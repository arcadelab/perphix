"""The Perphix dataset"""
from typing import List, Tuple, Dict, Union, Optional, Any
import copy
import pycocotools.mask as mask_util
import numpy as np
import cv2
import logging

import datetime

log = logging.getLogger(__name__)


class PerphixBase:
    """A base class for Perphix datasets.

    Contains class variables for base dataset information.
    See :ref:`Data Format<perphix_data_format>` for more information.

    Attributes:
        info (dict): Base dataset information.

    """

    @classmethod
    def get_base_annotation(cls) -> dict[str, list[dict]]:
        """Get the base annotation dictionary."""
        return {
            "info": cls.info.copy(),
            "licenses": cls.licenses.copy(),
            "images": [],
            "annotations": [],
            "categories": cls._base_categories.copy(),
            "sequences": [],
            "seq_categories": cls.seq_categories.copy(),
            "class_categories": cls.class_categories.copy(),
        }

    info = {
        "description": """Percutaneous fracture fixation. If you use this dataset, kindly cite the paper.""",
        "url": "https://github.com/arcadelab/perphix.",
        "version": "0.1",
        "year": 2023,
        "contributor": "Benjamin D. Killeen, ARCADE Lab, Johns Hopkins University",
        "date_created": datetime.datetime.now().strftime("%Y-%m-%d"),
    }

    licenses = [
        {
            "url": "https://nmdid.unm.edu/resources/data-use",
            "id": 1,
            "name": "NMDID Data Use Agreement",
        },
        {
            "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
            "id": 2,
            "name": "Attribution-NonCommercial-ShareAlike License",
        },
        {
            "url": "http://creativecommons.org/licenses/by-nc/2.0/",
            "id": 3,
            "name": "Attribution-NonCommercial License",
        },
    ]

    # For instance segmentation/detection, there are X supercategories of object: instrument, patient, corridor
    _base_categories = [
        {
            "supercategory": "instrument",  # 0
            "id": 1,
            "name": "wire",
        },
        {
            "supercategory": "instrument",  # 1
            "id": 2,
            "name": "screw",
        },
        {
            "supercategory": "patient",  # 2
            "id": 3,
            "name": "hip_left",
            "keypoints": [
                "l_sps",  # 0
                "l_ips",  # 1
                "l_iof",  # 2
                "l_gsn",  # 3
                "l_it",  # 4
                "l_mof",  # 5
                "l_asis",  # 6
                "l_is",  # 7
            ],
            "skeleton": [
                [0, 1],
                [0, 5],
                [1, 5],
                [1, 2],
                [2, 7],
                [7, 3],
                [3, 6],
                [7, 6],
            ],
        },
        {
            "supercategory": "patient",  # 3
            "id": 4,
            "name": "hip_right",
            "keypoints": [
                "r_sps",  # 0
                "r_ips",  # 1
                "r_iof",  # 2
                "r_gsn",  # 3
                "r_it",  # 4
                "r_mof",  # 5
                "r_asis",  # 6
                "r_is",  # 7
            ],
            "skeleton": [
                [0, 1],
                [0, 5],
                [1, 5],
                [1, 2],
                [2, 7],
                [7, 3],
                [3, 6],
                [7, 6],
            ],
        },
        {
            "supercategory": "patient",  # 4
            "id": 5,
            "name": "femur_left",
        },
        {
            "supercategory": "patient",  # 5
            "id": 6,
            "name": "femur_right",
        },
        {
            "supercategory": "patient",
            "id": 7,
            "name": "sacrum",
        },
        {
            "supercategory": "patient",
            "id": 8,
            "name": "vertebrae_L5",
        },
        {
            "supercategory": "patient",
            "id": 9,
            "name": "pelvis",
            "keypoints": [
                "r_sps",  # 0
                "r_ips",  # 1
                "r_iof",  # 2
                "r_gsn",  # 3
                "r_it",  # 4
                "r_mof",  # 5
                "r_asis",  # 6
                "r_is",  # 7
                "l_sps",  # 8
                "l_ips",  # 9
                "l_iof",  # 10
                "l_gsn",  # 11
                "l_it",  # 12
                "l_mof",  # 13
                "l_asis",  # 14
                "l_is",  # 15
            ],
            "skeleton": [
                [0, 1],
                [0, 5],
                [1, 5],
                [1, 2],
                [2, 7],
                [7, 3],
                [3, 6],
                [7, 6],
                [8, 9],
                [8, 13],
                [9, 13],
                [9, 10],
                [10, 15],
                [15, 11],
                [11, 14],
                [15, 14],
            ],
        },
        {
            "supercategory": "corridor",
            "id": 10,
            "name": "s1_left",
        },
        {
            "supercategory": "corridor",
            "id": 11,
            "name": "s1_right",
        },
        {
            "supercategory": "corridor",
            "id": 12,
            "name": "s1",
        },
        {
            "supercategory": "corridor",
            "id": 13,
            "name": "s2",
        },
        {
            "supercategory": "corridor",
            "id": 14,
            "name": "ramus_left",
        },
        {
            "supercategory": "corridor",
            "id": 15,
            "name": "ramus_right",
        },
        {
            "supercategory": "corridor",
            "id": 16,
            "name": "teardrop_left",
        },
        {
            "supercategory": "corridor",
            "id": 17,
            "name": "teardrop_right",
        },
    ]

    num_categories = len(_base_categories)

    _annotation_pretty_names = {
        "wire": "wire",
        "screw": "screw",
        "hip_left": "L hip",
        "hip_right": "R hip",
        "femur_left": "L femur",
        "femur_right": "R femur",
        "sacrum": "sacrum",
        "vertebrae_L5": "L5",
        "pelvis": "pelvis",
        "s1_left": "L S1",
        "s1_right": "R S1",
        "s1": "S1",
        "s2": "S2",
        "ramus_left": "L ramus",
        "ramus_right": "R ramus",
        "teardrop_left": "L teardrop",
        "teardrop_right": "R teardrop",
    }

    _annotation_ids = dict((ann["name"], ann["id"]) for ann in _base_categories)
    _annotation_from_id = dict((ann["id"], ann) for ann in _base_categories)
    _annotation_names: dict[str, int] = dict(
        (ann["name"], i) for i, ann in enumerate(_base_categories)
    )
    _annotation_labels: dict[int, str] = dict(
        (i, ann["name"]) for i, ann in enumerate(_base_categories)
    )

    @classmethod
    def get_annotation_catid(cls, name: str) -> int:
        return cls._annotation_ids[name]

    @classmethod
    def get_annotation_category(cls, name: int) -> Dict:
        catid = cls.get_annotation_catid(name)
        return cls._annotation_from_id[catid]

    @classmethod
    def get_annotation_name(cls, catid: int) -> str:
        return cls._annotation_from_id[catid]["name"]

    @classmethod
    def get_annotation_pretty_name(cls, catid: int) -> str:
        return cls._annotation_pretty_names[cls.get_annotation_name(catid)].upper()

    @classmethod
    def get_annotation_name_from_label(cls, label: int) -> List[str]:
        """Get the pretty name of the annotation with the given label."""
        return cls._annotation_labels[label]

    @classmethod
    def get_annotation_pretty_name_from_label(cls, label: int) -> int:
        name = cls._annotation_labels[label]
        return cls._annotation_pretty_names[name].upper()

    keypoint_names: list[str] = _annotation_from_id[9]["keypoints"]
    _keypoint_names: dict[str, int] = dict((kname, i) for i, kname in enumerate(keypoint_names))
    _keypoint_labels: dict[int, str] = dict((i, kname) for i, kname in enumerate(keypoint_names))
    keypoint_pretty_names = [kname.replace("_", " ").upper() for kname in keypoint_names]
    num_keypoints = len(_keypoint_names)

    @classmethod
    def get_keypoint_name(cls, label: int) -> str:
        """Get the name of the keypoint with the given label."""
        return cls._keypoint_labels[label]

    @classmethod
    def get_keypoint_pretty_name(cls, label: int) -> str:
        """Get the pretty name of the keypoint with the given label."""
        return cls._keypoint_labels[label].replace("_", " ").upper()

    seq_categories = [
        {
            "supercategory": "task",
            "id": 0,
            "name": "s1_left",
        },
        {
            "supercategory": "task",
            "id": 1,
            "name": "s1_right",
        },
        {
            "supercategory": "task",
            "id": 2,
            "name": "s1",
        },
        {
            "supercategory": "task",
            "id": 3,
            "name": "s2",
        },
        {
            "supercategory": "task",
            "id": 4,
            "name": "ramus_left",
        },
        {
            "supercategory": "task",
            "id": 5,
            "name": "ramus_right",
        },
        {
            "supercategory": "task",
            "id": 6,
            "name": "teardrop_left",
        },
        {
            "supercategory": "task",
            "id": 7,
            "name": "teardrop_right",
        },
        {
            "supercategory": "activity",
            "id": 8,
            "name": "position_wire",
        },
        {
            "supercategory": "activity",
            "id": 9,
            "name": "insert_wire",
        },
        {
            "supercategory": "activity",
            "id": 10,
            "name": "insert_screw",
        },
        {
            "supercategory": "acquisition",
            "id": 11,
            "name": "ap",
        },
        {
            "supercategory": "acquisition",
            "id": 12,
            "name": "lateral",
        },
        {
            "supercategory": "acquisition",
            "id": 13,
            "name": "inlet",
        },
        {
            "supercategory": "acquisition",
            "id": 14,
            "name": "outlet",
        },
        {
            "supercategory": "acquisition",
            "id": 15,
            "name": "oblique_left",
        },
        {
            "supercategory": "acquisition",
            "id": 16,
            "name": "oblique_right",
        },
        {
            "supercategory": "acquisition",
            "id": 17,
            "name": "teardrop_left",
        },
        {
            "supercategory": "acquisition",
            "id": 18,
            "name": "teardrop_right",
        },
        {
            "supercategory": "frame",
            "id": 19,
            "name": "fluoro_hunting",
        },
        {
            "supercategory": "frame",
            "id": 20,
            "name": "assessment",
        },
    ]

    _seq_category_names = dict(((ann["supercategory"], ann["name"]), ann) for ann in seq_categories)
    _seq_category_ids = dict((ann["id"], ann) for ann in seq_categories)

    @classmethod
    def get_sequence_catid(cls, supercategory: str, name: str) -> int:
        """Have to use the supercategory because there are multiple categories with the same name."""
        return cls._seq_category_names[(supercategory, name)]["id"]

    @classmethod
    def get_sequence_category(cls, catid: int) -> Dict:
        return cls._seq_category_ids[catid]

    @classmethod
    def get_sequence_name(cls, catid: int) -> str:
        return cls._seq_category_ids[catid]["name"]

    class_categories = [
        {
            "supercategory": "view",
            "id": 1,
            "name": "ap",
        },
        {
            "supercategory": "view",
            "id": 2,
            "name": "lateral",
        },
        {
            "supercategory": "view",
            "id": 3,
            "name": "inlet",
        },
        {
            "supercategory": "view",
            "id": 4,
            "name": "outlet",
        },
        {
            "supercategory": "view",
            "id": 5,
            "name": "oblique_left",
        },
        {
            "supercategory": "view",
            "id": 6,
            "name": "oblique_right",
        },
        {
            "supercategory": "view",
            "id": 7,
            "name": "teardrop_left",
        },
        {
            "supercategory": "view",
            "id": 8,
            "name": "teardrop_right",
        },
        {
            "supercategory": "view",
            "id": 9,
            "name": "fluoro_hunting",
        },
    ]

    _class_names = dict((ann["name"], ann) for ann in class_categories)
    _class_catids = dict((ann["id"], ann) for ann in class_categories)

    def get_class_catid(self, name: str) -> int:
        return self._class_names[name]["id"]

    def get_class_name(self, class_catid: int) -> str:
        return self._class_catids[class_catid]["name"]

    @classmethod
    def remove_keypoints(cls, annotation: Dict[str, Any]) -> Dict[str, Any]:
        """Remove the keypoints from the annotation."""
        annotation = copy.deepcopy(annotation)
        for anno in annotation["annotations"]:
            if "keypoints" in anno:
                del anno["keypoints"]

        return annotation

    @classmethod
    def pelvis_only(cls, annotation: Dict[str, Any]) -> Dict[str, Any]:
        """Make a new annotation with only the pelvis (for keypoint detection).

        This is useful for training a model to segment the hips and detect keypoints.

        """
        annotation = copy.deepcopy(annotation)
        new_annotations = []
        for anno in annotation["annotations"]:
            if anno["category_id"] == cls.get_annotation_catid("pelvis"):
                new_annotations.append(anno)

        categories = [cat for cat in annotation["categories"] if cat["name"] == "pelvis"]

        annotation["annotations"] = new_annotations
        annotation["categories"] = categories
        return annotation

    @classmethod
    def fix_sequences(cls, annotation: dict[str, Any]) -> dict[str, Any]:
        """Fix the sequences based on image file names.

        This is a convenience function in case image sequences are stored improperly in the
        dictionary.
        """
        annotation = copy.deepcopy(annotation)

        sequences = []
        prev_seq_category_names = None
        seq_lengths = [0, 0, 0, 0]
        for image in annotation["images"]:
            image_id = image["id"]
            seq_category_names = image["file_name"].split(".")[0].split("-")[-4:]

            if prev_seq_category_names is None:
                prev_seq_category_names = seq_category_names
                first_frame_ids = [image_id, image_id, image_id, image_id]
                seq_lengths = [1, 1, 1, 1]
                continue

            # TODO: figure out why the acquisition sequences are overlapping and fix it.
            for i, supercategory in enumerate(["task", "activity", "acquisition", "frame"]):
                if seq_category_names[i] != prev_seq_category_names[i]:
                    seq_category_name = prev_seq_category_names[i].replace("screw_", "")
                    seq = {
                        "id": len(sequences),
                        "first_frame_id": first_frame_ids[i],
                        "seq_length": seq_lengths[i],
                        "seq_category_id": cls.get_sequence_catid(supercategory, seq_category_name),
                    }

                    sequences.append(seq)
                    first_frame_ids[i] = image_id
                    seq_lengths[i] = 0

                seq_lengths[i] += 1
                prev_seq_category_names[i] = seq_category_names[i]

        # Add the last sequence.
        for i, supercategory in enumerate(["task", "activity", "acquisition", "frame"]):
            seq_category_name = prev_seq_category_names[i].replace("screw_", "")
            seq = {
                "id": len(sequences),
                "first_frame_id": first_frame_ids[i],
                "seq_length": seq_lengths[i],
                "seq_category_id": cls.get_sequence_catid(supercategory, seq_category_name),
            }

            sequences.append(seq)

        annotation["sequences"] = sequences
        return annotation

    @staticmethod
    def decode_annotations(
        image_info: dict[str, Any], annos: list[dict[str, Any]]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Decode a list of annotations.

        Args:
            annos (list[dict[str, Any]]): List of `n` annotations.

        Returns:
            category_ids: (n,) integer category IDs.
            keypoints: (n, k, 2) float32 keypoint coordinates in (x, y) order. -1 means not visible.
            masks: (n, h, w) bool masks.
            bboxes: (n, 4) float32 bounding boxes in (x, y, size_x, size_y) style (COCO).

        """
        category_ids = []
        keypoints = []
        masks = []
        bboxes = []
        for anno in annos:
            bbox = anno["bbox"]
            if bbox[2] < 2 or bbox[3] < 2:
                continue
            bboxes.append(bbox)
            category_ids.append(anno["category_id"])

            if anno["category_id"] == 9:  # pelvis
                assert "keypoints" in anno
                keypoint_array = np.array(anno["keypoints"]).reshape(-1, 3)
                keypoints = [(x, y) for x, y, _ in keypoint_array]

            segm = anno["segmentation"]
            if isinstance(segm, list):
                # Convert polygon
                mask = mask_util.decode(
                    mask_util.frPyObjects(segm, image_info["height"], image_info["width"])
                )
                mask = mask[:, :, 0]
            elif isinstance(segm, dict):
                # RLE
                mask = mask_util.decode(segm)
            else:
                raise ValueError(
                    "Cannot transform segmentation of type '{}'!"
                    "Supported types are: polygons as list[list[float] or ndarray],"
                    " COCO-style RLE as a dict.".format(type(segm))
                )
            masks.append(mask)

        return np.array(category_ids), np.array(keypoints), masks, np.array(bboxes)

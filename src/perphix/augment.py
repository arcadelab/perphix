"""Build an augmentation pipeline."""


from typing import Optional
import numpy as np
import albumentations as A
from .augmenters import Dropout, CoarseDropout, gaussian_contrast


def build_augmentation(
    use_keypoint: bool = False,
    is_train: bool = True,
    image_only: bool = False,
    resize: Optional[int] = None,
) -> A.Compose:
    if image_only:
        kwargs = dict()
    elif use_keypoint:
        kwargs = dict(
            keypoint_params=A.KeypointParams(format="xy", remove_invisible=False),
            bbox_params=A.BboxParams(
                format="coco", label_fields=["category_ids"], min_visibility=0.1, min_area=10
            ),
        )
    else:
        kwargs = dict(
            bbox_params=A.BboxParams(
                format="coco", label_fields=["category_ids"], min_visibility=0.1, min_area=10
            ),
        )

    if not is_train:
        return A.Compose(
            [
                *([A.Resize(resize, resize)] if resize else []),
                A.Normalize(
                    mean=[0.5, 0.5, 0.5],
                    std=[0.5, 0.5, 0.5],
                    max_pixel_value=255,
                    always_apply=True,
                ),
            ],
            **kwargs,
        )

    return A.Compose(
        [
            A.Sequential(
                [
                    *([A.Resize(resize, resize)] if resize else []),
                    A.Affine(
                        rotate=(-1, 1),
                        translate_percent={"x": (-0.01, 0.01), "y": (-0.01, 0.01)},
                    ),
                ]
            ),
            A.SomeOf(
                [
                    A.OneOf(
                        [
                            A.GaussianBlur((3, 5)),
                            A.MotionBlur(blur_limit=(3, 5)),
                            A.MedianBlur(blur_limit=(3, 5)),
                        ],
                    ),
                    A.IAASharpen(alpha=(0.2, 0.5)),
                    A.IAAEmboss(alpha=(0.2, 0.5)),
                    A.CLAHE(clip_limit=(1, 4)),
                    A.OneOf(
                        [
                            A.MultiplicativeNoise(multiplier=(0.9, 1.1)),
                            A.HueSaturationValue(
                                hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20
                            ),
                            A.RandomBrightnessContrast(
                                brightness_limit=(-0.4, 0.2), contrast_limit=(-0.4, 0.2)
                            ),
                            gaussian_contrast(alpha=(0.6, 1.4), sigma=(0.1, 0.5), max_value=255),
                        ],
                    ),
                    A.RandomToneCurve(scale=0.1),
                    A.RandomShadow(),
                    A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, alpha_coef=0.08),
                    A.OneOf(
                        [
                            Dropout(dropout_prob=0.05),
                            CoarseDropout(
                                max_holes=24,
                                max_height=32,
                                max_width=32,
                                min_holes=8,
                                min_height=8,
                                min_width=8,
                            ),
                        ],
                        p=3,
                    ),
                    *(
                        []
                        if use_keypoint
                        else [A.ElasticTransform(alpha=50, alpha_affine=0, sigma=8)]
                    ),
                ],
                n=np.random.randint(0, 5),
                replace=False,
            ),
            A.InvertImg(p=0.5),
            A.Normalize(
                mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255, always_apply=True
            ),
        ],
        **kwargs,
    )

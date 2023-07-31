from typing import List
import logging
import numpy as np
import cv2
from typing import Optional
import seaborn as sns
from deepdrr.utils.image_utils import ensure_cdim, as_uint8, as_float32


log = logging.getLogger(__name__)


# TODO: redo each of these to allow for passing in a color palette and labels, as well as a scale
# factor.


def draw_keypoints(
    image: np.ndarray,
    keypoints: np.ndarray,
    names: Optional[List[str]] = None,
    colors: Optional[np.ndarray] = None,
    palette: str = "hls",
    seed: Optional[int] = None,
) -> np.ndarray:
    """Draw keypoints on an image.

    Args:
        image (np.ndarray): the image to draw on.
        keypoints (np.ndarray): the keypoints to draw. [N, 2] array of [x, y] coordinates.
            -1 indicates no keypoint present.
        names (List[str], optional): the names of the keypoints. Defaults to None.
        colors (np.ndarray, optional): the colors to use for each keypoint. Defaults to None.

    """

    image = ensure_cdim(as_uint8(image)).copy()
    keypoints = np.array(keypoints)
    if colors is None:
        colors = np.array(sns.color_palette(palette, keypoints.shape[0]))
        if seed is not None:
            np.random.seed(seed)
        colors = colors[np.random.permutation(colors.shape[0])]

    if np.any(colors < 1):
        colors = (colors * 255).astype(int)

    for i, keypoint in enumerate(keypoints):
        if np.any(keypoint < 0):
            continue
        color = colors[i].tolist()
        x, y = keypoint
        image = cv2.circle(image, (int(x), int(y)), 5, color, -1)
        if names is not None:
            image = cv2.putText(
                image,
                names[i],
                (int(x) + 5, int(y) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
    return image


def draw_masks(
    image: np.ndarray,
    masks: np.ndarray,
    alpha: float = 0.3,
    threshold: float = 0.5,
    names: Optional[List[str]] = None,
    colors: Optional[np.ndarray] = None,
    palette: str = "hls",
    seed: Optional[int] = None,
) -> np.ndarray:
    """Draw contours of masks on an image.

    Args:
        image (np.ndarray): the image to draw on.
        masks (np.ndarray): the masks to draw. [H, W, num_masks] array of masks.
    """

    image = as_float32(image)
    if colors is None:
        colors = np.array(sns.color_palette(palette, masks.shape[0]))
        if seed is not None:
            np.random.seed(seed)
        colors = colors[np.random.permutation(colors.shape[0])]

    image *= 1 - alpha
    for i, mask in enumerate(masks):
        bool_mask = mask > threshold
        image[bool_mask] = colors[i] * alpha + image[bool_mask] * (1 - alpha)

        contours, _ = cv2.findContours(
            bool_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        image = as_uint8(image)
        cv2.drawContours(image, contours, -1, (255 * colors[i]).tolist(), 1)
        image = as_float32(image)

    image = as_uint8(image)

    if names is not None:
        for i, mask in enumerate(masks):
            bool_mask = mask > threshold
            ys, xs = np.argwhere(bool_mask).T
            y = (np.min(ys) + np.max(ys)) / 2
            x = (np.min(xs) + np.max(xs)) / 2
            image = cv2.putText(
                image,
                names[i],
                (int(x) + 5, int(y) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255 * colors[i]).tolist(),
                1,
                cv2.LINE_AA,
            )

    return image

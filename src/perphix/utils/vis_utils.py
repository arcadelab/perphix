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


def stack(images: list[np.ndarray], axis: int = 0):
    """Stack a list of images into a single image."""
    pad_axis = 1 - axis

    # Get the biggest image along the pad_axis
    max_size = np.max([image.shape[pad_axis] for image in images])

    # Pad on the other axis to make all images the same size
    images = [
        np.pad(
            image,
            [(0, 0)] * pad_axis
            + [(0, max_size - image.shape[pad_axis])]
            + [(0, 0)] * (2 - pad_axis),
        )
        for image in images
    ]
    log.debug(f"Stacking images with shapes {[image.shape for image in images]} along {axis}")

    # Stack the images
    return np.concatenate(images, axis=axis)


def combine_heatmap(
    image: np.ndarray,
    heatmap: np.ndarray,
    channel=0,
    normalize=True,
) -> np.ndarray:
    """Visualize a heatmap on an image.

    Args:
        image (Union[np.ndarray, torch.Tensor]): 2D float image, [H, W]
        heatmap (Union[np.ndarray, torch.Tensor]): 2D float heatmap, [H, W], or [C, H, W] array of heatmaps.
        channel (int, optional): Which channel to use for the heatmap. For an RGB image, channel 0 would render the heatmap in red.. Defaults to 0.
        normalize (bool, optional): Whether to normalize the heatmap. This can lead to all-red images if no landmark was detected. Defaults to True.

    Returns:
        np.ndarray: A [H,W,3] numpy image.
    """
    image_arr = ensure_cdim(as_float32(image), c=3)
    heatmap_arr = ensure_cdim(heatmap, c=1)

    heatmap_arr = heatmap_arr.transpose(2, 0, 1)
    image_arr = image_arr.transpose(2, 0, 1)

    seg = False
    if heatmap_arr.dtype == bool:
        heatmap_arr = heatmap_arr.astype(np.float32)
        seg = True

    _, h, w = heatmap_arr.shape
    heat_sum = np.zeros((h, w), dtype=np.float32)
    for heat in heatmap_arr:
        heat_min = heat.min()
        heat_max = 4 if seg else heat.max()
        heat_min_minus_max = heat_max - heat_min
        heat = heat - heat_min
        if heat_min_minus_max > 1.0e-3:
            heat /= heat_min_minus_max

        heat_sum += heat

    for c in range(3):
        image_arr[c] = ((1 - heat_sum) * image_arr[c]) + (heat_sum if c == channel else 0)

    # log.debug(f"Combined heatmap with shape {image_arr.shape} and dtype {image_arr.dtype}")

    return as_uint8(image_arr.transpose(1, 2, 0))


def draw_corridor(
    image: np.ndarray,
    corridor: np.ndarray,
    color: Optional[np.ndarray] = None,
    thickness: int = 1,
    palette: str = "hls",
    name: Optional[str] = None,
):
    """Draw a corridor on an image (copy).

    Args:
        image (np.ndarray): the image to draw on.
        corridor (np.ndarray): the corridor to draw. [2, 2] array of [x, y] coordinates.

    """

    image = draw_keypoints(image, corridor, names=["0", "1"], palette=palette)

    if color is None:
        color = np.array(sns.color_palette(palette, 1))[0]

    if np.any(color < 1):
        color = (color * 255).astype(int)

    color = np.array(color).tolist()
    log.debug(f"Drawing corridor with color: {color}")

    # Draw a line between the two points
    image = cv2.line(
        image,
        (int(corridor[0, 0]), int(corridor[0, 1])),
        (int(corridor[1, 0]), int(corridor[1, 1])),
        color,
        thickness,
    )

    # Draw the name of the corridor
    if name is not None:
        fontscale = 0.75 / 512 * image.shape[0]
        thickness = max(int(1 / 256 * image.shape[0]), 1)
        offset = max(5, int(5 / 512 * image.shape[0]))
        x, y = np.mean(corridor, axis=0)
        image = cv2.putText(
            image,
            name,
            (int(x) + offset, int(y) - offset),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontscale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    return image


def draw_keypoints(
    image: np.ndarray,
    keypoints: np.ndarray,
    names: Optional[List[str]] = None,
    colors: Optional[np.ndarray] = None,
    palette: str = "hls",
    seed: Optional[int] = None,
) -> np.ndarray:
    """Draw keypoints on an image (copy).

    Args:
        image (np.ndarray): the image to draw on.
        keypoints (np.ndarray): the keypoints to draw. [N, 2] array of [x, y] coordinates.
            -1 indicates no keypoint present.
        names (List[str], optional): the names of the keypoints. Defaults to None.
        colors (np.ndarray, optional): the colors to use for each keypoint. Defaults to None.

    Returns:
        np.ndarray: the image with the keypoints drawn.

    """

    if len(keypoints) == 0:
        return image

    image = ensure_cdim(as_uint8(image)).copy()
    keypoints = np.array(keypoints)
    if colors is None:
        colors = np.array(sns.color_palette(palette, keypoints.shape[0]))
        if seed is not None:
            np.random.seed(seed)
        colors = colors[np.random.permutation(colors.shape[0])]

    if np.any(colors < 1):
        colors = (colors * 255).astype(int)

    fontscale = 0.75 / 512 * image.shape[0]
    thickness = max(int(1 / 256 * image.shape[0]), 1)
    offset = max(5, int(5 / 512 * image.shape[0]))
    radius = max(1, int(5 / 512 * image.shape[0]))

    for i, keypoint in enumerate(keypoints):
        if np.any(keypoint < 0):
            continue
        color = colors[i].tolist()
        x, y = keypoint
        image = cv2.circle(image, (int(x), int(y)), radius, color, -1)
        if names is not None:
            image = cv2.putText(
                image,
                names[i],
                (int(x) + offset, int(y) - offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontscale,
                color,
                thickness,
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
    """Draw contours of masks on an image (copy).

    Args:
        image (np.ndarray): the image to draw on.
        masks (np.ndarray): the masks to draw. [num_masks, H, W] array of masks.
    """

    image = as_float32(image)
    image = ensure_cdim(image)
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

    fontscale = 0.75 / 512 * image.shape[0]
    thickness = max(int(1 / 256 * image.shape[0]), 1)

    if names is not None:
        for i, mask in enumerate(masks):
            bool_mask = mask > threshold
            ys, xs = np.argwhere(bool_mask).T
            if len(ys) == 0:
                continue
            y = (np.min(ys) + np.max(ys)) / 2
            x = (np.min(xs) + np.max(xs)) / 2
            image = cv2.putText(
                image,
                names[i],
                (int(x) + 5, int(y) - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontscale,
                (255 * colors[i]).tolist(),
                thickness,
                cv2.LINE_AA,
            )

    return image

from pathlib import Path
import pydicom
from pydicom.errors import InvalidDicomError
from typing import Generator
import numpy as np


def get_dicom_paths(input_dir: Path) -> Generator[Path, None, None]:
    """Recursively yield all DICOM files in a directory.

    Args:
        input_dir (Path): Input directory

    Yields:
        Generator[Path, None, None]: DICOM file paths
    """
    for f in input_dir.iterdir():
        if f.is_dir():
            yield from get_dicom_paths(f)
        elif f.is_file():
            try:
                ds = pydicom.dcmread(f)
            except InvalidDicomError:
                continue
            yield f


def heatmap(x: float, y: float, scale: float, size: tuple[int, int]) -> np.ndarray:
    """Create a heatmap for a point in the image.

    Args:
        x (float): X coordinate (in width direction)
        y (float): Y coordinate (in height direction)
        scale (float): Scale of the heatmap
        size (tuple[int, int]): Size of the heatmap

    Returns:
        np.ndarray: Heatmap
    """
    H, W = size
    xs, ys = np.meshgrid(
        np.arange(H, dtype=np.float32), np.arange(W, dtype=np.float32), indexing="xy"
    )
    xdiff = xs - x
    ydiff = ys - y
    distance_squared = np.square(xdiff) + np.square(ydiff)
    h = np.exp(-distance_squared / (2 * scale * scale))
    return h.astype(np.float32)

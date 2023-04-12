from pathlib import Path
import pydicom
from pydicom.errors import InvalidDicomError
from typing import Generator



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
            
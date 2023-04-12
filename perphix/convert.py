from pathlib import Path
import logging
import pydicom
from PIL import Image
from rich.progress import track


from .utils import get_dicom_paths

log = logging.getLogger(__name__)


def convert(input_dir: Path, output_dir: Path, format: str = "png"):
    """Convert DICOM files to other formats.

    Does not preserve the naming convention of the source, but does preserve sorting order (mapping
    to integers).

    Args:
        input_dir (Path): Input directory
        output_dir (Path): Output directory
        format (str, optional): Output format. Defaults to "png".

    """

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory {input_dir} does not exist")

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    paths = sorted(list(get_dicom_paths(input_dir)))
    n = max(len(str(len(paths))), 3)
    format = format.lower()

    for i, p in track(enumerate(paths), description=f"Converting DICOMs to {format.upper()}"):
        stem = f"{i:0{n}d}"
        image = pydicom.dcmread(p).pixel_array
        image = Image.fromarray(image)
        # May need some debugging
        image.save(output_dir / f"{stem}.{format}")

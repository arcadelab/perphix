from pathlib import Path
from typing import Optional
import pydicom
from pydicom.errors import InvalidDicomError
import random
import logging
from copy import deepcopy

log = logging.getLogger(__name__)


def deidentify_dataset(ds: pydicom.Dataset, case_id: str = "") -> pydicom.Dataset:
    """Remove patient identifiable information from a DICOM dataset.

    Args:
        ds (pydicom.dataset.Dataset): Input DICOM dataset
        case (str, optional): Case ID. This is used as the new patient identifier. Defaults to "".

    Returns:
        pydicom.dataset.Dataset: Deidentified DICOM dataset

    """
    ds = deepcopy(ds)
    ds.PatientName = "Anonymous"
    ds.PatientID = case_id
    ds.PatientBirthDate = ""
    ds.PatientAddress = ""
    ds.MilitaryRank = ""
    ds.EthnicGroup = ""

    return ds


def deidentify(input_dir: Path, output_dir: Path, case_id: Optional[str] = None):
    """Remove patient identifiable information from DICOM files, recursively.

    Assumes all the dicom files in the directory correspond to the same patient.

    Args:
        input_dir (Path): Input directory
        output_dir (Path): Output directory

    """

    if case_id is None:
        case_id = f"{random.randint(0, 999999):06d}"

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory {input_dir} does not exist")

    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    for f in input_dir.iterdir():
        if f.is_dir():
            deidentify(f, output_dir / f.name, case_id)
        elif f.is_file():
            try:
                ds = pydicom.dcmread(f)
            except InvalidDicomError:
                continue
            ds = deidentify_dataset(ds, case_id)
            ds.save_as(output_dir / f.name)
        else:
            raise ValueError(f"Unexpected file type: {f}")

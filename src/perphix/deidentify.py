from pathlib import Path
from typing import Optional
import pydicom
from pydicom.errors import InvalidDicomError
import random
import logging
from copy import deepcopy

log = logging.getLogger(__name__)


def remove_tag(ds: pydicom.Dataset, tag: int, value: str = ""):
    """Remove a tag from a DICOM dataset.

    Args:
        ds (pydicom.dataset.Dataset): Input DICOM dataset
        tag (str): Tag to remove

    """
    if tag not in ds:
        return

    ds[tag].value = value


def remove_age(ds: pydicom.Dataset):
    """Remove the patient's age from a DICOM dataset.

    Args:
        ds (pydicom.dataset.Dataset): Input DICOM dataset

    """
    if 0x00101010 not in ds:
        return

    age = ds[0x00101010].value
    if age == "None":
        return

    try:
        a = int(age)
        if a >= 90:
            age = "≥90"
    except ValueError:
        pass

    ds[0x00101010].value = age


def deidentify_dataset(ds: pydicom.Dataset, case_id: str = "") -> pydicom.Dataset:
    """Remove patient identifiable information from a DICOM dataset.

    Args:
        ds (pydicom.dataset.Dataset): Input DICOM dataset
        case (str, optional): Case ID. This is used as the new patient identifier. Defaults to "".

    Returns:
        pydicom.dataset.Dataset: Deidentified DICOM dataset

    """
    ds = deepcopy(ds)

    remove_tag(ds, 0x00080014)  # Instance creator UID
    remove_tag(ds, 0x00080018)  # SOP Instance UID
    remove_tag(ds, 0x00080050)  # Accession Number
    remove_tag(ds, 0x00080081)  # Institution Address
    remove_tag(ds, 0x00080090)  # Referring Physician's Name
    remove_tag(ds, 0x00080092)  # Referring Physician's Address
    remove_tag(ds, 0x00080094)  # Referring Physician's Telephone Numbers
    remove_tag(ds, 0x00081030)  # Study Description
    remove_tag(ds, 0x0008103E)  # Series Description
    remove_tag(ds, 0x00081048)  # Physician(s) of Record
    remove_tag(ds, 0x00081050)  # Performing Physician's Name
    remove_tag(ds, 0x00081060)  # Name of Physician(s) Reading Study
    remove_tag(ds, 0x00081070)  # Operator's Name
    remove_tag(ds, 0x00081080)  # Admitting Diagnoses Description
    remove_tag(ds, 0x00081155)  # Referenced SOP Instance UID
    remove_tag(ds, 0x00082111)  # Derivation Description
    remove_tag(ds, 0x00100010, case_id)  # Patient's Name
    remove_tag(ds, 0x00100020, case_id)  # Patient ID
    remove_tag(ds, 0x00100030)  # Patient's Birth Date
    remove_tag(ds, 0x00100032)  # Patient's Birth Time
    remove_tag(ds, 0x00101000)  # Other Patient IDs
    remove_tag(ds, 0x00101001)  # Other Patient Names
    remove_tag(ds, 0x00101010)  # Patient's Age
    remove_tag(ds, 0x00101020)  # Patient's Size
    remove_tag(ds, 0x00101030)  # Patient's Weight
    remove_tag(ds, 0x00101090)  # Medical Record Locator
    remove_tag(ds, 0x00102160)  # Ethnic Group
    remove_tag(ds, 0x00102180)  # Occupation
    remove_tag(ds, 0x001021B0)  # Additional Patient History
    remove_tag(ds, 0x00104000)  # Patient Comments
    remove_tag(ds, 0x0020000D)  # Study Instance UID
    remove_tag(ds, 0x0020000E)  # Series Instance UID
    remove_tag(ds, 0x00200010)  # Study ID
    remove_tag(ds, 0x00200052)  # Frame of Reference UID
    remove_tag(ds, 0x00200200)  # Synchronization Frame of Reference UID
    remove_tag(ds, 0x00204000)  # Image Comments
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

    input_dir = Path(input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory {input_dir} does not exist")

    output_dir = Path(output_dir)
    if not output_dir.exists():
        output_dir.mkdir(parents=True)

    output_dicom_dir = output_dir / f"case-{case_id}"
    if not output_dicom_dir.exists():
        output_dicom_dir.mkdir(parents=True)

    for f in input_dir.iterdir():
        if f.is_dir():
            deidentify(f, output_dicom_dir / f.name, case_id)
        elif f.is_file():
            try:
                ds = pydicom.dcmread(f)
            except InvalidDicomError:
                continue
            ds = deidentify_dataset(ds, case_id)
            ds.save_as(output_dicom_dir / f.name)
        else:
            raise ValueError(f"Unexpected file type: {f}")

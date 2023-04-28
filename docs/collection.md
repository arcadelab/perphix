# Data Collection

This document describes the process of collecting and de-identifying pre-operative CT images and
X-ray image sequences. Since the collection and de-identification process depends on the source from
which the data are being obtained, details for each are provided. During the course of collection,
the physician carries out these procedures. Although the physician identifies patient records to
obtain corresponding images, no patient identifiers are preserved in the preserved records. The
de-identification process described for each collection method below ensures that no patient
identifers are preserved.

In all cases, the collected data will be transferred to a OneDrive sharepoint at the time of
collection via a lab-owned PC laptop, and any intermediate storage devices will be erased
immediately after transfer.

## Creating a New Patient Folder

In all cases, a new patient folder must be created on the Sharepoint. This folder name contains a
randomly generated 6-digit number that is used internally as the case number. The internal case
number is not derived in any way from patient identifiers. To add a new patient folder, do the
following:

- On the lab laptop, go to [numbergenerator.org](https://numbergenerator.org) and generate a 6-digit
   number, `XXXXXX`. (If the number is already in use, regenerate.)
- On the "Perphix" sharepoint, create a new folder under "General/collections." Name the folder
   `perphix-case-XXXXXX` where `XXXXXX` is the 6-digit number generated above.

Now follow the instructions for the specific collection method below.

## X-ray Images

In all cases, using the lab laptop, access the online sharepoint and in the case folder
`perphix-case-XXXXXX` created above, create a new folder named `procedure-YY`, where `YY` is the
procedure number starting at `00`. If multiple procedures have been collected, increment the
procedure number.

### USB Export from Siemens Cios Mobile C-arm

![Cios](img/siemens-cios.png)

The Cios is a flat-panel mobile C-arm preferred by Dr. Osgood. The C-arm is equipped with export
functionalities that de-identify data directly. Following the procedure, navigate to the viewing
pane and ensure the following:

- Select "USB Device" as the export target.
- Selected images is set to "All images" (may be set to "Marked Images" by default).
- Select the box to "Anonymize" the data. (The Cios software uses the word "Anonymize" instead of
"De-identify".)
- Click "Export." Do not immediately remove the USB drive! The export process may take several
minutes. You can monitor the progress by following the same steps and selecting "See Status."
- Once the export is complete, remove the USB drive.
- Upload the de-identified export data from the USB drive to the new procedure folder `procedure-YY` created above.
- Double check the data has been uploaded correctly.
- Delete the patient data on the USB drive by re-formatting the drive. Be sure to perform a
"complete format" to ensure that the data is completely erased.

### PACS Export

Sometimes, the Cios may not be accessible (because it is being used). In these cases, X-ray data can
be obtained and de-identified directly from the PACS system. If the PACS system enables
de-identification at export time, this is the preferred method. Otherwise, the data must be
de-identified after the fact (but before uploading to the sharepoint).

- On the hospital PACS viewer, identify the procedure in question, often described as "Fluoro No Charge."
- Select the "Save" button.
- Select "DICOM" as the export format. This will save the data as a ZIP folder containing the data
  and a PACS viewer as an EXE file.
- Once the export has finished, navigate to the folder where the data was saved. Use the "Move To"
  button to move the zip to the USB drive.
- Using the lab laptop, unzip the saved data on the USB drive. The unzipped data will contain a
  folder `exam/ZZZZZZZZ` where `ZZZZZZZZ` is a random alphanumeric code. This folder contains the
  DICOM images.
- Create a temporary folder on the USB drive and run

```bash
python -m perphix deidentify -i /PATH/TO/exam/ZZZZZZZZ -o /PATH/TO/tmp --case XXXXXX
```

The `deidentify` command removes patient identifying information from the headers with the following code block:

```python
import pydicom
ds = pydicom.dcmread("PATH/TO/DICOM/FILE")
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
remove_tag(ds, 0x00100010, f"case-{case_id}")  # Patient's Name
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
ds.save_as("PATH/TO/OUTPUT/DICOM/FILE")
```

where `ds` is the `pydicom.Dataset` representing the DICOM image, and `case_id` is the internal case
number generated above. The `remove_tag` function is defined as

```python
def remove_tag(ds: pydicom.Dataset, tag: int, value: str = ""):
    if tag not in ds:
        return

    ds[tag].value = value
```

- Upload the de-identified data to the new procedure folder `procedure-YY` created above.
- Double check the data has been uploaded correctly.
- Delete the patient data on the USB drive by re-formatting the drive. Be sure to perform a
  "complete format" to ensure that the data is completely erased.

## CT Images

CT images may obtained directly from PACS. If the PACS system enables de-identification at export
time, this is the preferred method. Otherwise, the data must be de-identified after the fact (but
before uploading to the sharepoint). The process is similar to the X-ray images above, but the
`dcm2niix` tool is used to convert the DICOM images to NIfTI format, which does not preserve patient
identifiers by nature of the format.

- On the hospital PACS viewer, with the same patient as above, identify the CT scan in question.
  This must be a high-resolution CT scan of the lower torso. If the head is visible in the CT scan,
  it will be cropped in a later step.
- Select the "Save" button.
- Select "DICOM" as the export format. This will save the data as a ZIP folder containing the data
  and a PACS viewer as an EXE file.
- Once the export has finished, navigate to the folder where the data was saved. Use the "Move To"
  button to move the zip to the USB drive.
- Using the lab laptop, unzip the saved data on the USB drive. Identify the folder containing the
  DICOM images.
- Create a temporary folder on the USB drive and run

```bash
dcm2niix -z y -o /PATH/TO/tmp /PATH/TO/DICOM/FOLDER
```

This will result in two files, a `.nii.gz` file and a `.json` file. The `.nii.gz` file contains
the image data, and the `.json` file contains the metadata. The metadata contains the patient
identifiers, but the image data does not. Delete the `.json` file.

- If the CT scan contains the head, crop the image to remove the head. This can be done using 3D
  Slicer. Open the `.nii.gz` file in 3D Slicer, and use the "Crop Volume" module to crop the image.
  Save the cropped image as a new `.nii.gz` file in the temporary folder.
- On the sharepoint, create a new folder under `perphix-case-XXXXXX` named `ct-YY`, where `YY` is
  the CT scan number starting at `00`. If multiple CT scans have been collected, increment the
  number.
- Upload the `.nii.gz` file to the new CT folder `ct-YY` created above.
- Double check the data has been uploaded correctly.
- Delete the patient data on the USB drive by re-formatting the drive. Be sure to perform a
  "complete format" to ensure that the data is completely erased.

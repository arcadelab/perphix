# Data Collection

This document describes the process of collecting and de-identifying pre-operative CT images and
X-ray image sequences from Johns Hopkins Hospital. Since the collection and de-identification
process depends on the source from which the data are being obtained, details for each are provided.
During the course of collection, the physician carries out these procedures, accessing PHI in order
to obtain corresponding X-ray and CT image data, but no patient identifiers are preserved in the
final dataset. The de-identification process described for each collection method below ensures that
no patient identifers are preserved.

In all cases, the collected data will be transferred to a OneDrive sharepoint at the time of
collection via a lab-owned PC laptop, and any intermediate storage devices will be erased
immediately after transfer.

## Creating a New Patient Folder

In all cases, a new patient folder must be created on the Sharepoint. This folder name contains a
randomly generated 6-digit number that is used internally as the case number. The internal case
number is not derived in any way from patient identifiers. To add a new patient folder, do the
following:

1. On the lab laptop, go to [numbergenerator.org](https://numbergenerator.org) and generate a
   6-digit number, `XXXXXX`. (If the number is already in use, regenerate.)
1. On the "Perphix" sharepoint, create a new folder under "General/collections." Name the folder
   `perphix-case-XXXXXX` where `XXXXXX` is the 6-digit number generated above.

Now follow the instructions for the specific collection method below.

## X-ray Images

### USB Export from Siemens Cios Mobile C-arm

![Cios](img/siemens-cios.png)

The Cios is a flat-panel mobile C-arm preferred by Dr. Osgood. The C-arm is equipped with export
functionalities that de-identify data directly. Following the procedure, navigate to the viewing
pane and ensure the following:

1. Select "USB Device" as the export target.
2. Selected images is set to "All images" (may be set to "Marked Images" by default).
3. Select the box to "Anonymize" the data. (The Cios software uses the word "Anonymize" instead of
"De-identify".)
4. Click "Export." Do not immediately remove the USB drive! The export process may take several
minutes. You can monitor the progress by following the same steps and selecting "See Status."
5. Once the export is complete, remove the USB drive.
6. Using the lab laptop, access the online sharepoint and in the case folder `perphix-case-XXXXXX`
created above, create a new folder named `procedure-YY`, where `YY` is the procedure number
starting at `00`. If multiple procedures have been collected, increment the procedure number.
7. Upload the de-identified export data from the USB drive to the new procedure folder created above.
8. Double check the data has been uploaded correctly.
9. Delete the patient data on the USB drive by re-formatting the drive. Be sure to perform a
"complete format" to ensure that the data is completely erased.

### PACS Export

Sometimes, the Cios may not be accessible (because it is being used). In these cases, X-ray data can
be obtained and de-identified directly from the PACS system:

1. On the hospital PACS viewer, identify the procedure in question, often described as "Fluoro No Charge."
2. Select the "Save" button.
3. Select "DICOM" as the export format. This will save the data as a ZIP folder containing the data
   and a PACS viewer as an EXE file.
4. Once the export has finished, navigate to the folder where the data was saved. Use the "Move To"
   button to move the zip to the USB drive.
5. Using the lab laptop, unzip the saved data on the USB drive. The zip 



TODO: update the above based on using the PACS export on the lab laptop.

## CT Images

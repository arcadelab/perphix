# Data Format

Perphix data is modeled after the COCO dataset. The images are stored en masse in a single
directory, called the `image_dir`. The annotations are stored in a single JSON file, called the
`annotation`. For a detailed description of the COCO dataset format, see [this
tutorial](https://www.immersivelimit.com/tutorials/create-coco-annotations-from-scratch).
Here, we provide an overview of the annotations.

## Annotation Format

### v0.0.1

The annotation file is a JSON file with the following fields:

- `info`: a dictionary with information about the dataset.
- `licenses`: a list of dictionaries with information about the licenses of the images in the dataset.
- `images`: a list of dictionaries each containing information about an image in the dataset.
- `annotations`: a list of dictionaries each containing a spatial annotations of the images in the dataset, i.e. bounding boxes, segmentation masks, and keypoints.
- `categories`: a list of dictionaries with information about the categories of the images in the dataset.
- `sequences`: a list of dictionaries with information about the image sequences in the dataset.
- `seq_categories`: a list of dictionaries with information about the categories of the image sequences in the dataset.

In each dictionary, the `id` field is the unique identifier of *that* dictionary. The dictionary may
also contain references to other Ids, e.g. the `image_id` field in the `annotations` dictionary
refers to the `id` field in the `images` dictionary.

In detail, here are the fields of each dictionary:

```json
{
    "info": {
        "description": "The description of the dataset.",
        "url": "The URL of the dataset.",
        "version": "The version of the dataset.",
        "year": "The year of the dataset.",
        "contributor": "The contributor of the dataset.",
        "date_created": "The date the dataset was created.",
    },
    "licenses": [
        {
            "url": "The URL of the license.",
            "id": "The ID of the license.",
            "name": "The name of the license.",
        },
    ],
    "images": [
        {
            "license": "The ID of the license.",
            "file_name": "The name of the image file.",
            "coco_url": "The URL of the image.",
            "height": "The height of the image.",
            "width": "The width of the image.",
            "date_captured": "The date the image was captured.",
            "flickr_url": "The URL of the image.",
            "id": "The ID of the image.",
            "frame_id": "The ID of the frame within the sequence.",
            "seq_length": "The length of the sequence.",
            "first_frame_id": "The ID of the first image in the sequence.",
            "case_name": "The name (case-XXXXXX) of the case. (Only when simulating from the NMDID dataset)",
        },
        ...
    ],
    "annotations": [
        {
            "segmentation": {
                "size": [HEIGHT, WIDTH],
                "counts": "The run-length encoding of the segmentation.",
            },
            "area": "The area of the object.",
            "iscrowd": "Whether the object is a crowd.",
            "image_id": "The ID of the image.",
            "bbox": "The bounding box of the object.",
            "category_id": "The ID of the category.",
            "id": "The ID of the annotation.",
            "track_id": "A unique ID for this instance of the object.",
        },
        ...
    ],
    "sequences": [
        {
            "id": "The ID of the sequence.",
            "first_frame_id": "The ID of the first frame in the sequence.",
            "seq_length": "The length of the sequence.",
            "seq_category_id": "The id of the sequence category.",
        }
    ],
    "categories": [
        {
            "supercategory": "The supercategory of the category.",
            "id": "The ID of the category.",
            "name": "The name of the category.",
        },
        ...
    ],
    "seq_categories": [
        {
            "supercategory": "The supercategory of the sequence category.",
            "id": "The ID of the sequence category.",
            "name": "The name of the sequence category.",
        }
    }
}
```

### v0.0.2

This version is updated to include classes which each image may belong to. In this case, we classify the images based on their view.

The annotation file is a JSON file with the following fields:

- `info`: a dictionary with information about the dataset.
- `licenses`: a list of dictionaries with information about the licenses of the images in the dataset.
- `images`: a list of dictionaries each containing information about an image in the dataset.
- `annotations`: a list of dictionaries each containing a spatial annotations of the images in the dataset, i.e. bounding boxes, segmentation masks, and keypoints.
- `categories`: a list of dictionaries with information about the categories of the images in the dataset.
- `sequences`: a list of dictionaries with information about the image sequences in the dataset.
- `seq_categories`: a list of dictionaries with information about the categories of the image sequences in the dataset.

If the dataset contains individual images, the `sequences` list will be empty.

In each dictionary, the `id` field is the unique identifier of *that* dictionary. The dictionary may also contain references to other Ids, e.g. the `image_id` field in the `annotations` dictionary refers to the `id` field in the `images` dictionary.

In detail, here are the fields of each dictionary:

```json
{
    "info": {
        "description": "The description of the dataset.",
        "url": "The URL of the dataset.",
        "version": "The version of the dataset.",
        "year": "The year of the dataset.",
        "contributor": "The contributor of the dataset.",
        "date_created": "The date the dataset was created.",
    },
    "licenses": [
        {
            "url": "The URL of the license.",
            "id": "The ID of the license.",
            "name": "The name of the license.",
        },
    ],
    "images": [
        {
            "license": "The ID of the license.",
            "file_name": "The name of the image file.",
            "coco_url": "The URL of the image.",
            "height": "The height of the image.",
            "width": "The width of the image.",
            "date_captured": "The date the image was captured.",
            "flickr_url": "The URL of the image.",
            "id": "The ID of the image.",
            "frame_id": "The ID of the frame within the sequence.",
            "seq_length": "The length of the sequence.",
            "first_frame_id": "The ID of the first image in the sequence.",
            "case_name": "The name (case-XXXXXX) of the case. (Only when simulating from the NMDID dataset)",
            "classes": "Integer list of the class categories the image belongs to.",
        },
        ...
    ],
    "annotations": [
        {
            "segmentation": {
                "size": [HEIGHT, WIDTH],
                "counts": "The run-length encoding of the segmentation.",
            },
            "area": "The area of the object.",
            "iscrowd": "Whether the object is a crowd.",
            "image_id": "The ID of the image.",
            "bbox": "The bounding box of the object.",
            "category_id": "The ID of the category.",
            "id": "The ID of the annotation.",
            "track_id": "A unique ID for this instance of the object.",
        },
        ...
    ],
    "sequences": [
        {
            "id": "The ID of the sequence.",
            "first_frame_id": "The ID of the first frame in the sequence.",
            "seq_length": "The length of the sequence.",
            "seq_category_id": "The id of the sequence category.",
        }
    ],
    "categories": [
        {
            "supercategory": "The supercategory of the category.",
            "id": "The ID of the category.",
            "name": "The name of the category.",
        },
        ...
    ],
    "seq_categories": [
        {
            "supercategory": "The supercategory of the sequence category.",
            "id": "The ID of the sequence category.",
            "name": "The name of the sequence category.",
        },
        ...
    ],
    "classes": [
        {
            "supercategory": "The supercategory of the class.",
            "id": "The ID of the class.",
            "name": "The name of the class.",
        },
        ...
    ]
}
```

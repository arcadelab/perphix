#!/usr/bin/env python
from setuptools import find_packages
from setuptools import setup

setup(
    name="perphix",
    version="0.0.1",
    description="Utilities and documentation for the collection, annotation, and usage of X-ray image sequences for surgical phase recognition.",
    package_dir={"": "src"},
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Benjamin D. Killeen",
    author_email="killeen@jhu.edu",
    url="https://github.com/arcadelab/perphix",
    install_requires=[
        "click",
        "rich",
        "numpy",
        "pycocotools",
        "albumentations",
        "opencv-python",
    ],
    packages=find_packages(),
)

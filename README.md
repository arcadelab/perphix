# Perphix

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](
<https://opensource.org/licenses/MIT>)
[![PyPI](https://img.shields.io/pypi/v/perphix.svg)](
<https://pypi.org/project/perphix/>)
[![Documentation Status](https://readthedocs.org/projects/perphix/badge/?version=latest)](
<https://perphix.readthedocs.io/en/latest/?badge=latest>)

Utilities and documentation for the collection, annotation, and usage of X-ray image sequences for
surgical phase recognition, with a focus on percutaneous fracture fixation.

This repository is used by [Pelphix](https://github.com/benjamindkilleen/pelphix) to provide
data-related utilities and documentation. It classes for loading and processing data, as well as
documentation for the Perphix datasets. It is provided separately for ease of installation.

## Installation

Perphix can be installed from PyPI:

```bash
pip install perphix
```

### Development

Clone the repository and install from source using pip:

```bash
git clone git@github.com:arcadelab/perphix.git
cd perphix
pip install -e .
```

Alternatively, you can use the included `environment.yml` file to install an Anaconda environment.

```bash
conda env create -f environment.yml
conda activate perphix
```

## Documentation

The documentation for this project is hosted on [Read the Docs](https://perphix.readthedocs.io/en/latest/).

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@article{Pelphix2023,
 author = {Killeen, Benjamin D. and Zhang, Han and Mangulabnan, Jan and Armand, Mehran and Taylor, Russel H. and Osgood, Greg and Unberath, Mathias},
 title = {{Pelphix: Surgical Phase Recognition from X-ray Images in Percutaneous Pelvic Fixation}},
 journal = {arXiv},
 year = {2023},
 month = apr,
 eprint = {2304.09285},
 doi = {10.48550/arXiv.2304.09285}
}
```

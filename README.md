# HCATNetwork

Heart Coronary Arterial Tree Network data structure based on NetworkX

## Requirements

Developed and tested on Python 3.11 or later

## Installation

We recommend you create a virtual environment in python in which to install this package.

```sh
cd ~/project/folder/
python -m venv env_name
```

Further instructions about virtual environment can be found [here](https://docs.python.org/3/library/venv.html).

Now, install the package with:

```sh
python -m pip install git+https://github.com/AAMIASoftwares-research/HCATNetwork.git
```

## For developers

For testing without always re-installing everything, create a ```venv-dev``` in the
main folder of this repo.

Distribution procedure ([link](https://godatadriven.com/blog/a-practical-guide-to-setuptools-and-pyproject-toml/)):

1. Install package ```build``` in ```venv-dev```.
2. Update ```setup.cfg``` file.
3. Build your package by running ```python -m build --wheel``` from the folder where the ```pyproject.toml``` resides.

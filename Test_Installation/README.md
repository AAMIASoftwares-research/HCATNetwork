# Installation testing folder

This folder contains tests useful for checking correct installation and functionality
of the package in a new, empty python environment.

To test package installation, run:

```sh
python -m venv venv-installation
.\venv-installation\Scripts\activate    # Windows
source .\venv-installation\bin\activate # Linux
python -m pip install --upgrade pip
python -m pip install git+https://github.com/AAMIASoftwares-research/HCATNetwork.git
```

To test a new package distribution in an empty environment, just repeat the commands above,
the environment will be overwritten.

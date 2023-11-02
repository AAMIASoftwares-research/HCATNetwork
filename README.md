# HCATNetwork

Heart Coronary Arterial Tree Network handling library based on NetworkX.

<img src="./assets/images/graph_art_example.png">

## Requirements

Developed and tested on Python 3.11 or later. Absolutely requires python >= 3.9.
It is possible to install multiple python versions on the same machine without having them clashing, check out the official website for more infos.

## Installation

To use this package inside your own personal project, we recommend you create a virtual environment in python in which to install this package.

```sh
cd ~/project/folder/
python -m venv env_name
```

Further instructions about virtual environment can be found [here](https://docs.python.org/3/library/venv.html).

Now, install the package with:

```sh
python -m pip install git+https://github.com/AAMIASoftwares-research/HCATNetwork.git
```

Now, you should be able to

```py
import HCATNetwork
print(HCATNetwork.edge.SimpleCenterlineEdgeAttributes_KeysList)
```

just as you would with numpy and other packages.


## For developers

For testing without always re-installing everything, create a ```venv-dev``` in the
main folder of this repo.

Each time changes are made, update the "version" field in the ```pyproject``` file.

For developing, the requirements are listed in ```requirements-dev.txt```.

import setuptools

third_party_requirements_list = [
    "numpy",
    "matplotlib"
]

with open("README.md", "r", encoding="utf-8") as fh:
    readme_file_content = fh.read()

setuptools.setup(
    name             = "HCATNetwork",
    version          = "2023.03.22-00",
    author           = "Matteo Leccardi",
    author_email     = "matteo.leccardi@polimi.it",
    description      = "Testing installation of Package",
    long_description = readme_file_content,
    long_description_content_type = "text/markdown",
    url              = "https://github.com/AAMIASoftwares-research/HCATNetwork",
    project_urls = {
        "Bug Tracker":         "https://github.com/AAMIASoftwares-research/HCATNetwork/issues",
        "Institution Website": "https://www.polimi.it/",
        "Lab Website":         "https://www.b3lab.deib.polimi.it/"
    },
    license          = "MIT",
    packages         = ['HCATNetwork'],
    install_requires = third_party_requirements_list,
)
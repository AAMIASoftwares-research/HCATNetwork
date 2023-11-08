"""Utility to easily replace strings in a graph file.

Since this library is still under development, the names of the properties
are subject to change. This script allows to easily replace the names of the
properties in the graph files.
"""
import os

dir_of_graphs = "C:\\Users\\lecca\\Desktop\\AAMIASoftwares-research\\Data\\ASOCA\\Normal\\Centerlines_graphs\\"

STRINGS_TO_BE_REPLACED = [
    "topology_class",
    "arterial_tree"
]
STRINGS_TO_BE_REPLACED_WITH = [
    "topology",
    "side"
]

if __name__ == "__main__":
    for filename in os.listdir(dir_of_graphs):
        file = os.path.join(dir_of_graphs, filename)
        if os.path.isfile(file) and file.endswith(".GML"):
            print(f"Working on file: {os.path.basename(file)}")
            with open(file, "r") as f:
                content = f.read()
            for s, t in zip(STRINGS_TO_BE_REPLACED, STRINGS_TO_BE_REPLACED_WITH):
                content = content.replace(s, t)
            with open(file, "w") as f:
                f.write(content)

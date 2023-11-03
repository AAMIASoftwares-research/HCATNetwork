"""
This is the core functions and definitions used throughout the node, edge and graph modules.
Here are defined the basic dictionary type, standards and functionalities.

CoreDict must be the parent class of every node, edge and graph dictionary, as it implements
runtime type checking and dict initialisation starting from dictionary keys definitions with types.

To run as a module, activate the venv, go inside the HCATNetwork parent directory,
and use: python -m hcatnetwork.core.core
"""
# ##########################
# Type checking helper dict
# ##########################

from numpy import ndarray
from networkx.classes.graph import Graph
from networkx.classes.digraph import DiGraph
from networkx.classes.multigraph import MultiGraph
from networkx.classes.multidigraph import MultiDiGraph

TYPE_NAME_TO_TYPE_DICT = {
    "int": int,
    "float": float,
    "str": str,
    "bool": bool,
    "list": list,
    "tuple": tuple,
    "set": set,
    "dict": dict,
    "NoneType": type(None),
    "numpy.ndarray": ndarray,
    "networkx.classes.graph.Graph": Graph,
    "networkx.classes.digraph.DiGraph": DiGraph,
    "networkx.classes.multigraph.MultiGraph": MultiGraph,
    "networkx.classes.multidigraph.MultiDiGraph": MultiDiGraph
}




from collections import UserDict


class CoreDict(UserDict):
    """CoreDict
    This is the base class upon which every feature dictionary must be based on.

    This class implements runtime type checking when setting the dictionaries attributes,
    types annotations inheritance from parent classes up to CoreDict class,
    and key searching from dictionary value.

    All child dictionaries must have string objects as keys.

    This is useful for any kind of automatic type checking across the whole developed framework.
    
    End users won't need to use this class directly, but only its children.
    """
    def __init__(self, **kws):
        super().__init__()
        # Update annotations with the ones from parent classes up to CoreDict, and no more
        if not type(self) == CoreDict:
            self.extend_annotations(self)
        # Initialise all allowed keys to None
        for k in self.__annotations__.keys():
            # Avoid initial type checking
            super().__setitem__(k, None)
        # Initialised any key-value pair passed to the class constructor
        for key, value in kws.items():
            self.__setitem__(key, value)
         
    @classmethod
    def extend_annotations(cls, obj):
        d = {}
        mro = cls.mro()
        mro = (mro[:mro.index(CoreDict)+1])[::-1]
        for c in mro:
            try:
                d.update(**c.__annotations__)
            except AttributeError:
                # has no __annotations__ attribute.
                pass
        obj.__annotations__ = d
        # Convert string type names to actual types
        for k, v in obj.__annotations__.items():
            # key type must be string
            if not isinstance(k, str):
                raise TypeError(f"Invalid key type. {k} must be of type string.")
            # value type must type
            if isinstance(v, str):
                obj.__annotations__[k] = TYPE_NAME_TO_TYPE_DICT[v]

    # Get, set , del
    # all dict items will be stored with "hcatnetwork_" prefix. User won't notice it at all.
    def __getitem__(self, key):
        if not isinstance(key, str):
            raise TypeError(f"Invalid key type. {key} must be of type string.")
        return super().__getitem__(key)

    def __setitem__(self, key, item) -> None:
        if key in self.__annotations__:
            if isinstance(item, self.__annotations__[key]):
                return super().__setitem__(key, item)
            else:
                raise TypeError(f"Invalid value type. {key} only supports values of type {self.__annotations__[key]}")
        else:
            raise KeyError(f"Invalid key. {key} is not part of the allowed keys: {self.__annotations__} ")
    
    def __delitem__(self, key) -> None:
        print("Warning: deleting a key from am hcatnetwork dict is not allowed.")
        pass

    # utilities
    def key_of(self, value):
        """Returns the first matching key for the given value"""
        for k, v in self.items():
            if v == value:
                return k
        raise ValueError(value)

    def keys_of(self, value):
        """Returns all the matching keys for the given value"""
        for k, v in self.items():
            if v == value:
                yield k

    def is_full(self) -> bool:
        """
        Asserts validity of the dictionary for graph purposes,
        which means that no dictionary values must be "None" when creating a dictionary
        for an edge, node, graph.
        """
        for k, v in self.items():
            if v is None:
                return False
            if not isinstance(v, self.__annotations__[k]):
                return False
        return True



# Utilities - dict structure
def key_of(d: dict | CoreDict, value):
    for k, v in d.items():
        if v == value:
            return k
    raise ValueError(value)

def keys_of(d: dict | CoreDict, value):
    for k, v in d.items():
        if v == value:
            yield k

def assert_dictionary_validity(dictionary: dict | CoreDict) -> bool:
    for v in dictionary.values():
        if v is None:
            return False
    return True











# unit test
if __name__ == "__main__":
    class A(CoreDict):
        x:float
        y:float
    class B(A):
        z: int
    class C(B):
        alpha: str 
    class Z(A):
        omega: float
    class summa(C, Z):
        my_ann: bool
    d = summa()
    print(d, d.__annotations__, "\n")


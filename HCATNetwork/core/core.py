"""
This is the core functions and definitions used throughout the node, edge and graph modules.
Here are defined the basic dictionary type, standards and functionalities.

CoreDict must be the parent class of every node, edge and graph dictionary, as it implements
runtime type checking and dict initialisation starting from dictionary keys definitions with types.
"""
from collections import UserDict

class CoreDict(UserDict):
    """CoreDict
    This is the base class upon which every feature dictionary must be based on.
    This class implements runtime type checking when setting the dictionaries attributes,
    types annotations inheritance from parent classes up to CoreDict class,
    and key searching from dictionary value.
    """
    def __init__(self, **kws):
        super().__init__()
        # Update annotations with the ones from parent classes up to CoreDict, and no more
        if not type(self) == CoreDict:
            self.extend_annotations(self)
        # Initialise all allowed keys to None
        for k in self.__annotations__.keys():
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

    def __setitem__(self, key, item) -> None:
        if key in self.__annotations__:
            if isinstance(item, self.__annotations__[key]):
                return super().__setitem__(key, item)
            else:
                raise TypeError(f"Invalid value type. {key} only supports values of type {self.__annotations__[key]}")
        else:
            raise KeyError(f"Invalid key. {key} is not part of the allowed keys: {self.__annotations__} ")
    
    def keyOf(self, value):
        """Returns the first matching key for the given value"""
        for k, v in self.items():
            if v == value:
                return k
        raise ValueError(value)

    def keysOf(self, value):
        """Returns all the matching keys for the given value"""
        for k, v in self.items():
            if v == value:
                yield k

    def isValid(self) -> bool:
        """
        Asserts validity of the dictionary for graph purposes,
        which means that no dictionary values must be "None" when creating a dictionary
        for an edge, node, graph.
        """
        for v in self.values():
            if v is None:
                return False
        return True



# Utilities - dict structure
def keyOf(d: dict | CoreDict, value):
    for k, v in d.items():
        if v == value:
            return k
    raise ValueError(value)

def keysOf(d: dict | CoreDict, value):
    for k, v in d.items():
        if v == value:
            yield k

def assertDictionaryValidity(dictionary: dict | CoreDict) -> bool:
    for v in dictionary.values():
        if v is None:
            return False
    return True

# Utilities - functionalities




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

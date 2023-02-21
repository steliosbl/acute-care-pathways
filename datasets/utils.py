class DotDict(dict):
    """ Class for forming dict/class hybrids that allow attribute access via dot notation
    e.g. MyDotDict.some_key = MyDotDict['some_key'] """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
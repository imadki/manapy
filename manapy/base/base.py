from copy import copy, deepcopy

import numpy as nm
import scipy.sparse as sp
import six


def make_get_conf(conf, kwargs):
    def _get_conf_item(name, default=None, msg_if_none=None):
        return kwargs.get(name, conf.get(name, default=default,
                                         msg_if_none=msg_if_none))

    return _get_conf_item

def get_default(arg, default, msg_if_none=None):
    if arg is None:
        out = default
    else:
        out = arg

    if (out is None) and (msg_if_none is not None):
        raise ValueError(msg_if_none)

    return out

def assert_(condition, msg='assertion failed!'):                                                                                                                                                           
    if not condition:                                                                                                                                                                                      
        raise ValueError(msg)   
        
def try_imports(imports, fail_msg=None):
    
    """                                                                                                                                                                                                    
    Try import statements until one succeeds.                                                                                                                                                              
                                                                                                                                                                                                           
    Parameters                                                                                                                                                                                             
    ----------                                                                                                                                                                                             
    imports : list                                                                                                                                                                                         
        The list of import statements.                                                                                                                                                                     
    fail_msg : str                                                                                                                                                                                         
        If not None and no statement succeeds, a `ValueError` is raised with                                                                                                                               
        the given message, appended to all failed messages.                                                                                                                                                
                                                                                                                                                                                                           
    Returns                                                                                                                                                                                                
    -------                                                                                                                                                                                                
    locals : dict                                                                                                                                                                                          
        The dictionary of imported modules.                                                                                                                                                                
    """
    msgs = []
    for imp in imports:
        try:
            exec(imp)
            break

        except Exception as inst:
            msgs.append(str(inst))

    else:
        if fail_msg is not None:
            msgs.append(fail_msg)
            raise ValueError('\n'.join(msgs))

    return locals()


class Struct(object):
    def __init__(self, **kwargs):
        if kwargs:
            self.__dict__.update(kwargs)

    def _format_sequence(self, seq, threshold):
        threshold_half = threshold // 2

        if len(seq) > threshold:
            out = ', '.join(str(ii) for ii in seq[:threshold_half]) \
                  + ', ..., ' \
                  + ', '.join(str(ii) for ii in seq[-threshold_half:])

        else:
            out = str(seq)

        return out

    def _str(self, keys=None, threshold=20):
        ss = '%s' % self.__class__.__name__
        if hasattr(self, 'name'):
            ss += ':%s' % self.name
        ss += '\n'

        if keys is None:
            keys = list(self.__dict__.keys())

        str_attrs = sorted(Struct.get(self, '_str_attrs', keys))
        printed_keys = []
        for key in str_attrs:
            if key[-1] == '.':
                key = key[:-1]
                full_print = True
            else:
                full_print = False

            printed_keys.append(key)

            try:
                val = getattr(self, key)

            except AttributeError:
                continue

            if isinstance(val, Struct):
                if not full_print:
                    ss += '  %s:\n    %s' % (key, val.__class__.__name__)
                    if hasattr(val, 'name'):
                        ss += ':%s' % val.name
                    ss += '\n'

                else:
                    aux = '\n' + str(val)
                    aux = aux.replace('\n', '\n    ')
                    ss += '  %s:\n%s\n' % (key, aux[1:])

            elif isinstance(val, dict):
                sval = self._format_sequence(list(val.keys()), threshold)
                sval = sval.replace('\n', '\n    ')
                ss += '  %s:\n    dict with keys: %s\n' % (key, sval)

            elif isinstance(val, list):
                sval = self._format_sequence(val, threshold)
                sval = sval.replace('\n', '\n    ')
                ss += '  %s:\n    list: %s\n' % (key, sval)

            elif isinstance(val, tuple):
                sval = self._format_sequence(val, threshold)
                sval = sval.replace('\n', '\n    ')
                ss += '  %s:\n    tuple: %s\n' % (key, sval)

            elif isinstance(val, nm.ndarray):
                ss += '  %s:\n    %s array of %s\n' \
                      % (key, val.shape, val.dtype)

            elif isinstance(val, sp.spmatrix):
                ss += '  %s:\n    %s spmatrix of %s, %d nonzeros\n' \
                      % (key, val.shape, val.dtype, val.nnz)

            else:
                aux = '\n' + str(val)
                aux = aux.replace('\n', '\n    ')
                ss += '  %s:\n%s\n' % (key, aux[1:])

        other_keys = sorted(set(keys).difference(set(printed_keys)))
        if len(other_keys):
            ss += '  other attributes:\n    %s\n' \
                  % '\n    '.join(key for key in other_keys)

        return ss.rstrip()

    def __repr__(self):
        ss = "%s" % self.__class__.__name__
        if hasattr(self, 'name'):
            ss += ":%s" % self.name
        return ss

    def __add__(self, other):
        """Merge Structs. Attributes of new are those of self unless an
        attribute and its counterpart in other are both Structs - these are
        merged then."""
        new = copy(self)
        for key, val in six.iteritems(other.__dict__):
            if hasattr(new, key):
                sval = getattr(self, key)
                if issubclass(sval.__class__, Struct) and \
                        issubclass(val.__class__, Struct):
                    setattr(new, key, sval + val)
                else:
                    setattr(new, key, sval)
            else:
                setattr(new, key, val)
        return new

    def __iadd__(self, other):
        """Merge Structs in place. Attributes of self are left unchanged
        unless an attribute and its counterpart in other are both Structs -
        these are merged then."""
        for key, val in six.iteritems(other.__dict__):
            if hasattr(self, key):
                sval = getattr(self, key)
                if issubclass(sval.__class__, Struct) and \
                       issubclass(val.__class__, Struct):
                    setattr(self, key, sval + val)
            else:
                setattr(self, key, val)
        return self

    def str_class(self):
        """
        As __str__(), but for class attributes.
        """
        return self._str(list(self.__class__.__dict__.keys()))

    def str_all(self):
        ss = "%s\n" % self.__class__
        for key, val in six.iteritems(self.__dict__):
            if issubclass(self.__dict__[key].__class__, Struct):
                ss += "  %s:\n" % key
                aux = "\n" + self.__dict__[key].str_all()
                aux = aux.replace("\n", "\n    ")
                ss += aux[1:] + "\n"
            else:
                aux = "\n" + str(val)
                aux = aux.replace("\n", "\n    ")
                ss += "  %s:\n%s\n" % (key, aux[1:])
        return(ss.rstrip())

    def to_dict(self):
        return copy(self.__dict__)

    def get(self, key, default=None, msg_if_none=None):
        """
        A dict-like get() for Struct attributes.
        """
        out = getattr(self, key, default)

        if (out is None) and (msg_if_none is not None):
            raise ValueError(msg_if_none)

        return out

    def update(self, other, **kwargs):
        """
        A dict-like update for Struct attributes.
        """
        if other is None: return

        if not isinstance(other, dict):
            other = other.to_dict()
        self.__dict__.update(other, **kwargs)

    def set_default(self, key, default=None):
        """
        Behaves like dict.setdefault().
        """
        return self.__dict__.setdefault(key, default)

    def copy(self, deep=False, name=None):
        """Make a (deep) copy of self.

        Parameters:

        deep : bool
            Make a deep copy.
        name : str
            Name of the copy, with default self.name + '_copy'.
        """
        if deep:
            other = deepcopy(self)
        else:
            other = copy(self)

        if hasattr(self, 'name'):
            other.name = get_default(name, self.name + '_copy')

        return other


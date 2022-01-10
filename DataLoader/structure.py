from collections import UserDict
import collections

#FIXME remove that?
__author__ = 'github.com/hangtwenty'

from copy import deepcopy


def structure(mapping):
    """ Convert mappings to 'structures' recursively.
    Lets you use dicts like they're JavaScript Object Literals (~=JSON)...
    It recursively turns mappings (dictionaries) into namedtuples.
    Thus, you can cheaply create an object whose attributes are accessible
    by dotted notation (all the way down).
    Use cases:
        * Fake objects (useful for dependency injection when you're making
         fakes/stubs that are simpler than proper mocks)
        * Storing data (like fixtures) in a structured way, in Python code
        (data whose initial definition reads nicely like JSON). You could do
        this with dictionaries, but namedtuples are immutable, and their
        dotted notation can be clearer in some contexts.
    .. doctest::
        >>> t = structure({
        ...     'foo': 'bar',
        ...     'baz': {'qux': 'quux'},
        ...     'tito': {
        ...             'tata': 'tutu',
        ...             'totoro': 'tots',
        ...             'frobnicator': ['this', 'is', 'not', 'a', 'mapping']
        ...     }
        ... })
        >>> t # doctest: +ELLIPSIS
        Structure(tito=Structure(...), foo='bar', baz=Structure(qux='quux'))
        >>> t.tito # doctest: +ELLIPSIS
        Structure(frobnicator=[...], tata='tutu', totoro='tots')
        >>> t.tito.tata
        'tutu'
        >>> t.tito.frobnicator
        ['this', 'is', 'not', 'a', 'mapping']
        >>> t.foo
        'bar'
        >>> t.baz.qux
        'quux'
    Args:
        mapping: An object that might be a mapping. If it's a mapping, convert
        it (and all of its contents that are mappings) to namedtuples
        (called 'Structures').
    Returns:
        A structure (a namedtuple (of namedtuples (of namedtuples (...)))).
        If argument is not a mapping, it just returns it (this enables the
        recursion).
    """

    if (isinstance(mapping, collections.abc.Mapping) and
            not isinstance(mapping, ProtectedDict)):
        for key, value in mapping.items():
            mapping[key] = structure(value)
        return namedtuple_from_mapping(mapping)
    return mapping


def namedtuple_from_mapping(mapping: collections.abc.Mapping, name="Structure"):
    # Append underscore to the beginning of a string if it begins with a digit
    mapping = deepcopy(mapping)
    keys = list(mapping.keys())
    for k in keys:
        if k[0].isdigit():
            #FIXME find a better name
            mapping["key_" + k] = mapping.pop(k)
    this_namedtuple_maker = collections.namedtuple(name, mapping.keys())
    return this_namedtuple_maker(**mapping)


class ProtectedDict(UserDict):
    """ A class that exists just to tell `structure` not to eat it.
    `structure` eats all dicts you give it, recursively; but what if you
    actually want a dictionary in there? This will stop it. Just do
    ProtectedDict({...}) or ProtectedDict(kwarg=foo).
    """


def structure_from_kwargs(**kwargs):
    return structure(kwargs)
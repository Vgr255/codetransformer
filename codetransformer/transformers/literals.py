from collections import OrderedDict
from decimal import Decimal
from itertools import islice
from textwrap import dedent

from .. import instructions
from ..core import CodeTransformer
from ..patterns import pattern,  matchany, var
from ..utils.instance import instance


IN_COMPREHENSION = 'in_comprehension'


class overloaded_dicts(CodeTransformer):
    """Transformer that allows us to overload dictionary literals.

    This acts by creating an empty map and then inserting every
    key value pair in order.

    The code that is generated will turn something like:

    {k_0: v_0, k_1: v_1, ..., k_n: v_n}

    into:

    _tmp = astype()
    _tmp[k_0] = v_0
    _tmp[k_1] = v_1
    ...
    _tmp[k_n] = v_n
    _tmp  # leaves the map on the stack.

    Parameters
    ----------
    astype : callable
        The constructor for the type to create.

    Examples
    --------
    >>> from collections import OrderedDict
    >>> ordereddict_literals = overloaded_dicts(OrderedDict)
    >>> @ordereddict_literals
    ... def f():
    ...     return {'a': 1, 'b': 2, 'c': 3}
    ...
    >>> f()
    OrderedDict([('a', 1), ('b', 2), ('c', 3)])
    """
    def __init__(self, astype):
        super().__init__()
        self.astype = astype

    @pattern(instructions.BUILD_MAP, matchany[var], instructions.MAP_ADD)
    def _start_comprehension(self, instr, *instrs):
        yield instructions.LOAD_CONST(self.astype).steal(instr)
        # TOS  = self.astype

        yield instructions.CALL_FUNCTION(0)
        # TOS  = m = self.astype()

        yield instructions.STORE_FAST('__map__')

        *body, map_add = instrs
        yield from body
        # TOS  = k
        # TOS1 = v

        yield instructions.LOAD_FAST('__map__').steal(map_add)
        # TOS  = __map__
        # TOS1 = k
        # TOS2 = v

        yield instructions.ROT_TWO()
        # TOS  = k
        # TOS1 = __map__
        # TOS2 = v

        yield instructions.STORE_SUBSCR()
        self.begin(IN_COMPREHENSION)

    @pattern(instructions.BUILD_MAP)
    def _build_map(self, instr):
        yield instructions.LOAD_CONST(self.astype).steal(instr)
        # TOS  = self.astype

        yield instructions.CALL_FUNCTION(0)
        # TOS  = m = self.astype()

        yield from (instructions.DUP_TOP(),) * instr.arg
        # TOS  = m
        # ...
        # TOS[instr.arg] = m

    @pattern(instructions.RETURN_VALUE, startcodes=(IN_COMPREHENSION,))
    def _return_value(self, instr):
        yield instructions.LOAD_FAST('__map__').steal(instr)
        # TOS  = __map__

        yield instr

    try:

        @pattern(instructions.STORE_MAP)
        def _store_map(self, instr):
            # TOS  = k
            # TOS1 = v
            # TOS2 = m
            # TOS3 = m

            yield instructions.ROT_THREE().steal(instr)
            # TOS  = v
            # TOS1 = m
            # TOS2 = k
            # TOS3 = m

            yield instructions.ROT_THREE()
            # TOS  = m
            # TOS1 = k
            # TOS2 = v
            # TOS3 = m

            yield instructions.ROT_TWO()
            # TOS  = k
            # TOS1 = m
            # TOS2 = v
            # TOS3 = m

            yield instructions.STORE_SUBSCR()
            # TOS  = m

    except AttributeError:

        def _store_map(self, instr):
            raise NotImplementedError("The 'STORE_MAP' opcode is "
                                      "not supported in CPython 3.5+")

ordereddict_literals = overloaded_dicts(OrderedDict)


def _format_constant_docstring(type_):
    return dedent(
        """
        Transformer that applies a callable to each {type_} constant in the
        transformed code object

        Parameters
        ----------
        astype : callable
            A callable to be applied to {type_} literals.
        """
    ).format(type_=type_.__name__)


class _ConstantTransformerBase(CodeTransformer):

    def __init__(self, astype):
        super().__init__()
        self.astype = astype

    def transform_consts(self, consts):
        # This is all one expression.
        return super().transform_consts(
            tuple(
                frozenset(self.transform_consts(tuple(const)))
                if isinstance(const, frozenset)
                else self.transform_consts(const)
                if isinstance(const, tuple)
                else self.astype(const)
                if isinstance(const, self._type)
                else const
                for const in consts
            )
        )


def overloaded_constants(type_):
    """Factory for constant transformers that apply to a particular type.

    Parameters
    ----------
    type_ : type
        The type to overload.

    Returns
    -------
    transformer : subclass of CodeTransformer
        A new code transformer class that will overload the provided
        literal types.
    """
    typename = type_.__name__
    if not typename.endswith('s'):
        typename += 's'

    return type(
        "overloaded_" + typename,
        (_ConstantTransformerBase,), {
            '_type': type_,
            '__doc__': _format_constant_docstring(type_),
        },
    )


overloaded_strs = overloaded_constants(str)
haskell_strs = overloaded_strs(tuple)
overloaded_bytes = overloaded_constants(bytes)
bytearray_literals = overloaded_bytes(bytearray)
overloaded_floats = overloaded_constants(float)
decimal_literals = overloaded_floats(Decimal)


def _start_comprehension(self, *instrs):
    yield from instrs
    self.begin(IN_COMPREHENSION)


def _return_value(self, instr):
    # TOS  = collection

    yield instructions.LOAD_CONST(self.astype).steal(instr)
    # TOS  = self.astype
    # TOS1 = collection

    yield instructions.ROT_TWO()
    # TOS  = collection
    # TOS1 = self.astype

    yield instructions.CALL_FUNCTION(1)
    # TOS  = self.astype(collection)

    yield instr


# Added as a method for overloaded_build
def _build(self, instr):
    yield instr
    # TOS  = new_list

    yield instructions.LOAD_CONST(self.astype)
    # TOS  = astype
    # TOS1 = new_list

    yield instructions.ROT_TWO()
    # TOS  = new_list
    # TOS1 = astype

    yield instructions.CALL_FUNCTION(1)
    # TOS  = astype(new_list)


def overloaded_build(type_, add_name=None):
    """Factory for constant transformers that apply to a given
    build instruction.

    Parameters
    ----------
    type_ : type
        The object type to overload the construction of. This must be one of
        "buildable" types, or types with a "BUILD_*" instruction.
    add_name : str, optional
        The suffix of the instruction tha adds elements to the collection.
        For example: 'add' or 'append'

    Returns
    -------
    transformer : subclass of CodeTransformer
        A new code transformer class that will overload the provided
        literal types.
    """
    typename = type_.__name__
    dict_ = OrderedDict()

    try:
        build_instr = getattr(instructions, 'BUILD_' + typename.upper())
    except AttributeError:
        raise TypeError("type %s is not buildable" % typename)

    if add_name is not None:
        try:
            add_instr = getattr(
                instructions,
                '_'.join((typename, add_name)).upper(),
            )
        except AttributeError:
            TypeError("type %s is not addable" % typename)

        dict_['_start_comprehension'] = pattern(
            build_instr, matchany[var], add_instr,
        )(_start_comprehension)
        dict_['_return_value'] = pattern(
            instructions.RETURN_VALUE, startcodes=(IN_COMPREHENSION,),
        )(_return_value)
    else:
        add_instr = None

    dict_['_build'] = pattern(build_instr)(_build)

    if not typename.endswith('s'):
        typename = typename + 's'

    return type(
        'overloaded_' + typename,
        (overloaded_constants(type_),),
        dict_,
    )

overloaded_slices = overloaded_build(slice)
overloaded_lists = overloaded_build(list, 'append')
overloaded_sets = overloaded_build(set, 'add')


# Add a special method for set overloader.
def transform_consts(self, consts):
    consts = super(overloaded_sets, self).transform_consts(consts)
    return tuple(
        # Always pass a thawed set so mutations can happen inplace.
        self.astype(set(const)) if isinstance(const, frozenset) else const
        for const in consts
    )

overloaded_sets.transform_consts = transform_consts
del transform_consts
frozenset_literals = overloaded_sets(frozenset)


overloaded_tuples = overloaded_build(tuple)


# Add a special method for the tuple overloader.
def transform_consts(self, consts):
    consts = super(overloaded_tuples, self).transform_consts(consts)
    return tuple(
        self.astype(const) if isinstance(const, tuple) else const
        for const in consts
    )

overloaded_tuples.transform_consts = transform_consts
del transform_consts


@instance
class islice_literals(CodeTransformer):
    """Transformer that turns slice indexing into an islice object.

    Examples
    --------
    >>> from codetransformer.transformers.literals import islice_literals
    >>> @islice_literals
    ... def f():
    ...     return map(str, (1, 2, 3, 4))[:2]
    ...
    >>> f()
    <itertools.islice at ...>
    >>> tuple(f())
    ('1', '2')
    """
    @pattern(instructions.BINARY_SUBSCR)
    def _binary_subscr(self, instr):
        yield instructions.LOAD_CONST(self._islicer).steal(instr)
        # TOS  = self._islicer
        # TOS1 = k
        # TOS2 = m

        yield instructions.ROT_THREE()
        # TOS  = k
        # TOS1 = m
        # TOS2 = self._islicer

        yield instructions.CALL_FUNCTION(2)
        # TOS  = self._islicer(m, k)

    @staticmethod
    def _islicer(m, k):
        if isinstance(k, slice):
            return islice(m, k.start, k.stop, k.step)

        return m[k]

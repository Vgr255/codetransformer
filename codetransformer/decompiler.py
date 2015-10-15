import ast
from functools import singledispatch
from itertools import chain
import types

from .code import Code, Flags
from . import instructions as instrs
from .utils.functional import partition
from .utils.immutable import immutable
from codetransformer import a as showa, d as showd  # noqa


class DecompilationError(Exception):
    pass


class DecompilationContext(immutable,
                           defaults={
                               "in_function": False,
                               "next_store_is_function": False}):
    """
    Value representing the context of the current decompilation run.
    """
    __slots__ = (
        'in_function',
        'next_store_is_function',
    )


def pycode_to_body(co, context):
    """
    Convert a Python code object to a list of AST body elements.
    """
    code = Code.from_pycode(co)
    return instrs_to_body(code.instrs, context)


def instrs_to_body(instrs, context):
    """
    Convert a list of Instruction objects to a list of AST body nodes.
    """
    stack = []
    body = []
    for instr in instrs:
        newcontext = process_instr(instr, stack, body, context)
        if newcontext:
            context = newcontext

    if stack:
        raise DecompilationError(
            "Non-empty stack at the end of instrs_to_body(): %s." % stack
        )
    return body


@singledispatch
def process_instr(instr, stack, current_body, context):
    raise AssertionError(
        "process_instr() passed a non-instruction argument %s" % type(instr)
    )


@process_instr.register(instrs.Instruction)
def _instr(instr, stack, current_body, context):
    raise DecompilationError(
        "Don't know how to decompile instructions of type %s" % type(instr)
    )


@process_instr.register(instrs.BINARY_SUBSCR)
@process_instr.register(instrs.LOAD_ATTR)
@process_instr.register(instrs.LOAD_GLOBAL)
@process_instr.register(instrs.LOAD_CONST)
@process_instr.register(instrs.LOAD_FAST)
@process_instr.register(instrs.LOAD_NAME)
@process_instr.register(instrs.LOAD_DEREF)
@process_instr.register(instrs.LOAD_CLOSURE)
@process_instr.register(instrs.BUILD_TUPLE)
@process_instr.register(instrs.BUILD_SET)
@process_instr.register(instrs.BUILD_LIST)
@process_instr.register(instrs.BUILD_MAP)
@process_instr.register(instrs.STORE_MAP)
@process_instr.register(instrs.CALL_FUNCTION)
@process_instr.register(instrs.BUILD_SLICE)
def _push(instr, stack, current_body, context):
    """
    Just push these instructions onto the stack for further processing
    downstream.
    """
    stack.append(instr)


@process_instr.register(instrs.MAKE_FUNCTION)
@process_instr.register(instrs.MAKE_CLOSURE)
def _make_function(instr, stack, current_body, context):
    """
    Set the `post_make_function` flag on context, then `instr` onto the stack.
    """
    stack.append(instr)
    return context.update(next_store_is_function=True)


@process_instr.register(instrs.STORE_FAST)
@process_instr.register(instrs.STORE_NAME)
def _store(instr, stack, current_body, context):
    # This is set by MAKE_FUNCTION nodes to register that the next `STORE_NAME`
    # should create a FunctionDef node.
    if context.next_store_is_function:
        function_builders = pop_arguments(instr, stack)
        *default_builders, load_name, load_code, make_func = function_builders
        current_body.append(
            make_function(
                default_builders,
                load_name,
                load_code,
                make_func,
            )
        )
        return context.update(next_store_is_function=False)

    current_body.append(
        ast.Assign(
            targets=[ast.Name(id=instr.arg, ctx=ast.Store())],
            value=make_expr(pop_arguments(instr, stack)),
        )
    )


@process_instr.register(instrs.STORE_GLOBAL)
def _store_global(instr, stack, current_body, context):
    if context.in_function:
        current_body.append(ast.Global(names=[instr.arg]))
    return _store(instr, stack, current_body, context)


@process_instr.register(instrs.STORE_DEREF)
def _store_deref(instr, stack, current_body, context):
    if instr.vartype == 'cell':
        current_body.append(ast.Nonlocal(names=[instr.arg]))
    return _store(instr, stack, current_body, context)


@process_instr.register(instrs.STORE_ATTR)
def _store_attr(instr, stack, current_body, context):
    target = make_expr(stack)
    rhs = make_expr(stack)
    current_body.append(
        ast.Assign(
            targets=[
                ast.Attribute(
                    value=target,
                    attr=instr.arg,
                    ctx=ast.Store(),
                )
            ],
            value=rhs,
        )
    )


@process_instr.register(instrs.STORE_SUBSCR)
def _store_subscr(instr, stack, current_body, context):

    slice_ = make_slice(stack)
    collection = make_expr(stack)
    rhs = make_expr(stack)

    current_body.append(
        ast.Assign(
            targets=[
                ast.Subscript(
                    value=collection,
                    slice=slice_,
                    ctx=ast.Store(),
                ),
            ],
            value=rhs,
        ),
    )


@process_instr.register(instrs.POP_TOP)
def _pop(instr, stack, current_body, context):
    current_body.append(
        ast.Expr(value=make_expr(pop_arguments(instr, stack)))
    )


@process_instr.register(instrs.RETURN_VALUE)
def _return(instr, stack, current_body, context):
    if not context.in_function:
        _check_stack_for_module_return(stack)
        stack.pop()
        return

    current_body.append(ast.Return(value=make_expr(stack)))


def make_expr(instrs):
    """
    Convert a sequence of instructions into AST expressions.
    """
    toplevel = instrs.pop()
    return _make_expr(toplevel, instrs)


@singledispatch
def _make_expr(toplevel, stack_builders):
    raise DecompilationError(
        "Don't know how to build expression for %s" % toplevel
    )


@_make_expr.register(instrs.BUILD_TUPLE)
def _make_expr_tuple(toplevel, stack_builders):
    elts = [make_expr(stack_builders) for _ in range(toplevel.arg)]
    elts.reverse()
    return ast.Tuple(elts=elts, ctx=ast.Load())


@_make_expr.register(instrs.BUILD_SET)
def _make_expr_set(toplevel, stack_builders):
    elts = [make_expr(stack_builders) for _ in range(toplevel.arg)]
    elts.reverse()
    return ast.Set(elts=elts)


@_make_expr.register(instrs.BUILD_LIST)
def _make_expr_list(toplevel, stack_builders):
    elts = [make_expr(stack_builders) for _ in range(toplevel.arg)]
    elts.reverse()
    return ast.List(ctx=ast.Load(), elts=elts)


@_make_expr.register(instrs.BUILD_MAP)
def _make_expr_empty_dict(toplevel, stack_builders):
    """
    This should only be hit for empty dicts.  Anything else should hit the
    STORE_MAP handler instead.
    """
    if toplevel.arg:
        raise DecompilationError(
            "make_expr() called with nonzero BUILD_MAP arg %d" % toplevel.arg
        )

    if stack_builders:
        raise DecompilationError(
            "Unexpected stack_builders for BUILD_MAP(0): %s" % stack_builders
        )
    return ast.Dict(keys=[], values=[])


@_make_expr.register(instrs.STORE_MAP)
def _make_expr_dict(toplevel, stack_builders):
    build_map, *stack_builders = stack_builders + [toplevel]
    if not isinstance(build_map, instrs.BUILD_MAP):
        raise DecompilationError(
            "Found STORE_MAP without BUILD_MAP: %s" % (
                [build_map] + stack_builders
            )
        )

    # Convert iterator of (k, v) pairs into list of keys and list of values.
    keys, values = map(list, zip(*_dict_kv_pairs(build_map, stack_builders)))

    # Keys and values are emitted in reverse order of how they appear in the
    # AST.
    keys.reverse()
    values.reverse()

    return ast.Dict(keys=keys, values=values)


def _dict_kv_pairs(build_instr, builders):

    for _ in range(build_instr.arg):
        popped = builders.pop()
        if not isinstance(popped, instrs.STORE_MAP):
            raise DecompilationError(
                "Expected a STORE_MAP but got %s" % popped
            )

        yield make_expr(builders), make_expr(builders)


@_make_expr.register(instrs.LOAD_DEREF)
@_make_expr.register(instrs.LOAD_NAME)
@_make_expr.register(instrs.LOAD_CLOSURE)
@_make_expr.register(instrs.LOAD_FAST)
@_make_expr.register(instrs.LOAD_GLOBAL)
def _make_expr_name(toplevel, stack_builders):
    return ast.Name(id=toplevel.arg, ctx=ast.Load())


@_make_expr.register(instrs.LOAD_ATTR)
def _make_expr_attr(toplevel, stack_builders):
    return ast.Attribute(
        value=make_expr(stack_builders),
        attr=toplevel.arg,
        ctx=ast.Load(),
    )


@_make_expr.register(instrs.BINARY_SUBSCR)
def _make_expr_getitem(toplevel, stack_builders):
    slice_ = make_slice(stack_builders)
    value = make_expr(stack_builders)
    return ast.Subscript(slice=slice_, value=value, ctx=ast.Load())


def make_slice(stack_builders):
    """
    Make an expression in the context of a slice.

    This mostly delegates to _make_expr, but wraps nodes in `ast.Index` or
    `ast.Slice` as appropriate.
    """
    return _make_slice(stack_builders.pop(), stack_builders)


@singledispatch
def _make_slice(toplevel, stack_builders):
    return ast.Index(_make_expr(toplevel, stack_builders))


@_make_slice.register(instrs.BUILD_SLICE)
def make_slice_build_slice(toplevel, stack_builders):
    return _make_expr(toplevel, stack_builders)


@_make_slice.register(instrs.BUILD_TUPLE)
def make_slice_tuple(toplevel, stack_builders):
    slice_ = _make_expr(toplevel, stack_builders)
    if isinstance(slice_, ast.Tuple):
        # a = b[c, d] generates Index(value=Tuple(...))
        # a = b[c:, d] generates ExtSlice(dims=[Slice(...), Index(...)])
        slice_ = normalize_tuple_slice(slice_)
    return slice_


def normalize_tuple_slice(node):
    """
    Normalize an ast.Tuple node representing the internals of a slice.

    Returns the node wrapped in an ast.Index.
    Returns an ExtSlice node built from the tuple elements if there are any
    slices.
    """
    if not any(isinstance(elt, ast.Slice) for elt in node.elts):
        return ast.Index(value=node)

    return ast.ExtSlice(
        [
            # Wrap non-Slice nodes in Index nodes.
            elt if isinstance(elt, ast.Slice) else ast.Index(value=elt)
            for elt in node.elts
        ]
    )


@_make_expr.register(instrs.BUILD_SLICE)
def _make_expr_build_slice(toplevel, stack_builders):
    # Arg is always either 2 or 3.  If it's 3, then the first expression is the
    # step value.
    if toplevel.arg == 3:
        step = make_expr(stack_builders)
    else:
        step = None

    def normalize_empty_slice(node):
        """
        Convert LOAD_CONST(None) to just None.

        This normalizes slices of the form a[b:None] to just a[b:].
        """
        if isinstance(node, ast.NameConstant) and node.value is None:
            return None
        return node

    upper = normalize_empty_slice(make_expr(stack_builders))
    lower = normalize_empty_slice(make_expr(stack_builders))

    return ast.Slice(lower=lower, upper=upper, step=step)


@_make_expr.register(instrs.LOAD_CONST)
def _make_expr_const(toplevel, stack_builders):
    return _make_const(toplevel.arg)


@singledispatch
def _make_const(const):
    raise DecompilationError(
        "Don't know how to make constant node for %r." % (const,)
    )


@_make_const.register(int)
def _make_const_int(const):
    return ast.Num(n=const)


@_make_const.register(float)
def _make_const_float(const):
    return ast.Num(n=const)


@_make_const.register(complex)
def _make_const_complex(const):
    return ast.Num(n=const)


@_make_const.register(str)
def _make_const_str(const):
    return ast.Str(s=const)


@_make_const.register(bytes)
def _make_const_bytes(const):
    return ast.Bytes(s=const)


@_make_const.register(tuple)
def _make_const_tuple(const):
    return ast.Tuple(elts=list(map(_make_const, const)), ctx=ast.Load())


@_make_const.register(type(None))
def _make_const_none(none):
    return ast.NameConstant(value=None)


binops = frozenset([
    (instrs.BINARY_ADD, ast.Add),
    (instrs.BINARY_SUBTRACT, ast.Sub),
    (instrs.BINARY_MULTIPLY, ast.Mult),
    (instrs.BINARY_POWER, ast.Pow),
    (instrs.BINARY_TRUE_DIVIDE, ast.Div),
    (instrs.BINARY_FLOOR_DIVIDE, ast.FloorDiv),
    (instrs.BINARY_MODULO, ast.Mod),
    (instrs.BINARY_LSHIFT, ast.LShift),
    (instrs.BINARY_RSHIFT, ast.RShift),
    (instrs.BINARY_AND, ast.BitAnd),
    (instrs.BINARY_XOR, ast.BitXor),
    (instrs.BINARY_OR, ast.BitOr),
])


def _binop_handler(nodetype):
    """
    Factory function for binary operator handlers.
    """
    def _handler(toplevel, stack_builders):
        right = make_expr(stack_builders)
        left = make_expr(stack_builders)
        return ast.BinOp(left=left, op=nodetype(), right=right)
    return _handler

for instrtype, nodetype in binops:
    process_instr.register(instrtype)(_push)
    _make_expr.register(instrtype)(_binop_handler(nodetype))


def make_function(default_builders,
                  load_code_instr,
                  load_name_instr,
                  make_function_instr):
    """
    Construct a FunctionDef AST node from a sequence of the form:
    <default builders>, LOAD_CONST(code), LOAD_CONST(name), MAKE_FUNCTION().
    """
    _check_make_function_instrs(
        load_code_instr,
        load_name_instr,
        make_function_instr,
    )

    co = load_code_instr.arg
    name = load_name_instr.arg
    if name == '<lambda>':
        raise DecompilationError("Don't know how to decompile lambdas.")

    args, kwonly, varargs, varkwargs = paramnames(co)
    defaults, kw_defaults, annotations = default_exprs(
        make_function_instr,
        default_builders
    )

    return ast.FunctionDef(
        name=name.split('.')[-1],
        args=ast.arguments(
            args=[ast.arg(arg=arg, annotation=None) for arg in args],
            kwonlyargs=[ast.arg(arg=arg, annotation=None) for arg in kwonly],
            defaults=defaults,
            kw_defaults=list(map(kw_defaults.get, kwonly)),
            vararg=None if varargs is None else ast.arg(
                arg=varargs, annotation=None
            ),
            kwarg=None if varkwargs is None else ast.arg(
                arg=varkwargs, annotation=None
            ),
        ),
        body=dedupe_global_and_nonlocal_decls(
            pycode_to_body(
                co,
                DecompilationContext(
                    in_function=True,
                    next_store_is_function=False,
                )
            )
        ),
        decorator_list=[],
        returns=None,
    )


def is_declaration(node):
    return isinstance(node, (ast.Nonlocal, ast.Global))


def dedupe_global_and_nonlocal_decls(body):
    """
    Remove duplicate `ast.Global` and `ast.Nonlocal` nodes from the body of a
    function.
    """
    decls, non_decls = map(list, partition(is_declaration, body))
    globals_, nonlocals_ = partition(
        lambda n: isinstance(n, ast.Global), decls
    )

    deduped = []
    global_names = list(
        set(chain.from_iterable(node.names for node in globals_))
    )
    if global_names:
        deduped.append(ast.Global(global_names))

    nonlocal_names = list(
        set(chain.from_iterable(node.names for node in nonlocals_))
    )
    if nonlocal_names:
        deduped.append(ast.Nonlocal(nonlocal_names))

    return deduped + list(non_decls)


def default_exprs(make_function_instr, default_builders):
    """
    Get the AST expressions corresponding to the defaults and kwonly defaults
    for a function created by `make_function_instr`.
    """
    # Integer counts.
    n_defaults, n_kwonlydefaults, n_annotations = unpack_make_function_arg(
        make_function_instr.arg
    )
    if n_annotations:
        raise DecompilationError("Don't know how to decompile annotations.")

    kwonlys = {}
    while n_kwonlydefaults:
        default_expr = make_expr(default_builders)
        key_instr = default_builders.pop()
        if not isinstance(key_instr, instrs.LOAD_CONST):
            raise DecompilationError(
                "kwonlydefault key is not a LOAD_CONST: %s" % key_instr
            )
        if not isinstance(key_instr.arg, str):
            raise DecompilationError(
                "kwonlydefault key builder is not a "
                "'LOAD_CONST of a string: %s" % key_instr
            )

        kwonlys[key_instr.arg] = default_expr
        n_kwonlydefaults -= 1

    defaults = []
    while n_defaults:
        defaults.append(make_expr(default_builders))
        n_defaults -= 1

    defaults.reverse()

    return defaults, kwonlys, []  # Annotations not supported.


def unpack_make_function_arg(arg):
    """
    Unpack the argument to a MAKE_FUNCTION instruction.

    Parameters
    ----------
    arg : int
        The argument to a MAKE_FUNCTION instruction.

    Returns
    -------
    num_defaults, num_kwonly_default_pairs, num_annotations

    See Also
    --------
    https://docs.python.org/3/library/dis.html#opcode-MAKE_FUNCTION
    """
    return arg & 0xFF, (arg >> 8) & 0xFF, (arg >> 16) & 0x7FFF


def paramnames(co):
    """
    Get the parameter names from a pycode object.

    Returns a 4-tuple of (args, kwonlyargs, varargs, varkwargs).
    varargs and varkwargs will be None if the function doesn't take *args or
    **kwargs, respectively.
    """
    flags = co.co_flags
    varnames = co.co_varnames

    argcount, kwonlyargcount = co.co_argcount, co.co_kwonlyargcount
    total = argcount + kwonlyargcount

    args = varnames[:argcount]
    kwonlyargs = varnames[argcount:total]
    varargs, varkwargs = None, None
    if flags & Flags.CO_VARARGS:
        varargs = varnames[total]
        total += 1
    if flags & Flags.CO_VARKEYWORDS:
        varkwargs = varnames[total]

    return args, kwonlyargs, varargs, varkwargs


def _check_make_function_instrs(load_code_instr,
                                load_name_instr,
                                make_function_instr):
    """
    Validate the instructions passed to a make_function call.
    """

    # Validate load_code_instr.
    if not isinstance(load_code_instr, instrs.LOAD_CONST):
        raise TypeError(
            "make_function expected 'load_code_instr` to be a "
            "LOAD_CONST, but got %s" % load_code_instr,
        )
    if not isinstance(load_code_instr.arg, types.CodeType):
        raise TypeError(
            "make_function expected load_code_instr "
            "to load a code object, but got %s" % load_code_instr.arg,
        )

    # Validate load_name_instr
    if not isinstance(load_name_instr, instrs.LOAD_CONST):
        raise TypeError(
            "make_function expected 'load_name_instr` to be a "
            "LOAD_CONST, but got %s" % load_code_instr,
        )

    if not isinstance(load_name_instr.arg, str):
        raise TypeError(
            "make_function expected load_name_instr "
            "to load a string, but got %r instead" % load_name_instr.arg
        )

    # Validate make_function_instr
    if not isinstance(make_function_instr, (instrs.MAKE_FUNCTION,
                                            instrs.MAKE_CLOSURE)):
        raise TypeError(
            "make_function expected a MAKE_FUNCTION or MAKE_CLOSURE"
            "instruction, but got %s instead." % make_function_instr
        )


def pop_arguments(instr, stack):
    """
    Pop instructions off `stack` until we pop all instructions that will
    produce values popped by `instr`.
    """
    needed = instr.stack_effect
    if needed >= 0:
        raise DecompilationError(
            "%s is does not have a negative stack effect" % instr
        )

    for popcount, to_pop in enumerate(reversed(stack), start=1):
        needed += to_pop.stack_effect
        if not needed:
            break
    else:
        raise DecompilationError(
            "Reached end of stack without finding inputs to %s" % instr,
        )

    popped = stack[-popcount:]
    stack[:] = stack[:-popcount]

    return popped


def _check_stack_for_module_return(stack):
    """
    Verify that the stack is in the expected state before the dummy
    RETURN_VALUE instruction of a module or class.
    """
    fail = (
        len(stack) != 1
        or not isinstance(stack[0], instrs.LOAD_CONST)
        or stack[0].arg is not None
    )

    if fail:
        raise DecompilationError(
            "Reached end of non-function code "
            "block with unexpected stack: %s." % stack
        )

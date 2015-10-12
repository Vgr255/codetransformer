import ast
from functools import singledispatch

from .code import Code
from . import instructions as instrs


class DecompilationError(Exception):
    pass


def pycode_to_body(co, *, in_function):
    """
    Convert a Python code object to a list of AST body elements.
    """
    stack = []
    body = []
    code = Code.from_pycode(co)
    for instr in code:
        process_instr(instr, stack, body, in_function)

    if stack:
        raise DecompilationError(
            "Non-empty stack at the end of pycode_to_body(): %s." % stack
        )
    return body


@singledispatch
def process_instr(instr, stack, current_body, in_function):
    raise AssertionError(
        "process_instr() passed a non-instruction argument %s" % type(instr)
    )


@process_instr.register(instrs.Instruction)
def _instr(instr, stack, current_body, in_function):
    raise DecompilationError(
        "Don't know how to decompile instructions of type %s" % type(instr)
    )


@process_instr.register(instrs.LOAD_CONST)
@process_instr.register(instrs.LOAD_NAME)
@process_instr.register(instrs.MAKE_FUNCTION)
@process_instr.register(instrs.BUILD_TUPLE)
@process_instr.register(instrs.BUILD_SET)
@process_instr.register(instrs.BUILD_LIST)
@process_instr.register(instrs.BUILD_MAP)
@process_instr.register(instrs.STORE_MAP)
def _push(instr, stack, current_body, in_function):
    """
    Just push these instructions onto the stack for further processing
    downstream.
    """
    stack.append(instr)


@process_instr.register(instrs.STORE_FAST)
@process_instr.register(instrs.STORE_NAME)
def _assign(instr, stack, current_body, in_function):
    current_body.append(
        ast.Assign(
            targets=[ast.Name(id=instr.arg, ctx=ast.Store())],
            value=make_expr(pop_arguments(instr, stack)),
        )
    )


@process_instr.register(instrs.POP_TOP)
def _pop(instr, stack, current_body, in_function):
    current_body.append(
        ast.Expr(value=make_expr(pop_arguments(instr, stack)))
    )


@process_instr.register(instrs.RETURN_VALUE)
def _return(instr, stack, current_body, in_function):
    if not in_function:
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


def _binop_handler(nodetype):
    """
    Factory function for binary operator handlers.
    """
    def _handler(toplevel, stack_builders):
        right = make_expr(stack_builders)
        left = make_expr(stack_builders)
        return ast.BinOp(left=left, op=nodetype(), right=right)
    return _handler


binops = [
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
]
for instrtype, nodetype in binops:
    process_instr.register(instrtype)(_push)
    _make_expr.register(instrtype)(_binop_handler(nodetype))


def _dict_kv_pairs(build_instr, builders):

    for _ in range(build_instr.arg):
        popped = builders.pop()
        if not isinstance(popped, instrs.STORE_MAP):
            raise DecompilationError(
                "Expected a STORE_MAP but got %s" % popped
            )

        yield make_expr(builders), make_expr(builders)


@_make_expr.register(instrs.LOAD_NAME)
def _make_expr_name(toplevel, stack_builders):
    return ast.Name(id=toplevel.arg, ctx=ast.Load())


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

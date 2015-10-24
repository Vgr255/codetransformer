import ast
from collections import deque
from functools import singledispatch
from itertools import chain
import types

from toolz import complement, compose, curry
import toolz.curried.operator as op

from .code import Code, Flags
from . import instructions as instrs
from .utils.functional import partition, not_a, is_a
from .utils.immutable import immutable
from codetransformer import a as showa, d as showd  # noqa


class DecompilationError(Exception):
    pass


class DecompilationContext(immutable,
                           defaults={
                               "in_function": False,
                               "make_function_context": None,
                               "top_of_loop": None}):

    """
    Value representing the context of the current decompilation run.
    """
    __slots__ = (
        'in_function',
        'make_function_context',
        'top_of_loop',
    )


class MakeFunctionContext(immutable):
    __slots__ = ('closure',)


def pycode_to_body(co, context):
    """
    Convert a Python code object to a list of AST body elements.
    """
    code = Code.from_pycode(co)
    return instrs_to_body(deque(code.instrs), context)


def instrs_to_body(instrs, context):
    """
    Convert a list of Instruction objects to a list of AST body nodes.
    """
    stack = []
    body = []
    process_instrs(instrs, stack, body, context)

    if stack:
        raise DecompilationError(
            "Non-empty stack at the end of instrs_to_body(): %s." % stack
        )
    return body


def process_instrs(queue, stack, body, context):
    """
    Process instructions from the instruction queue.
    """
    next_instr = queue.popleft
    while queue:
        newcontext = _process_instr(next_instr(), queue, stack, body, context)
        if newcontext is not None:
            context = newcontext


@singledispatch
def _process_instr(instr, queue, stack, body, context):
    raise AssertionError(
        "process_instr() passed a non-instruction argument %s" % type(instr)
    )


@_process_instr.register(instrs.Instruction)
def _instr(instr, queue, stack, body, context):
    raise DecompilationError(
        "Don't know how to decompile instructions of type %s" % type(instr)
    )


@_process_instr.register(instrs.EXTENDED_ARG)
def _process_instr_extended_arg(instr, queue, stack, body, context):
    """We account for EXTENDED_ARG when constructing Code objects."""
    pass


@_process_instr.register(instrs.UNPACK_SEQUENCE)
def _process_instr_unpack_sequence(instr, queue, stack, body, context):
    body.append(make_assignment(instr, queue, stack))


@_process_instr.register(instrs.IMPORT_NAME)
def _process_instr_import_name(instr, queue, stack, body, context):
    """
    Process an IMPORT_NAME instruction.

    Side Effects
    ------------
    Pops two instuctions from `stack`
    Consumes instructions from `queue` to the end of the import statement.
    Appends an ast.Import or ast.ImportFrom node to `body`.
    """
    # If this is "import module", fromlist is None.
    # If this this is "from module import a, b fromlist will be ('a', 'b').
    fromlist = stack.pop().arg

    # (optimization) level argument to __import__.  Should be 0, 1, or 2.
    # Doesn't affect the AST.
    level = stack.pop().arg

    module = instr.arg
    if fromlist is None:  # Regular import.
        _pop_import_LOAD_ATTRs(module, queue)
        store = queue.popleft()
        update_global_and_nonlocal_decls(store, body, context)
        asname = store.arg
        body.append(
            ast.Import(
                names=[
                    ast.alias(
                        name=module,
                        asname=(
                            asname if asname != module.split('.')[0] else None
                        )
                    ),
                ],
                level=level,
            ),
        )
        return
    elif fromlist == ('*',):  # From module import *.
        expect_instruction(
            queue.popleft(), instrs.IMPORT_STAR, "after IMPORT_NAME"
        )
        body.append(
            ast.ImportFrom(
                module=module,
                names=[ast.alias(name='*', asname=None)],
                level=level,
            ),
        )
        return

    # Consume a pair of IMPORT_FROM, STORE_NAME instructions for each entry in
    # fromlist.
    names = list(map(make_importfrom_alias(queue, body, context), fromlist))
    body.append(ast.ImportFrom(module=module, names=names, level=level))

    # Remove the final POP_TOP of the imported module.
    expect_instruction(queue.popleft(), instrs.POP_TOP, "after 'from import'")


def _pop_import_LOAD_ATTRs(module_name, queue):
    """
    Pop LOAD_ATTR instructions for an import of the form::

        import a.b.c as d

    which should generate bytecode like this::

        1           0 LOAD_CONST               0 (0)
                    3 LOAD_CONST               1 (None)
                    6 IMPORT_NAME              0 (a.b.c.d)
                    9 LOAD_ATTR                1 (b)
                   12 LOAD_ATTR                2 (c)
                   15 LOAD_ATTR                3 (d)
                   18 STORE_NAME               3 (d)

    We don't need the LOAD_ATTRs to generate the correct AST, so we just throw
    it away.
    """
    popped = popwhile(is_a(instrs.LOAD_ATTR), queue, side='left')
    if popped:
        expected = module_name.split('.', maxsplit=1)[1]
        actual = '.'.join(map(op.attrgetter('arg'), popped))
        if expected != actual:
            raise DecompilationError(
                "Decompiling import of module %s, but LOAD_ATTRS imply %s" % (
                    expected, actual,
                )
            )


@curry
def make_importfrom_alias(queue, body, context, name):
    """
    Make an ast.alias node for the names list of an ast.ImportFrom.

    Parameters
    ----------
    queue : deque
        Instruction Queue
    body : list
        Current body.
    context : DecompilationContext
    name : str
        Expected name of the IMPORT_FROM node to be popped.

    Returns
    -------
    alias : ast.alias

    Side Effects
    ------------
    Consumes IMPORT_FROM and STORE_NAME instructions from queue.
    """
    import_from, store = queue.popleft(), queue.popleft()
    expect_instruction(
        import_from, instrs.IMPORT_FROM, "after IMPORT_NAME"
    )
    update_global_and_nonlocal_decls(store, body, context)
    if not import_from.arg == name:
        raise DecompilationError(
            "IMPORT_FROM name mismatch. Expected %r, but got %s." % (
                name, import_from,
            )
        )
    return ast.alias(
        name=name,
        asname=store.arg if store.arg != name else None,
    )


@_process_instr.register(instrs.UNARY_NOT)
@_process_instr.register(instrs.BINARY_SUBSCR)
@_process_instr.register(instrs.LOAD_ATTR)
@_process_instr.register(instrs.LOAD_GLOBAL)
@_process_instr.register(instrs.LOAD_CONST)
@_process_instr.register(instrs.LOAD_FAST)
@_process_instr.register(instrs.LOAD_NAME)
@_process_instr.register(instrs.LOAD_DEREF)
@_process_instr.register(instrs.LOAD_CLOSURE)
@_process_instr.register(instrs.BUILD_TUPLE)
@_process_instr.register(instrs.BUILD_SET)
@_process_instr.register(instrs.BUILD_LIST)
@_process_instr.register(instrs.BUILD_MAP)
@_process_instr.register(instrs.STORE_MAP)
@_process_instr.register(instrs.CALL_FUNCTION)
@_process_instr.register(instrs.CALL_FUNCTION_VAR)
@_process_instr.register(instrs.CALL_FUNCTION_KW)
@_process_instr.register(instrs.CALL_FUNCTION_VAR_KW)
@_process_instr.register(instrs.BUILD_SLICE)
def _push(instr, queue, stack, body, context):
    """
    Just push these instructions onto the stack for further processing
    downstream.
    """
    stack.append(instr)


@_process_instr.register(instrs.MAKE_FUNCTION)
@_process_instr.register(instrs.MAKE_CLOSURE)
def _make_function(instr, queue, stack, body, context):
    """
    Set a make_function_context, then push onto the stack.
    """
    stack.append(instr)
    return context.update(
        make_function_context=MakeFunctionContext(
            closure=isinstance(instr, instrs.MAKE_CLOSURE),
        )
    )


@_process_instr.register(instrs.STORE_FAST)
@_process_instr.register(instrs.STORE_NAME)
@_process_instr.register(instrs.STORE_DEREF)
@_process_instr.register(instrs.STORE_GLOBAL)
def _store(instr, queue, stack, body, context):
    update_global_and_nonlocal_decls(instr, body, context)
    # This is set by MAKE_FUNCTION nodes to register that the next `STORE_NAME`
    # should create a FunctionDef node.
    if context.make_function_context is not None:
        body.append(
            make_function(
                pop_arguments(instr, stack),
                **context.make_function_context.to_dict()
            ),
        )
        return context.update(make_function_context=None)

    body.append(make_assignment(instr, queue, stack))


def make_assignment(instr, queue, stack):
    """
    Make an ast.Assign node.
    """
    return ast.Assign(
        targets=[make_assign_target(instr, queue)],
        value=make_expr(stack),
    )


@singledispatch
def make_assign_target(instr, queue):
    """
    Make an AST node for the LHS of an assignment beginning at `instr`.
    """
    raise DecompilationError("Can't make assignment for %s." % instr)


@make_assign_target.register(instrs.STORE_FAST)
@make_assign_target.register(instrs.STORE_NAME)
@make_assign_target.register(instrs.STORE_DEREF)
@make_assign_target.register(instrs.STORE_GLOBAL)
def make_assign_target_store(instr, queue):
    return ast.Name(id=instr.arg, ctx=ast.Store())


@make_assign_target.register(instrs.UNPACK_SEQUENCE)
def make_assign_target_unpack(instr, queue):
    return ast.Tuple(
        elts=[
            make_assign_target(queue.popleft(), queue)
            for _ in range(instr.arg)
        ],
        ctx=ast.Store(),
    )


@singledispatch
def update_global_and_nonlocal_decls(instr, body, context):
    raise DecompilationError(
        "Can't update global/nonlocal declarations with %s" % instr
    )


@update_global_and_nonlocal_decls.register(instrs.STORE_DEREF)
def _update_decls_deref(instr, body, context):
    if instr.vartype == 'cell':
        body.append(ast.Nonlocal(names=[instr.arg]))


@update_global_and_nonlocal_decls.register(instrs.STORE_GLOBAL)
def _update_decls_global(instr, body, context):
    if context.in_function:
        body.append(ast.Global(names=[instr.arg]))


@update_global_and_nonlocal_decls.register(instrs.STORE_FAST)
@update_global_and_nonlocal_decls.register(instrs.STORE_NAME)
def _update_decls_noop(instr, body, context):
    pass


@_process_instr.register(instrs.STORE_ATTR)
def _store_attr(instr, queue, stack, body, context):
    target = make_expr(stack)
    rhs = make_expr(stack)
    body.append(
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


@_process_instr.register(instrs.STORE_SUBSCR)
def _store_subscr(instr, queue, stack, body, context):

    slice_ = make_slice(stack)
    collection = make_expr(stack)
    rhs = make_expr(stack)

    body.append(
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


@_process_instr.register(instrs.POP_TOP)
def _pop(instr, queue, stack, body, context):
    body.append(
        ast.Expr(value=make_expr(pop_arguments(instr, stack)))
    )


@_process_instr.register(instrs.RETURN_VALUE)
def _return(instr, queue, stack, body, context):
    if not context.in_function:
        _check_stack_for_module_return(stack)
        stack.pop()
        return

    body.append(ast.Return(value=make_expr(stack)))


@_process_instr.register(instrs.BREAK_LOOP)
def _jump_break_loop(instr, queue, stack, body, context):
    if context.top_of_loop is None:
        raise DecompilationError("BREAK_LOOP outside of loop.")
    body.append(ast.Break())


@_process_instr.register(instrs.JUMP_ABSOLUTE)
def _jump_absolute(instr, queue, stack, body, context):
    if instr.arg is context.top_of_loop:
        body.append(ast.Continue())
        return
    raise DecompilationError("Don't know how to decompile %s." % instr)


@_process_instr.register(instrs.SETUP_LOOP)
def _loop(instr, queue, stack, body, context):
    loop_type, loop_body, else_body = pop_loop_instrs(instr, queue)
    assert loop_type in ('for', 'while'), "Unknown loop type %r" % loop_type
    if loop_type == 'for':
        body.append(make_for_loop(loop_body, else_body, context))
    elif loop_type == 'while':
        body.append(make_while_loop(loop_body, else_body, context))


def make_for_loop(loop_body_instrs, else_body_instrs, context):
    """
    Make an ast.For node.
    """
    # Instructions from start until GET_ITER are the builders for the iterator
    # expression.
    iterator_expr = make_expr(
        popwhile(not_a(instrs.GET_ITER), loop_body_instrs, side='left')
    )

    # Next is the GET_ITER instruction, which we don't need.
    loop_body_instrs.popleft()

    # Next is FOR_ITER, which is the jump target for Continue nodes.
    top_of_loop = loop_body_instrs.popleft()

    store_instr = loop_body_instrs.popleft()
    if not isinstance(store_instr, instrs.STORE_NAME):
        raise DecompilationError(
            "Don't know how to bind loop variable with %s" % store_instr,
        )

    # TODO: Share code with assignments.
    target = ast.Name(id=store_instr.arg, ctx=ast.Store())

    body, orelse_body = make_loop_body_and_orelse(
        top_of_loop, loop_body_instrs, else_body_instrs, context
    )

    return ast.For(
        target=target,
        iter=iterator_expr,
        body=body,
        orelse=orelse_body,
    )


def make_loop_body_and_orelse(top_of_loop, body_instrs, else_instrs, context):
    """
    Make body and orelse lists for a for/while loop whose first instruction is
    `top_of_loop`.

    Parameters
    ----------
    top_of_loop : Instruction
        The first body of the loop.  For a for-loop, this should always be a
        FOR_ITER.  For a while loop, it's the first instruction of the stack
        builders for the loop test expression
    body_instrs : deque
        Queue of Instructions that form the body of the loop.  The last two
        elements of body_instrs should be a JUMP_ABSOLUTE to `top_of_loop` and
        a POP_BLOCK.
    else_instrs : deque
        Queue of Instructions that form the else block of the loop.  Should be
        an empty deque if there is no else block.
    context : DecompilationContext

    Returns
    -------
    body : list[ast.AST]
        List of ast nodes forming the loop body.
    orelse_body : list[ast.AST]
        List of ast nodes forming the else-block body.
    """
    # Remove the JUMP_ABSOLUTE and POP_BLOCK instructions at the bottom of the
    # loop.
    body_instrs.pop()
    body_instrs.pop()
    body = instrs_to_body(body_instrs, context.update(top_of_loop=top_of_loop))

    if else_instrs:
        else_body = instrs_to_body(else_instrs, context)
    else:
        else_body = []

    return body, else_body


def make_while_loop(test_and_body_instrs, else_body_instrs, context):
    """
    Make an ast.While node.

    Parameters
    ----------
    test_and_body_instrs : deque
        Queue of instructions forming the loop test expression and body.
    else_body_instrs : deque
        Queue of instructions forming the else block of the loop.
    context : DecompilationContext
    """
    top_of_loop = test_and_body_instrs[0]

    # The popped elements are the stack_builders for the loop test expression.
    # The top of the loop_body_instrs is either a POP_JUMP_IF_TRUE or a
    # POP_JUMP_IF_FALSE.
    test, body_instrs = make_while_loop_test_expr(test_and_body_instrs)
    body, orelse_body = make_loop_body_and_orelse(
        top_of_loop, body_instrs, else_body_instrs, context,
    )

    # while-else blocks are not yet supported or handled.
    return ast.While(test=test, body=body, orelse=orelse_body)


def make_while_loop_test_expr(loop_body_instrs):
    """
    Make an expression in the context of a while-loop test.

    Code of the form::

        while <expr>:
            <body>

    generates a POP_JUMP_IF_FALSE for the loop test, while code of the form::

        while not <expr>:
            <body>

    generates a POP_JUMP_IF_TRUE for the loop test.

    Code of the form::

        while True:
            <body>

    generates no jumps at all.
    """
    bottom_of_loop = loop_body_instrs[-1]
    is_jump_to_bottom = compose(op.is_(bottom_of_loop), op.attrgetter('arg'))

    # Consume instructions until we find a jump to the bottom of the loop.
    test_builders = deque(
        popwhile(complement(is_jump_to_bottom), loop_body_instrs, side='left')
    )
    # If we consumed the entire loop body without finding a jump, assume this
    # is a while True loop.  Return the rest of the instructions as the loop
    # body.
    if not loop_body_instrs:
        return ast.NameConstant(value=True), test_builders

    # Top of the body is either a POP_JUMP_IF_TRUE or POP_JUMP_IF_FALSE.
    jump = loop_body_instrs.popleft()
    expr = make_expr(test_builders)
    if isinstance(jump, instrs.POP_JUMP_IF_TRUE):
        return ast.UnaryOp(op=ast.Not(), operand=expr), loop_body_instrs
    else:
        return expr, loop_body_instrs


def pop_loop_instrs(setup_loop_instr, queue):
    """
    Determine whether setup_loop_instr is setting up a for-loop or a
    while-loop.  Then pop the loop instructions from queue.

    The easiest way to tell the difference is to look at the target of the
    JUMP_ABSOLUTE instruction at the end of the loop.  If it jumps to a
    FOR_ITER, then this is a for-loop.  Otherwise it's a while-loop.

    The jump we want is the first JUMP_ABSOLUTE instruction prior to the jump
    target of `setup_loop_instr`.

    Parameters
    ----------
    setup_loop_instr : instructions.SETUP_LOOP
        First instruction of the loop being parsed.
    queue : collections.deque
        Queue of unprocessed instructions.

    Returns
    -------
    loop_type : str, {'for', 'while'}
        The kind of loop being constructed.
    loop_instrs : deque
        The instructions forming body of the loop.
    else_instrs : deque
        The instructions forming the else-block of the loop.

    Side Effects
    ------------
    Pops all returned instructions from `queue`.
    """
    # Grab everything from left side of the queue until the jump target of
    # SETUP_LOOP.
    body = popwhile(op.is_not(setup_loop_instr.arg), queue, side='left')

    # Anything after the last POP_BLOCK instruction is the else-block.
    else_body = popwhile(not_a(instrs.POP_BLOCK), body, side='right')

    jump_to_top, pop_block = body[-2], body[-1]
    if not isinstance(jump_to_top, instrs.JUMP_ABSOLUTE):
        raise DecompilationError(
            "Penultimate instruction of loop body is "
            "%s, not JUMP_ABSOLUTE." % jump_to_top,
        )

    if not isinstance(pop_block, instrs.POP_BLOCK):
        raise DecompilationError(
            "Last instruction of loop body is "
            "%s, not pop_block." % pop_block,
        )

    loop_expr = jump_to_top.arg
    if isinstance(loop_expr, instrs.FOR_ITER):
        return 'for', body, else_body
    return 'while', body, else_body


def make_expr(stack_builders):
    """
    Convert a sequence of instructions into AST expressions.
    """
    return _make_expr(stack_builders.pop(), stack_builders)


@singledispatch
def _make_expr(toplevel, stack_builders):
    raise DecompilationError(
        "Don't know how to build expression for %s" % toplevel
    )


@_make_expr.register(instrs.UNARY_NOT)
def _make_expr_unary_not(toplevel, stack_builders):
    return ast.UnaryOp(
        op=ast.Not(),
        operand=make_expr(stack_builders),
    )


@_make_expr.register(instrs.CALL_FUNCTION)
def _make_expr_call_function(toplevel, stack_builders):
    keywords = make_call_keywords(stack_builders, toplevel.keyword)
    positionals = make_call_positionals(stack_builders, toplevel.positional)
    return ast.Call(
        func=make_expr(stack_builders),
        args=positionals,
        keywords=keywords,
        starargs=None,
        kwargs=None,
    )


@_make_expr.register(instrs.CALL_FUNCTION_VAR)
def _make_expr_call_function_var(toplevel, stack_builders):
    starargs = make_expr(stack_builders)
    keywords = make_call_keywords(stack_builders, toplevel.keyword)
    positionals = make_call_positionals(stack_builders, toplevel.positional)
    return ast.Call(
        func=make_expr(stack_builders),
        args=positionals,
        keywords=keywords,
        starargs=starargs,
        kwargs=None,
    )


@_make_expr.register(instrs.CALL_FUNCTION_KW)
def _make_expr_call_function_kw(toplevel, stack_builders):
    kwargs = make_expr(stack_builders)
    keywords = make_call_keywords(stack_builders, toplevel.keyword)
    positionals = make_call_positionals(stack_builders, toplevel.positional)
    return ast.Call(
        func=make_expr(stack_builders),
        args=positionals,
        keywords=keywords,
        starargs=None,
        kwargs=kwargs,
    )


@_make_expr.register(instrs.CALL_FUNCTION_VAR_KW)
def _make_expr_call_function_var_kw(toplevel, stack_builders):
    kwargs = make_expr(stack_builders)
    starargs = make_expr(stack_builders)
    keywords = make_call_keywords(stack_builders, toplevel.keyword)
    positionals = make_call_positionals(stack_builders, toplevel.positional)
    return ast.Call(
        func=make_expr(stack_builders),
        args=positionals,
        keywords=keywords,
        starargs=starargs,
        kwargs=kwargs,
    )


def make_call_keywords(stack_builders, count):
    """
    Make the keywords entry for an ast.Call node.
    """
    out = []
    for _ in range(count):
        value = make_expr(stack_builders)
        load_kwname = stack_builders.pop()
        if not isinstance(load_kwname, instrs.LOAD_CONST):
            raise DecompilationError(
                "Expected a LOAD_CONST, but got %s" % load_kwname
            )
        if not isinstance(load_kwname.arg, str):
            raise DecompilationError(
                "Expected LOAD_CONST of a str, but got %s." % load_kwname,
            )
        out.append(ast.keyword(arg=load_kwname.arg, value=value))
    out.reverse()
    return out


def make_call_positionals(stack_builders, count):
    """
    Make the args entry for an ast.Call node.
    """
    out = [make_expr(stack_builders) for _ in range(count)]
    out.reverse()
    return out


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
    """
    Yield (key, value) expression pairs for the elements of the dict literal
    generated by `instr`.
    """
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
    _process_instr.register(instrtype)(_push)
    _make_expr.register(instrtype)(_binop_handler(nodetype))


def make_function(function_builders, *, closure):
    """
    Construct a FunctionDef AST node from a sequence of the form:

    LOAD_CLOSURE, N times (when handling MAKE_CLOSURE)
    BUILD_TUPLE(N) (when handling MAKE_CLOSURE)
    <decorator builders> (optional)
    <default builders>, (optional)
    <annotation builders> (optional)
    LOAD_CONST(<tuple of annotated names>) (optional)
    LOAD_CONST(code),
    LOAD_CONST(name),
    MAKE_FUNCTION | MAKE_CLOSURE()
    <decorator calls> (optional)
    """
    decorator_calls = deque()
    while isinstance(function_builders[-1], instrs.CALL_FUNCTION):
        decorator_calls.appendleft(function_builders.pop())

    *builders, load_code_instr, load_name_instr, make_function_instr = (
        function_builders
    )

    _check_make_function_instrs(
        load_code_instr, load_name_instr, make_function_instr,
    )

    co = load_code_instr.arg
    name = load_name_instr.arg
    if name == '<lambda>':
        raise DecompilationError("Don't know how to decompile lambdas.")

    args, kwonly, varargs, varkwargs = paramnames(co)

    # Convert default and annotation builders to AST nodes.
    defaults, kw_defaults, annotations = make_defaults_and_annotations(
        make_function_instr,
        builders,
    )

    # Convert decorator function builders.  The stack is in reverse order.
    decorators = [make_expr(builders) for _ in decorator_calls]
    decorators.reverse()

    if closure:
        # There should be a tuple of closure cells still on the stack here.
        closure_cells = make_expr(builders)
        if not isinstance(closure_cells, ast.Tuple):
            raise DecompilationError(
                "Expected an ast.Tuple of closure cells, "
                "but got %s" % closure_cells,
            )

    # We should have consumed all our builders by this point.
    if builders:
        raise DecompilationError(
            "Unexpected leftover builders for %s: %s." % (
                make_function_instr, builders
            )
        )

    return ast.FunctionDef(
        name=name.split('.')[-1],
        args=ast.arguments(
            args=[ast.arg(arg=a, annotation=annotations.get(a)) for a in args],
            kwonlyargs=[
                ast.arg(arg=a, annotation=annotations.get(a)) for a in kwonly
            ],
            defaults=defaults,
            kw_defaults=list(map(kw_defaults.get, kwonly)),
            vararg=None if varargs is None else ast.arg(
                arg=varargs, annotation=annotations.get(varargs),
            ),
            kwarg=None if varkwargs is None else ast.arg(
                arg=varkwargs, annotation=annotations.get(varkwargs)
            ),
        ),
        body=dedupe_global_and_nonlocal_decls(
            pycode_to_body(
                co,
                DecompilationContext(in_function=True),
            )
        ),
        decorator_list=decorators,
        returns=annotations.get('return'),
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
        sorted(
            set(chain.from_iterable(node.names for node in globals_))
        )
    )
    if global_names:
        deduped.append(ast.Global(global_names))

    nonlocal_names = list(
        sorted(set(chain.from_iterable(node.names for node in nonlocals_)))
    )
    if nonlocal_names:
        deduped.append(ast.Nonlocal(nonlocal_names))

    return deduped + list(non_decls)


def make_defaults_and_annotations(make_function_instr, builders):
    """
    Get the AST expressions corresponding to the defaults, kwonly defaults, and
    annotations for a function created by `make_function_instr`.
    """
    # Integer counts.
    n_defaults, n_kwonlydefaults, n_annotations = unpack_make_function_arg(
        make_function_instr.arg
    )
    if n_annotations:
        # TOS should be a tuple of annotation names.
        load_annotation_names = builders.pop()
        annotations = dict(zip(
            reversed(load_annotation_names.arg),
            (make_expr(builders) for _ in range(n_annotations - 1))
        ))
    else:
        annotations = {}

    kwonlys = {}
    while n_kwonlydefaults:
        default_expr = make_expr(builders)
        key_instr = builders.pop()
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
        defaults.append(make_expr(builders))
        n_defaults -= 1

    defaults.reverse()

    return defaults, kwonlys, annotations


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


def expect_instruction(instr, expected, context):
    """
    Check that an instruction is of the expected type.
    """
    if not isinstance(instr, expected):
        raise DecompilationError(
            "Expected a {expected} instruction {context}. Got {instr}.".format(
                instr=instr, expected=expected, context=context,
            )
        )
    return instr


def popwhile(cond, queue, *, side):
    """
    Pop elements off a queue while `cond(nextelem)` is True.

    Parameters
    ----------
    cond : predicate
    queue : deque
    side : {'left', 'right'}

    Returns
    -------
    popped : deque

    Examples
    --------
    >>> from collections import deque
    >>> d = deque([1, 2, 3, 2, 1])
    >>> popwhile(lambda x: x < 3, d, side='left')
    deque([1, 2])
    >>> d
    deque([3, 2, 1])
    >>> popwhile(lambda x: x < 3, d, side='right')
    deque([2, 1])
    >>> d
    deque([3])
    """
    if side not in ('left', 'right'):
        raise ValueError("`side` must be one of 'left' or 'right'")

    out = deque()

    if side == 'left':
        popnext = queue.popleft
        pushnext = out.append
        nextidx = 0
    else:
        popnext = queue.pop
        pushnext = out.appendleft
        nextidx = -1

    while queue:
        if not cond(queue[nextidx]):
            break
        pushnext(popnext())
    return out

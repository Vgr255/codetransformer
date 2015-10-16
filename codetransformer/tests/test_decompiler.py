"""
Tests for decompiler.py
"""
from ast import AST, iter_fields, Module, parse
from itertools import product, zip_longest, combinations_with_replacement
from textwrap import dedent

import pytest
from toolz.curried.operator import add

from ..decompiler import DecompilationContext, paramnames, pycode_to_body


def make_indented_body(body_str):
    """
    Helper for generating an indented string to use as the body of a function.
    """
    return '\n'.join(
        map(
            add("    "),
            dedent(body_str).splitlines(),
        )
    )


def compare(computed, expected):
    """
    Assert that two AST nodes are the same.
    """
    assert type(computed) == type(expected)

    if isinstance(computed, list):
        for cv, ev in zip_longest(computed, expected):
            compare(cv, ev)
        return

    if not isinstance(computed, AST):
        assert computed == expected
        return

    for (cn, cv), (en, ev) in zip_longest(*map(iter_fields,
                                               (computed, expected))):
        assert cn == en
        compare(cv, ev)


def check(text, ast_text=None):
    """
    Check that compiling and disassembling `text` produces the same AST tree as
    calling ast.parse on `ast_text`.  If `ast_text` is not passed, use `text`
    for both.
    """
    if ast_text is None:
        ast_text = text

    ast = parse(ast_text)

    code = compile(text, '<test>', 'exec')

    decompiled_ast = Module(
        body=pycode_to_body(
            code,
            DecompilationContext(in_function=False)
        )
    )

    compare(decompiled_ast, ast)


def test_trivial_expr():
    check("a")


def test_trivial_expr_assign():
    check("a = b")


def test_unary_not():
    check("a = not b")
    check("a = not not b")
    check("a = not ((not a) + b)")


@pytest.mark.parametrize(
    'op', [
        '+',
        '-',
        '*',
        '**',
        '/',
        '//',
        '%',
        '<<',
        '>>',
        '&',
        '^',
        '|',
    ]
)
def test_binary_ops(op):
    check("a {op} b".format(op=op))
    check("a = b {op} c".format(op=op))
    check("a = (b {op} c) {op} d".format(op=op))
    check("a = b {op} (c {op} d)".format(op=op))


def test_string_literal():
    # A string literal as the first expression in a module generates a
    # STORE_NAME to __doc__.  We can't tell the difference between this and an
    # actual assignment to __doc__.
    check("'a'", "__doc__ = 'a'")
    check("'abc'", "__doc__ = 'abc'")

    check("a = 'a'")
    check("a = u'a'")


def test_bytes_literal():
    check("b'a'")
    check("b'abc'")
    check("a = b'a'")


def test_int_literal():
    check("1", "")  # This gets constant-folded out
    check("a = 1")
    check("a = 1 + b")
    check("a = b + 1")


def test_float_literal():
    check('1.0', "")   # This gets constant-folded out
    check("a = 1.0")
    check("a = 1.0 + b")
    check("a = b + 1.0")


def test_complex_literal():
    check('1.0j', "")  # This gets constant-folded out
    check("a = 1.0j")
    check("a = 1.0j + b")
    check("a = b + 1.0j")


def test_tuple_literals():
    check("()")
    check("(1,)")
    check("(a,)")
    check("(1, a)")
    check("(1, 'a')")
    check("((1,), a)")
    check("((1,(b,)), a)")


def test_set_literals():
    check("{1}")
    check("{1, 'a'}")
    check("a = {1, 'a'}")


def test_list_literals():
    check("[]")
    check("[1]")
    check("[a]")
    check("[[], [a, 1]]")


def test_dict_literals():
    check("{}")
    check("{a: b}")
    check("{a + a: b + b}")
    check("{a: b, c: d}")
    check("{1: 2, c: d}")
    check("{a: {b: c}, d: e}")
    check("{a: {b: [c, d, e]}}")


def test_paramnames():

    def foo(a, b):
        x = 1
        return x

    args, kwonlyargs, varargs, varkwargs = paramnames(foo.__code__)
    assert args == ('a', 'b')
    assert kwonlyargs == ()
    assert varargs is None
    assert varkwargs is None

    def bar(a, *, b):
        x = 1
        return x

    args, kwonlyargs, varargs, varkwargs = paramnames(bar.__code__)
    assert args == ('a',)
    assert kwonlyargs == ('b',)
    assert varargs is None
    assert varkwargs is None

    def fizz(a, **kwargs):
        x = 1
        return x

    args, kwonlyargs, varargs, varkwargs = paramnames(fizz.__code__)
    assert args == ('a',)
    assert kwonlyargs == ()
    assert varargs is None
    assert varkwargs == 'kwargs'

    def buzz(a, b=1, *args, c, d=3, **kwargs):
        x = 1
        return x

    args, kwonlyargs, varargs, varkwargs = paramnames(buzz.__code__)
    assert args == ('a', 'b')
    assert kwonlyargs == ('c', 'd')
    assert varargs == 'args'
    assert varkwargs == 'kwargs'


def test_simple_function():
    check(
        dedent(
            """\
            def foo(a, b):
                return a + b
            """
        )
    )


def test_annotations():
    check(
        dedent(
            """\
            def foo(a: b, c=1, *args: d, e:f, g:h=i, **kwargs: j):
                return a + c
            """
        )
    )



@pytest.mark.parametrize(
    "signature,body",
    product(
        [
            "()",
            "(a)",
            "(a, b)",
            "(*a, b)",
            "(a, **b)",
            "(*a, **b)",
            "(a=1, b=2, c=3)",
            "(a, *, b=1, c=2, d=3)",
            "(a, b=1, c=2, *, d, e=3, f, g=4)",
            "(a, b=1, *args, c, d=2, **kwargs)",
            "(a, b=c + d, *, e=f + g)",
        ],
        [
            """\
            return a + b
            """,
            """\
            x = 1
            y = 2
            return x + y
            """,
            """\
            x = 3
            def bar(m, n):
                global x
                x = 4
                return m + n + x
            return None
            """,
            """\
            def bar():
                x = 3
                def buzz():
                    nonlocal x
                    x = 4
                    return x
                return x
            return None
            """
        ],
    ),
)
def test_function_signatures(signature, body):
    check(
        dedent(
            """\
            def foo{signature}:
            {body}
            """
        ).format(signature=signature, body=make_indented_body(body))
    )


def test_decorators():
    check(
        dedent(
            """
            @decorator2
            @decorator1.attr1.attr2
            def foo(a, b=1, *, c, d=2):
                @decorator3
                def bar(c, d):
                    x = 1
                    return None
                return None
            """
        )
    )


def test_store_twice_to_global():
    check(
        dedent(
            """\
            x = 3
            def foo():
                global x
                x = 4
                x = 5
                return None
            """
        )
    )


def test_store_twice_to_nonlocal():
    check(
        dedent(
            """\
            def foo():
                x = 1
                def bar():
                    nonlocal x
                    x = 2
                    x = 3
                    return None
                return None
            """
        )
    )


def test_getattr():
    check("a.b")
    check("a.b.c")
    check("a.b.c + a.b.c")

    check("(1).real")
    check("1..real")

    check("(a + b).c")

    check("a = b.c")


def test_setattr():
    check("a.b = c")
    check("a.b.c = d")
    check("a.b.c = d.e.f")
    check("(a + b).c = (d + e).f")


def test_getitem():
    check("a = b[c]")
    check("a = b[c:]")
    check("a = b[:c]")
    check("a = b[c::]")
    check("a = b[c:d]")
    check("a = b[c:d:e]")

    check("a = b[c, d]")
    check("a = b[c:, d]")
    check("a = b[c:d:e, f:g:h, i:j:k]")

    check("a = b[c + d][e]")


def test_setitem():
    check("a[b] = c")
    check("b[c:] = a")
    check("b[:c] = a")
    check("b[c::] = a")
    check("b[c:d] = a")
    check("b[c:d:e] = a")

    check("b[c, d] = a")
    check("b[c:, d] = a")
    check("b[c:d:e, f:g:h, i:j:k] = a")

    check("b[c + d][e] = a")


LOOP_BODIES = tuple(map(
    '\n'.join,
    combinations_with_replacement(
        [
            "x = 1",
            "break",
            "continue",
            dedent(
                """\
                while u + v:
                    w = z
                """,
            ),
            dedent(
                """\
                for u in v:
                    w = z
                """,
            ),
        ],
        3,
    ),
))

ORELSE_BODIES = ["", "x = 3"]


@pytest.mark.parametrize(
    "loop,body,else_body",
    product(
        [
            "for a in b:",
            "for a in b.c.d:",
        ],
        LOOP_BODIES,
        ORELSE_BODIES,
    )
)
def test_for(loop, body, else_body):
    check(
        dedent(
            """\
            {loop}
            {body}
            {else_}
            {else_body}
            x = 4
            """
        ).format(
            loop=loop,
            body=make_indented_body(body),
            else_="else:" if else_body else "",
            else_body=make_indented_body(else_body) if else_body else "",
        )
    )


@pytest.mark.parametrize(
    "condition,body,else_body",
    product(
        [
            "a",
            "not a",
            "not not a",
            "a.b.c.d",
            "not a.b.c.d",
            "True",
        ],
        LOOP_BODIES,
        ORELSE_BODIES,
    )
)
def test_while(condition, body, else_body):
    check(
        dedent(
            """\
            while {condition}:
            {body}
            {else_}
            {else_body}
            x = 4
            """
        ).format(
            condition=condition,
            body=make_indented_body(body),
            else_="else:" if else_body else "",
            else_body=make_indented_body(else_body) if else_body else "",
        )
    )


def test_while_False():
    # The peephole optimizer removes while <falsey constant> blocks entirely.
    check(
        dedent(
            """\
            while False:
                x = 1
                y = 2
            """
        ),
        ""
    )

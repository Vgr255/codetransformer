"""
Tests for decompiler.py
"""
from ast import AST, iter_fields, Module, parse
import pytest

from ..decompiler import pycode_to_body


def compare(left, right):
    """
    Assert that two AST nodes are the same.
    """
    assert type(left) == type(right)

    if isinstance(left, list):
        for lv, rv in zip(left, right):
            compare(lv, rv)
        return

    if not isinstance(left, AST):
        assert left == right
        return

    for (ln, lv), (rn, rv) in zip(iter_fields(left), iter_fields(right)):
        assert ln == rn
        compare(lv, rv)


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
    decompiled_ast = Module(body=pycode_to_body(code, in_function=False))

    compare(ast, decompiled_ast)


def test_trivial_expr():
    check("a")


def test_trivial_expr_assign():
    check("a = b")


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
    check("1")
    check("a = 1")
    check("a = 1 + b")
    check("a = b + 1")


def test_float_literal():
    check('1.0')
    check("a = 1.0")
    check("a = 1.0 + b")
    check("a = b + 1.0")


def test_complex_literal():
    check('1.0j')
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
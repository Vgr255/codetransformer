from ast import iter_fields, AST, Name, Num, parse
import dis
from functools import partial
from io import StringIO
from itertools import chain
from operator import attrgetter
import sys
from types import CodeType

from codetransformer.code import Flags


INCLUDE_ATTRIBUTES_DEFAULT = False
INDENT_DEFAULT = '  '


def pformat_ast(node,
                include_attributes=INCLUDE_ATTRIBUTES_DEFAULT,
                indent=INDENT_DEFAULT):
    """
    Pretty-format an AST tree element

    Parameters
    ----------
    node : ast.AST
       Top-level node to render.
    include_attributes : bool, optional
        Whether to include node attributes.  Default False.
    indent : str, optional.
        Indentation string for nested expressions.  Default is two spaces.
    """
    def _fmt(node, prefix, level):

        def with_indent(*strs):
            return ''.join(((indent * level,) + strs))

        with_prefix = partial(with_indent, prefix)

        if isinstance(node, Name):
            # Special Case:
            # Render Name nodes on a single line.
            yield with_prefix(
                type(node).__name__,
                '(id=',
                repr(node.id),
                ', ctx=',
                type(node.ctx).__name__,
                '()),',
            )

        elif isinstance(node, Num):
            # Special Case:
            # Render Num nodes on a single line without names.
            yield with_prefix(
                type(node).__name__,
                '(%r),' % node.n,
            )

        elif isinstance(node, AST):
            fields_attrs = list(
                chain(
                    iter_fields(node),
                    iter_attributes(node) if include_attributes else (),
                )
            )
            if not fields_attrs:
                # Special Case:
                # Render the whole expression on one line if there are no
                # attributes.
                yield with_prefix(type(node).__name__, '(),')
                return

            yield with_prefix(type(node).__name__, '(')
            for name, value in fields_attrs:
                yield from _fmt(value, name + '=', level + 1)
            # Put a trailing comma if we're not at the top level.
            yield with_indent(')', ',' if level > 0 else '')

        elif isinstance(node, list):
            if not node:
                # Special Case:
                # Render empty lists on one line.
                yield with_prefix('[],')
                return

            yield with_prefix('[')
            yield from chain.from_iterable(
                map(partial(_fmt, prefix='', level=level + 1), node)
            )
            yield with_indent('],')
        else:
            yield with_prefix(repr(node), ',')

    return '\n'.join(_fmt(node, prefix='', level=0))


def _extend_name(prev, parent_co):
    return prev + (
        '.<locals>.' if parent_co.co_flags & Flags.CO_NEWLOCALS else '.'
    )


def pprint_ast(node,
               include_attributes=INCLUDE_ATTRIBUTES_DEFAULT,
               indent=INDENT_DEFAULT,
               file=None):
    """
    Pretty-print an AST tree.

    Parameters
    ----------
    node : ast.AST
       Top-level node to render.
    include_attributes : bool, optional
        Whether to include node attributes.  Default False.
    indent : str, optional.
        Indentation string for nested expressions.  Default is two spaces.
    file : None or file-like object, optional
        File to use to print output.  If the default of `None` is passed, we
        use sys.stdout.
    """
    if file is None:
        file = sys.stdout

    print(
        pformat_ast(
            node,
            include_attributes=include_attributes,
            indent=indent
        ),
        file=file,
    )


def walk_code(co, _prefix=''):
    """
    Traverse a code object, finding all consts which are also code objects.

    Yields pairs of (name, code object).
    """
    name = _prefix + co.co_name
    yield name, co
    yield from chain.from_iterable(
        walk_code(c, _prefix=_extend_name(name, co))
        for c in co.co_consts
        if isinstance(c, CodeType)
    )


def iter_attributes(node):
    attrs = node._attributes
    if not attrs:
        return

    yield from zip(attrs, attrgetter(*attrs)(node))


def a(text, mode='exec', indent='  ', file=None):
    """
    Interactive convenience for displaying the AST of a code string.

    Writes a pretty-formatted AST-tree to `file`.

    Parameters
    ----------
    text : str
        Text of Python code to render as AST.
    mode : {'exec', 'eval'}, optional
        Mode for `ast.parse`.  Default is 'exec'.
    indent : str, optional
        String to use for indenting nested expressions.  Default is two spaces.
    file : None or file-like object, optional
        File to use to print output.  If the default of `None` is passed, we
        use sys.stdout.
    """
    pprint_ast(parse(text, mode=mode), indent=indent, file=file)


def d(text, mode='exec', file=None):
    """
    Interactive convenience for displaying the disassembly of a code string.

    Compiles `text` and recursively traverses the result looking for `code`
    objects to render with `dis.dis`.

    Parameters
    ----------
    text : str
        Text of Python code to compile and disassemble.
    mode : {'exec', 'eval'}, optional
        Mode for `compile`.  Default is 'exec'.
    file : None or file-like object, optional
        File to use to print output.  If the default of `None` is passed, we
        use sys.stdout.
    """
    if file is None:
        file = sys.stdout

    for name, co in walk_code(compile(text, '<show>', mode)):
        print(name, file=file)
        print('-' * len(name), file=file)
        dis.dis(co, file=file)
        print('', file=file)


_DISPLAY_TEMPLATE = """\
====
Text
====

{text}

====================
Abstract Syntax Tree
====================

{ast}

===========
Disassembly
===========

{code}
"""


def display(text, mode='exec', file=None):
    """
    Show `text`, rendered as AST and as Bytecode.

    Parameters
    ----------
    text : str
        Text of Python code to render.
    mode : {'exec', 'eval'}, optional
        Mode for `ast.parse` and `compile`.  Default is 'exec'.
    file : None or file-like object, optional
        File to use to print output.  If the default of `None` is passed, we
        use sys.stdout.
    """

    if file is None:
        file = sys.stdout

    ast_section = StringIO()
    a(text, mode=mode, file=ast_section)

    code_section = StringIO()
    d(text, mode=mode, file=code_section)

    rendered = _DISPLAY_TEMPLATE.format(
        text=text,
        ast=ast_section.getvalue(),
        code=code_section.getvalue(),
    )
    print(rendered, file=file)

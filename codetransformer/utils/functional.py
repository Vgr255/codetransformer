from collections import deque


def scanl(f, n, ns):
    """Reduce ns by f starting with n yielding each intermediate value.

    tuple(scanl(f, n, ns))[-1] == reduce(f, ns, n)

    Parameters
    ----------
    f : callable
        A binary function.
    n : any
        The starting value.
    ns : iterable of any
        The iterable to scan over.

    Yields
    ------
    p : any
        The value of reduce(f, ns[:idx]) where idx is the current index.

    Examples
    --------
    >>> import operator as op
    >>> tuple(scanl(op.add, 0, (1, 2, 3, 4))
    (0, 1, 3, 6, 10)
    """
    yield n
    for m in ns:
        n = f(n, m)
        yield n


def flip(f, a, b):
    """Flips the argument order to f.

    Parameters
    ----------
    f : callable
        The function to call.
    a : any
        The second argument to f.
    b : any
        The first argument to f.

    Returns
    -------
    c : any
        f(b, a)
    """
    return f(b, a)


def complement(predicate):
    return lambda *args, **kwargs: not predicate(*args, **kwargs)


def partition(predicate, elems):
    """
    Partition an iterable into values matching a predicate and values not
    matching a predicate.

    FUTURE_OPTIMIZATION: Use Joe's ugly version of this.
    """
    elems = iter(elems)
    _trues, _falses = deque(), deque()

    def pop_all(d):
        while d:
            yield d.pop()

    def gen(pred, trues, falses):
        while True:
            yield from pop_all(trues)
            n = next(elems)
            if predicate(n):
                trues.appendleft(n)
            else:
                falses.appendleft(n)

    return (
        gen(predicate, _trues, _falses),
        gen(complement(predicate), _falses, _trues)
    )

import re
import sys
from collections import deque
from itertools import tee
from functools import wraps

import click

from mode import Mode
from constants import Publisher, PUBLISHER, console


def norm_id(n_traps, noise, example):
    """
    Get the standard filename prefix for a normal example from its number of
    traps, noise level, and example number.
    """
    return f'{int(n_traps)}-trap_wn={noise:.1f}_example={int(example)}'


def consume(iterator):
    """
    Consume an iterator
    """
    deque(iterator, maxlen=0)


def has_labels(signals):
    """
    Add a repeatable and centralized way to check if an example has labels (is
    real data)
    """
    return 'trap_0' in signals


def auto_n_traps(n_traps, mode, filename):
    """
    Automatically determine the number of traps based on the filename
    """

    if n_traps != 'auto':
        return int(n_traps)

    if mode == Mode.WN_STUDY:
        match = re.search(r'(\d)-trap', filename)
        print(filename)
        assert match, filename

        return int(match.group(1))

    if mode == Mode.METASTABLE:
        return 1

    if mode in (Mode.MISSING_LEVEL, Mode.COUPLED):
        return 2

    raise NotImplementedError(
        f'Cannot detect number of traps for {mode} and {filename}')


def subfig(i):
    # Useful when iterating traps
    # It's possible to catch the case where it's zero, but the other way around,
    # it's impossible to catch
    if i == 0:
        raise Exception('Argument should be creater than zero.')

    return {
        Publisher.IEEE: f'({chr(i + ord("a") - 1)})',
        Publisher.NATURE: fr'\textbf{{{chr(i + ord("a") - 1)}}}',
    }[PUBLISHER]


def pairwise(iterable):
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def run_click(command, *args):
    args = list(str(x) for x in args)
    ctx = click.Context(command)
    with ctx.scope():
        command.parse_args(ctx, args)
        return ctx.invoke(command, **ctx.params)


def print_filename_on_exception(func):
    @wraps(func)
    def wrapped_func(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception:

            # Has to be last arg because of thread pool
            filename = args[-1]

            sys.stderr.write(' '.join([
                f'Exception in "{func.__name__}"',
                f'while processing "{filename}"\n',
            ]))
            console.print_exception()

    return wrapped_func


def error(pred: float, truth: float) -> float:
    """
    Calculate the error between a true design value and an experimental
    value.
    """
    return abs(pred - truth) / truth


def nearest_multiple(multiple, number):
    return multiple * round(number / multiple)


def identity(x):
    """
    Functional-programming utility to return its only argument
    """

    return x

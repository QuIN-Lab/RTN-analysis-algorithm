"""
The files from the server are messy. Group all the amplitude-related files into
a single pandas dataframe. Many cells exist for the many ways the files can be
organized in the server.

Developed by Marcel Robitaille on 2022/03/18 Copyright © 2021 QuIN Lab
"""

import re
from pathlib import Path
from operator import sub
from itertools import chain
from multiprocessing.pool import Pool

import click
import pandas as pd

from constants import console
from example import Example
from utils import run_click, identity
from .digitization_tau_aggregation import parse_filename

# %%


def match_traps(theory, guess):
    """
    Associate fitting results to theoretical results
    Find the separation that is the closest to a theoretical one, then remove
    each from consideration
    """

    index_map = list(theory)
    theory = list(theory)
    guess = list(guess)

    while guess and theory:
        closest = {
            g: min(theory, key=lambda t: abs(g - t))
            for g in guess
        }
        g, t = min(closest.items(), key=lambda args: abs(sub(*args)))
        i = index_map.index(t)
        yield i, t, g, abs(t - g) / t
        index_map[i] = None
        theory.remove(t)
        guess.remove(g)

    if guess and not theory:
        print('WARNING!!!!')

    for t in theory:
        i = index_map.index(t)
        yield i, t, None, None
        index_map[i] = None
# %%



def calculate_amplitude_error(example):
    example = Example[pd.DataFrame](example)
    console.print(example)
    assert example.path.name.endswith('_decomp_data_traps.csv')

    guess = example.read()['sep'].to_list()
    theory = example.parameters.read()['amplitude'].to_list()

    rtn_type, true_n_traps, noise, example_number = \
        parse_filename(str(example.path))

    return [dict(
        rtn_type=rtn_type,
        n_traps=true_n_traps,
        predicted_n_traps=len(guess),
        noise=noise,
        example=example_number,
        trap=trap,
        theory=t,
        guess=g,
        amplitude_error=err,
    ) for trap, t, g, err in match_traps(theory, guess)]


@click.command()
@click.argument('files', nargs=-1, required=True, type=click.Path())
@click.argument('output', required=True, type=click.Path())
def aggregate_amplitude_error(files, output):
    """
    Calculate the amplitude error for a set of time-series predictions
    specified by FILES (files must end in `_decomp_data_traps.csv`).
    Save the resulting dataframe to OUTPUT (supports `.csv`, `.feather`).
    """

    output = Example(output)
    assert not output.path.is_dir()
    assert output.path.suffix in {'.csv', '.feather'}

    # Prevent shooting yourself in the foot
    # Since FILES takes a variable number of arguments, if you forget to specify
    # OUTPUT, the last file you intended for FILES will be overwritten with the
    # results
    assert not output.path.name.endswith('_decomp_data_traps.csv'), \
            'Maybe you forgot to specify OUTPUT?'

    with Pool() as p:
        df = pd.DataFrame(
            chain(*p.imap_unordered(calculate_amplitude_error, files)),
        )

    df.sort_values(by=['rtn_type', 'n_traps', 'noise', 'example', 'trap'], inplace=True)
    output.write(df)


if __name__ == '__main__':
    data_dir = Path.home() / 'OneDrive/02. QuIN_Research/31. Noise-RTN' / \
        '01. 2021_Algorithm paper/simulated_rtn/2022_04_08_pink_noise'

    run_click(
        aggregate_amplitude_error,
        *data_dir.glob('*_decomp_data_traps.csv'),
        data_dir / 'amplitude_error.csv',
    )


__all__ = ('aggregate_amplitude_error',)

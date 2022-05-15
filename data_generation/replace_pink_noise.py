"""
Go through the existing datasets and replace the white noise with 1/f pink
noise. Don't overwrite those, but rather save the dataframe with the replaced
pink noise in a new location

Developed by Marcel Robitaille on 2022/04/08 Copyright © 2021 QuIN Lab
"""

from functools import partial
from multiprocessing import Pool
from pathlib import Path

import click
import colorednoise as cn

from example import Example
from utils import consume


def replace_pink_noise_worker(example, results_dir):
    example = Example(example)
    print(example)
    assert example.path.name.endswith('_signals.feather'), example
    assert results_dir != example.path.parent

    df = example.read()

    # No need to divide/multiply by the smallest trap amplitude
    # We want to control the standard deviation of the pink noise
    # knowing the standard deviation of the white noise
    alpha = 1
    df['pink_noise'] = df.white_noise.std() * \
            cn.powerlaw_psd_gaussian(alpha, len(df))
    df['full_signal'] = df['rtn_sum'] + df['pink_noise']
    df = df.drop(columns=['white_noise'])

    Example(results_dir / example.path.name).write(df)


@click.command()
@click.argument('files', type=click.Path(), nargs=-1, required=True)
@click.argument('results-dir', type=click.Path(), required=True)
def replace_pink_noise(files, results_dir):
    """
    Go through the existing examples specified by FILES and replace the white
    noise with 1/f pink noise. Don't overwrite those, but rather save the
    dataframe with the replaced pink noise in wa new location

    """
    with Pool() as p:
        consume(p.imap_unordered(
            partial(replace_pink_noise_worker, results_dir=Path(results_dir)),
            files,
        ))


__all__ = ('replace_pink_noise',)

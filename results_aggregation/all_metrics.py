"""
Aggregate all metrics into a signle file.

Developed by Marcel Robitaille on 2022/04/26 Copyright © 2021 QuIN Lab
"""

from functools import reduce

import click
import pandas as pd

from example import Example


@click.command()
@click.argument('files', type=click.Path(), nargs=-1)
@click.argument('output-file', type=click.Path())
def merge_metrics(files, output_file):
    """
    Merge many results files (amplitude, digitization, tau) into a single one.
    Specify results files by FILES and write the result to OUTPUT-FILE.
    FILES should be `.csv` or `.feather` encoded pandas dataframes.
    Files will be merged on the intersection of their columns.
    Usually, these columns are `n_traps`, `noise`, `example`, `trap`,
    `artn_type`, `zone`.
    """

    output_file = Example(output_file)
    if output_file.path.exists() and not \
            click.confirm(f'Overwrite output file `{output_file.path}`?'):
        return

    dataframes = [Example[pd.DataFrame](f).read() for f in files]
    df = reduce(pd.merge, dataframes)
    Example(output_file).write(df)


__all__ = ('merge_metrics',)

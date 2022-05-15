"""
Click commands related to data generation

Developed by Marcel Robitaille on 2022/03/18 Copyright © 2021 QuIN Lab
"""

import random
from pathlib import Path

import click

from .generate_normal_data import save_rtn_signals
from .generate_metastable_data import save_metastable
from .generate_missing_level_data import save_missing_level
from .generate_coupled_data import save_coupled


@click.command()
@click.option('--examples', type=int, required=True,
              help='Number of examples to generate.')
@click.option('--signal-length', type=int, required=True,
              help='Length of signal to generate')
@click.option(
    '--noise', required=True,
    help='Level of white noise to add to the signal. '
    'Can either be the absolute value, or the low and high value '
    '(separated by `-`) to generate randomly.')
@click.option(
    '--variety', required=True,
    type=click.Choice(['metastable', 'missing-level', 'coupled', 'normal']),
)
@click.option('--traps', type=int, help='The number of traps', required=True)
@click.option('--out-dir', required=True, type=click.Path(),
              help='Where to save the data')
def generate_data(examples, noise, signal_length, variety, out_dir, traps):
    """
    Generate RTN data of different varieties.
    """

    function = {
        'metastable': save_metastable,
        'missing-level': save_missing_level,
        'coupled': save_coupled,
        'normal': save_rtn_signals,
    }[variety]

    out_dir = Path(out_dir)
    if not out_dir.exists() and click.confirm(f'Create directory `{out_dir}`?'):
        out_dir.mkdir(parents=True)

    def get_noise(noise):
        """
        Noise can either be a number, or a low and high value for a random
        number. The former is useful if you already know exactly what noise you
        want (tailored training data) and the latter is useful to generate a
        large quantity of unique validation data.
        """
        try:
            return float(noise)
        except ValueError:
            low, high = map(int, noise.split('-'))
            return lambda: random.randint(low, high) / 100

    for i in range(examples):
        print(i)
        function(
            example_number=i,
            out_dir=out_dir,
            signal_length=signal_length,
            n_traps=traps,
            noise=get_noise(noise),
        )


__all__ = ('generate_data',)

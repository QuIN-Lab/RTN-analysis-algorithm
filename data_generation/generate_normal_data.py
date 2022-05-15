"""
Generate 330 validation normal RTN examples

Developed by Marcel Robitaille on 2021/07/14 Copyright © 2021 QuIN Lab
"""

import random

import pandas as pd
from typing import cast

from example import Example
from constants import DATA_DIR, VALIDATION_SIGNAL_DURATION
from .generate_rtn_component import generate_rtn_component
from .add_noise import add_noise
from .result import DataGenerationResult


def generate_rtn_signals(amplitudes, signal_length, noise, **_kwargs) \
        -> DataGenerationResult:
    """
    Generate simple RTN signals. Useful for generating training data, as it
    takes amplitudes. This can either be the seed value for training data, or
    some random value for validation data.
    """

    parameters = pd.DataFrame([
        dict(
            amplitude=amplitude,
            tau_high=random.randint(50, 1_000),
            tau_low=random.randint(50, 1_000),
        )
        for _trap, amplitude in enumerate(amplitudes)
    ])
    parameters = parameters.sort_values(by='amplitude')
    parameters = cast(pd.DataFrame, parameters)
    parameters['trap'] = parameters.index
    parameters = parameters[['trap', 'amplitude', 'tau_high', 'tau_low']]

    # Generate each individual single-trap RTN signal
    df = pd.DataFrame({
        f'trap_{int(row.trap)}': generate_rtn_component(
            tau_low=row.tau_low,
            tau_high=row.tau_high,
            value_high=row.amplitude,
            value_low=0,
            sig_len=signal_length,
        )
        for _, row in parameters.iterrows()
    })

    # Add all independent RTN signals
    df['rtn_sum'] = df.sum(axis=1)

    # Generate white noise
    df, noise = add_noise(df=df, noise=noise)

    # Add white noise to RTN signal
    df['full_signal'] = df['rtn_sum'] + df['white_noise']

    return DataGenerationResult(df, parameters, noise)


def save_rtn_signals(n_traps, example_number, noise, out_dir=DATA_DIR,
                     signal_length=VALIDATION_SIGNAL_DURATION):
    """
    Generate RTN signals and do a lot of housekeeping around it. Wrapper around
    `generate_rtn_signals` to make it more useful for generating validation data
    we want to save rather than throw-away training data.
    """

    amplitudes = [random.randint(10, 100) for _ in range(n_traps)]

    df, parameters, noise = generate_rtn_signals(
        amplitudes=amplitudes,
        noise=noise,
        signal_length=signal_length,
    )

    example_parameters = Example(
        out_dir /
        f'{n_traps}-trap_wn={noise:.2f}_example={example_number:03d}'
        '_parameters.csv',
    )
    example_parameters.write(parameters)

    example = example_parameters.signals
    example.signals.write(df)

    return df

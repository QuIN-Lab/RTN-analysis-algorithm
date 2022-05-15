"""
Generate missing level aRTN data for validating our algorithm on anomalous
examples

Developed by Marcel Robitaille on 2022/02/18 Copyright © 2021 QuIN Lab
"""

import random

import pandas as pd
import numpy as np

from example import Example
from constants import ARTN_DATA_DIR, VALIDATION_SIGNAL_DURATION
from .generate_rtn_component import generate_rtn_component
from .add_noise import add_noise
from .result import DataGenerationResult


def generate_missing_level(amplitudes, noise, signal_length, sort=False,
                           **kwargs) -> DataGenerationResult:
    trap_params = [dict(
        amplitude=amplitude,
        tau_high=random.randint(50, 1000),
        tau_low=random.randint(50, 1000),
    ) for amplitude in amplitudes]
    parameters = pd.DataFrame(trap_params)
    parameters['trap'] = parameters.index

    df = pd.DataFrame({
        f'trap_{i}': generate_rtn_component(
            tau_low=d['tau_low'],
            tau_high=d['tau_high'],
            value_high=1,
            value_low=0,
            sig_len=signal_length,
        )
        for i, d in enumerate(trap_params)
    })

    # When generating validation data, we should figure out which trap was
    # faster and make that one the dependent one
    if sort:
        fast = np.argmin([d['tau_low'] + d['tau_high'] for d in trap_params])
        slow = (fast + 1) % 2
    # When we're trying to match the output of the GMM, don't sort
    else:
        fast = 1
        slow = 0

    df[f'trap_{fast}'] &= df[f'trap_{slow}']
    for i, d in enumerate(trap_params):
        df[f'trap_{i}'] *= d['amplitude']

    # Add all independent RTN signals
    df['rtn_sum'] = df.sum(axis=1)

    # Generate white noise
    df, noise = add_noise(df=df, noise=noise)

    # Add white noise to RTN signal
    df['full_signal'] = df['rtn_sum'] + df['white_noise']

    return DataGenerationResult(df, parameters, noise)


def save_missing_level(example_number, noise, out_dir=ARTN_DATA_DIR,
                       signal_length=VALIDATION_SIGNAL_DURATION, **kwargs):
    amplitudes = [random.randint(10, 100) for _ in range(2)]

    df, parameters, noise = generate_missing_level(
        amplitudes=amplitudes,
        noise=noise,
        signal_length=signal_length,
    )

    example = Example(
        out_dir /
        f'artn_missing_level_wn={noise:.2f}_example={example_number:02d}_'
        'signals.feather'
    )
    example.write(df)
    example.parameters.write(parameters)

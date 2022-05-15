"""
Generate coupled aRTN data for validating our algorithm on anomalous examples

Developed by Marcel Robitaille on 2022/02/18 Copyright © 2021 QuIN Lab
"""

import random
import pandas as pd

from example import Example
from constants import ARTN_DATA_DIR, VALIDATION_SIGNAL_DURATION, rng
from .generate_rtn_component import generate_rtn_component
from .add_noise import add_noise
from .result import DataGenerationResult


def generate_coupled(amplitudes, noise, signal_length, **kwargs):
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
            value_high=d['amplitude'],
            value_low=0,
            sig_len=signal_length,
        )
        for i, d in enumerate(trap_params)
    })

    try:
        # TODO: This is an ugly way to implement this
        # If the combined level is given as a third amplitude, use that
        # This is useful if we extract the combined level in GMM
        # and want to generate realistic training data
        combined_level = amplitudes[2]
    except IndexError:
        # Otherwise, shift the "normal" combined level by about 20%
        # This is the case of generating new coupled RTN
        combined_level = sum(amplitudes) * rng.uniform(0.8, 1.2)

    df[df.trap_0 + df.trap_1 == parameters.amplitude.sum()]['rtn_sum'] = \
        combined_level

    # Add all independent RTN signals
    df['rtn_sum'] = df.sum(axis=1)
    # plt.plot(df.rtn_sum)
    # plt.show()

    # Generate white noise
    df, noise = add_noise(df=df, noise=noise)

    # Add white noise to RTN signal
    df['full_signal'] = df['rtn_sum'] + df['white_noise']

    return DataGenerationResult(df, parameters, noise)


def save_coupled(example_number, noise, out_dir=ARTN_DATA_DIR,
                 signal_length=VALIDATION_SIGNAL_DURATION, **kwargs):
    amplitudes = [random.randint(10, 100) for _ in range(2)]

    df, parameters, noise = generate_coupled(
        amplitudes=amplitudes,
        noise=noise,
        signal_length=signal_length,
    )

    example = Example(
        out_dir /
        f'artn_coupled_wn={noise:.2f}_example={example_number:02d}_'
        'signals.feather'
    )
    example.write(df)
    example.parameters.write(parameters)

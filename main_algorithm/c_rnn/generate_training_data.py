"""
The purpose of this file is to generate training data for the RNN model based on
data learned in previous steps

Developed by Marcel Robitaille on 2022/03/17 Copyright © 2021 QuIN Lab
"""

import random
from typing import cast

from numpy.random import default_rng
import pandas as pd

from mode import Mode
from data_generation.generate_rtn_component import generate_rtn_component
from data_generation.generate_normal_data import generate_rtn_signals
from data_generation.generate_metastable_data import generate_metastable
from data_generation.generate_missing_level_data import generate_missing_level
from data_generation.generate_coupled_data import generate_coupled
from data_generation.result import DataGenerationResult
from data_generation.add_noise import add_noise
from constants import NUM_TRAINING_DATA_TO_GENERATE, TRAINING_SIGNAL_LENGTH

rng = default_rng()


def generate_data(noise, amplitudes, mode, filename_template, regenerate=True):
    if not regenerate:
        return [
            pd.read_feather(str(filename_template).format(n))
            for n in range(NUM_TRAINING_DATA_TO_GENERATE)
        ]

    f = {
        Mode.WN_STUDY: generate_rtn_signals,
        Mode.CNT_REAL_DATA: generate_rtn_signals,
        Mode.METASTABLE: generate_metastable,
        Mode.MISSING_LEVEL: generate_missing_level,
        Mode.COUPLED: generate_coupled,
        Mode.MUTUALLY_EXCLUSIVE: _generate_mutually_exclusive_rtn,
    }[mode]

    def generator():
        for n in range(NUM_TRAINING_DATA_TO_GENERATE):
            df, _parameters, _noise = f(
                amplitudes=amplitudes,
                noise=noise,
                sort=False,
                signal_length=TRAINING_SIGNAL_LENGTH,
            )

            df.to_feather(str(filename_template).format(n))
            yield df

    return list(generator())


def _generate_mutually_exclusive_rtn(amplitudes, noise, signal_length, **kwargs):
    trap_params = [dict(
        amplitude=amplitude,
        tau_high=random.randint(50, 1000),
        tau_low=random.randint(50, 1000),
    ) for amplitude in amplitudes]
    trap_params_df = pd.DataFrame(trap_params)
    trap_params_df['trap'] = trap_params_df.index
    trap_params_df['tau'] = \
        trap_params_df['tau_high'] + trap_params_df['tau_low']

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
    sorted_taus = cast(
        pd.DataFrame,
        trap_params_df.sort_values(by='tau', ascending=False)).index
    for i, trap in enumerate(sorted_taus):
        for other in sorted_taus[i+1:]:
            df[f'trap_{other}'] &= ~df[f'trap_{trap}']

    for i, d in enumerate(trap_params):
        df[f'trap_{i}'] *= d['amplitude']

    # Add all independent RTN signals
    df['rtn_sum'] = df.sum(axis=1)

    # Generate white noise
    df, noise = add_noise(df=df, noise=noise)

    # Add white noise to RTN signal
    df['full_signal'] = df['rtn_sum'] + df['white_noise']

    return DataGenerationResult(df, noise, trap_params_df)

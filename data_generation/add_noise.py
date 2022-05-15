"""
See the docstring of the function below.

Developed by Marcel Robitaille on 2022/03/18 Copyright © 2021 QuIN Lab
"""

from constants import rng


def add_noise(df, noise):
    """
    Add noise to signal. Allow noise to be a callable (useful for generating
    noise randomly) or an absolute value.
    """

    amplitude = sorted(set(df.rtn_sum))[1]
    noise = noise() if callable(noise) else noise
    df['white_noise'] = rng.normal(scale=amplitude * noise, size=len(df))
    return df, noise

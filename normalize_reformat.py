"""
Normalize (min/max between zero and one) the data and batch it up for use in a
reccurrent neural network.

Developed by Marcel Robitaille on 2022/03/18 Copyright © 2021 QuIN Lab
"""

import numpy as np

from constants import STEPS, BATCH_SIZE, console
from numpy_stride_tricks import sliding_window_view
from utils import has_labels


# %%
def _make_multiple(a, b):
    """
    Round down a to the nearest multiple of b
    """
    return (a // b) * b
# %%


def _make_batches(dataframes, keys, window_shape):

    # Take only certain keys from dataframe
    dataframes = [df[keys] for df in dataframes]

    # Take only a multiple of batch size per file
    length = min(
        _make_multiple(len(df) - STEPS, BATCH_SIZE)
        for df in dataframes
    ) + STEPS - 1
    dataframes = [df.iloc[:length] for df in dataframes]

    # Normalize between 0 and 1
    # We could also ensure min is zero, but I don't want to change this and
    # risk a regression
    dataframes = [df / df.max() for df in dataframes]

    batches = np.array([
        sliding_window_view(df.to_numpy(), window_shape=window_shape)
        for df in dataframes
    ])
    # Reshape to flatten multiple simulated files and each file's many chunks
    return batches.reshape(-1, *window_shape)


def reformat_generated_data(n_traps, generated_dfs):
    """
    Reformat generated training data from pandas DataFrame into batched
    numpy arrays
    """

    # Batch raw, noisy signal into chunks of STEPS=64
    feature_g = _make_batches(dataframes=generated_dfs, keys=['full_signal'],
                              window_shape=(STEPS, 1))
    assert feature_g.max() == 1, feature_g.max()

    # Batch individual trap components into chunks of (STEPS, n_traps)
    label_g = _make_batches(
        dataframes=generated_dfs,
        keys=[f'trap_{i}' for i in range(n_traps)],
        window_shape=(STEPS, n_traps),
    )
    assert label_g.max() == 1, label_g.max()
    assert set(label_g.ravel()) == {0, 1}

    return feature_g, label_g


def reformat_real_data(signals, detected_traps):
    """
    Reformat "real" (not generated training) data from pandas DataFrame into
    batched numpy arrays
    """

    n_traps = detected_traps if isinstance(detected_traps, int) \
        else len(detected_traps)
    feature_t = _make_batches(dataframes=[signals], keys=['full_signal'],
                              window_shape=(STEPS, 1))
    assert feature_t.max() == 1, feature_t.max()

    # This means real data, no labels
    if not has_labels(signals):
        return feature_t, None

    true_n_traps = len([c for c in signals.columns if c.startswith('trap_')])
    if true_n_traps < n_traps:
        console.print('[WARNING]: More traps detected than actually present! '
                      f'({true_n_traps}, {n_traps})')
        signals[[f'trap_{i}' for i in range(true_n_traps, n_traps)]] = 0

    label_t = _make_batches(dataframes=[signals], window_shape=(STEPS, n_traps),
                            keys=[f'trap_{i}' for i in range(n_traps)])
    dataframes = [
        df[[f'trap_{trap}' for trap in range(n_traps)]]
        for df in [signals]
    ]

    # Take only a multiple of batch size per file
    length = min(
        _make_multiple(len(df) - STEPS, BATCH_SIZE)
        for df in dataframes
    ) + STEPS - 1
    dataframes = [df.iloc[:length] for df in dataframes]

    # Normalize between 0 and 1
    # We could also ensure min is zero, but I don't want to change this and
    # risk a regression
    for df in dataframes:
        df[df > 0] = 1
    window_shape = (STEPS, n_traps)
    label_t = np.array([
        sliding_window_view(df.to_numpy(), window_shape=window_shape)
        for df in dataframes
    ])
    # Reshape to flatten multiple simulated files and each file's many chunks
    label_t = label_t.reshape(-1, *window_shape)
    assert label_t.max() == 1, label_t.max()
    assert set(label_t.ravel()) == {0, 1}

    return feature_t, label_t

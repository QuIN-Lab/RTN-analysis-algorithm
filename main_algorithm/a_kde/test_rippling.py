"""
The purpose of this file is to test that all rippling gets filtered out
correctly.

Developed by Marcel Robitaille on 2022/02/18 Copyright © 2021 QuIN Lab
"""

import numpy as np


from mode import Mode
from example import Example
from constants import DATA_DIR
from .process_kde import process_kde

# Examples that are susceptible to artifacts that look like rippling,
# but are true peaks
not_rippling = [
    (3, 0.4, 7, 6, [192]),
    (3, 0.5, 1, 7, [0]),
    (3, 0.7, 2, 6, [168]),
    (3, 0.7, 5, 6, [10]),
    (3, 0.8, 0, 6, [0]),
    (3, 0.8, 6, 8, [152]),
    (3, 0.8, 8, 7, [0]),
    (3, 0.9, 3, 5, [0]),
    (3, 1.0, 5, 4, [225]),
    (3, 1.0, 9, 4, [250]),
]

# Examples that are susceptible to rippling (false peaks)
ripple = [
    (3, 0.6, 7, 6),
    (2, 0.3, 7, 4),
]

for n_traps, noise, example, expected_len, expected in not_rippling:
    example = DATA_DIR / \
        f'{n_traps}-trap_wn={noise}_example={example}_signals.feather'
    example = Example(example)
    process_kde(example.path, mode=Mode.WN_STUDY)
    kde_data = example.kde_data.read().squeeze()
    peaks = kde_data.peaks_intensities.copy()

    assert len(peaks) == expected_len, f'Actual len: {len(peaks)} {example}'
    for x in expected:
        diff = np.abs(peaks - x)
        i = diff.argmin()
        assert diff[i] < 2, f'Nearest: {peaks[i]}'

for n_traps, noise, example, expected_len in ripple:
    example = DATA_DIR / \
        f'{n_traps}-trap_wn={noise}_example={example}_signals.feather'
    example = Example(example)
    process_kde(example.path, mode=Mode.WN_STUDY)
    kde_data = example.kde_data.read().squeeze()
    peaks = kde_data.peaks_intensities.copy()
    assert len(peaks) == expected_len, example.path.name

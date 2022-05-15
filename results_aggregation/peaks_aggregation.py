"""
Create a dataframe comparing the number of theoretical traps with the number of
detected traps.

Developed by Marcel Robitaille on 2022/03/18 Copyright © 2021 QuIN Lab
"""

from pathlib import Path

import numpy as np
import pandas as pd

# from constants import DATA_DIR

noise_levels = np.arange(0, 1.1, 0.1)


def generator():
    # for n_traps in np.arange(3) + 1:
    for n_traps in [1]:
        for noise in noise_levels:
            for example in range(10):
                with open(
                    Path('/tmp/hmm') /
                    f'{n_traps}-'
                    f'trap_wn={noise:.1f}_'
                    f'example={example}_hmm_timer.txt',
                    'r',
                ) as f:
                    yield n_traps, int(noise * 100), example, float(f.read())


df = pd.DataFrame(
    generator(),
    columns=('n_traps', 'noise', 'example', 'hmm_time'),
)
print(df)
df.to_csv('/tmp/hmm/2022_05_07_hmm_timing.csv', index=False)
# df.to_csv('results/num_peaks.csv', index=False)

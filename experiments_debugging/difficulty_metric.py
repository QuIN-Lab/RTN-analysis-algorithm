"""
My attempt to create a "difficulty" metric with which to sort the examples.
During the meeting, I found that there was a problem with my attempt. I have not
had the chance to correct this.

Developed by Marcel Robitaille on 2022/04/01 Copyright © 2021 QuIN Lab
"""

import math

import pandas as pd
import matplotlib.pyplot as plt

from example import Example
from constants import DATA_DIR
from utils import norm_id

# %% Plot by difficulty (I-1 - I-2) / Q_wn

# Some "all metrics" aggregated results file
df = pd.read_csv('all_metrics.csv')

def calculate_difficulty(row):
    noise = row.noise / 100
    parameters = Example(DATA_DIR / norm_id(
        n_traps=row.n_traps,
        noise=noise,
        example=row.example,
    )).parameters.read()
    amplitudes = parameters.amplitude
    return noise * math.prod(a / amplitudes.min() for a in amplitudes)

df['difficulty'] = df.apply(calculate_difficulty, axis=1)
df.sort_values('difficulty', inplace=True)

# %%

plt.close('all')
for column in ('amplitude_error', 'digitization_error'):
    fig, ax = plt.subplots()
    ax.scatter(df.difficulty, df[column], s=1)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel(r'Difficulty ($Q_\mathrm{wn} \cdot \prod_\mathrm{i\in{}traps} \Delta_\mathrm{RTN_{i}} / \min_\mathrm{traps}(\Delta_\mathrm{RTN})$)')
    ax.set_ylabel(rf'{column.replace("_", " ")} (\%)')
    fig.savefig(f'/tmp/difficulty_{column}.png', dpi=300)
    plt.close(fig)

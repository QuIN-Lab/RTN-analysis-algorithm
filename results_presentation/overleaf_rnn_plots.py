"""
Plot the predicted digitization before and after filtering out the false jumping
points.

Developed by Marcel Robitaille on 2022/02/29 Copyright © 2021 QuIN Lab
"""

import os
import subprocess
from operator import add
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

from tau_extraction import get_taus

plt.switch_backend('Agg')
plt.style.use('science')


try:
    DATA_DIR = Path(__file__).parent
except NameError:  # ipython terminal
    DATA_DIR = Path(os.getcwd())
DATA_DIR /= 'lu_figure_data'
figsize = (2, 1.80)
ylim = (-0.1, 1.25)


# %% Raw figure


fig, ax = plt.subplots(figsize=figsize)
df = pd.concat([
    pd.read_csv(
        './lu_figure_data/before_filtered_predicted_trap_siganl.csv',
        header=None,
    ).rename(columns={0: 'predicted'})[['predicted']],
    pd.read_csv(
        './lu_figure_data/before_filtered_ground_truth_trap_siganl.csv',
        header=None,
    ).rename(columns={0: 'truth'})[['truth']],
], axis=1).head(5001)

arrows = [i for i, t in add(*get_taus(df['predicted'])) if t < 20]
for x in arrows:
    ax.arrow(
        x=x,
        y=1.2,
        dx=0,
        dy=-0.05,
        head_length=0.05,
        head_width=100,
        head_starts_at_zero=True,
        color='k',
    )

ax.scatter(df.index, df['truth'], alpha=0.3, s=0.5, c='r')
ax.plot(df['predicted'])
ax.set_xticks([0, 1000, 2000, 3000, 4000, 5000])
ax.xaxis.set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, y: x // 1000))
ax.set_xlim(0, df.index.max())
ax.set_ylim(ylim)
ax.set_ylabel('Intensity (arb. unit)')
ax.set_xlabel(r'Signal step $(\times 10^3)$')

fig.savefig(DATA_DIR / 'rnn_output_before_filtered.pdf')
plt.close(fig)


# %% Filtered figure


fig, ax = plt.subplots(figsize=figsize)
df = pd.concat([
    pd.read_csv(
        './lu_figure_data/filtered_predicted_trap_siganl.csv',
        header=None,
    ).rename(columns={0: 'predicted'})[['predicted']],
    pd.read_csv(
        './lu_figure_data/filtered_ground_truth_trap_siganl.csv',
        header=None,
    ).rename(columns={0: 'truth'})[['truth']],
], axis=1).head(5001)

ax.scatter(df.index, df['truth'], alpha=0.3, s=0.5, c='r')
ax.plot(df['predicted'], c='g')
ax.set_xticks([0, 1000, 2000, 3000, 4000, 5000])
ax.xaxis.set_major_formatter(
    matplotlib.ticker.FuncFormatter(lambda x, y: x // 1000))
ax.set_xlim(0, df.index.max())
ax.set_ylim(ylim)
ax.set_ylabel('Intensity (arb. unit)')
ax.set_xlabel(r'Signal step $(\times 10^3)$')

fig.savefig(DATA_DIR / 'rnn_output_after_filtered.pdf')
plt.close(fig)


# %%

subprocess.call(['sh', '-c', 'cp /home/marcel/code/quin/CMOS_RTN/Lu/lu_figure_data/*.pdf "/home/marcel/Nextcloud/Waterloo/QuIN_Lab/CMOS RTN/Algorithm Paper V2/Overleaf/figs"'])

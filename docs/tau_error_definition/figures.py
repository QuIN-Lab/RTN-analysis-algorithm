"""
Create figures for the tau error definition change documentation

Developed by Marcel Robitaille on 2022/03/14 Copyright © 2021 QuIN Lab
"""

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from constants import DATA_DIR
from utils import norm_id, pairwise
from tau_extraction import get_taus

from results_presentation.data_only_plots import calculate_colors

plt.style.use(['science'])

out = Path(__file__).parent
out.mkdir(exist_ok=True, parents=True)

n_traps = 1
noise = 0.1
example = 1

colors = calculate_colors()


f = DATA_DIR / f'{norm_id(n_traps, noise, example)}_signals.feather'
df = pd.read_feather(f)
parameters = pd.read_csv(DATA_DIR /
                         f'{norm_id(n_traps, noise, example)}_parameters.csv')
print(parameters)
print(list(map(np.mean, get_taus(df.trap_0, df.index))))
df = df.iloc[30_000:]
df.reset_index(drop=True, inplace=True)
df = df.loc[:8_200]
print(df)

# Make the "other" type of figure, not for this documentation, but for the
# figure of the Nature paper
# This is a horrible variable name, but I wanted something short if I'm
# sprinkling it everywhere. I will find a better way to do this
other = True

for (name, column, c) in [
        ('noiseless', 'trap_0', 'C0'),
        ('noisy', 'full_signal', 'C0'),
        ('pred', 'trap_0', colors['pred'] if other else 'C1'),
        ('events', 'trap_0', colors['pred']),
]:
    plt.close()
    fig, ax = plt.subplots(figsize=(4.5, 1.5))

    ax.plot(df[column], c=c)
    if not other:
        ax.set_xticks((np.array([0, 2, 4, 6, 8]) * 1000))
        ax.set_xticklabels(np.array([0, 2, 4, 6, 8]), fontsize=8)
        ax.set_yticks([0, 10, 20])
        ax.set_yticklabels([0, 10, 20], fontsize=8)
        ax.set_xlabel(r'Signal step ($\times 10^3$)', fontsize=8, labelpad=0)
        ax.set_xlim(0, df.index.max())
    else:
        ax.axes.xaxis.set_visible(False)
        ax.axes.yaxis.set_visible(False)
        ax.axis('off')

    if name == 'events':
        low = 0
        high = df.trap_0.max()
        for i, row in df[df.trap_0.diff() != 0].iterrows():
            # ax.plot([i, i], [low, high], c='orange')
            padding = 0.05
            ax.annotate(
                '',
                xy=(i, high * padding),
                xytext=(i, (1 - padding) * high),
                arrowprops=dict(
                    arrowstyle='->' if row.trap_0 == 0 else '<-',
                    color=colors['events'],
                ),
            )


    if name == 'pred':
        diff = df[column].diff()
        fontsize = 8 if not other else 12

        for a, b in pairwise(df.index[diff != 0][1:]):
            tau = (b - a) / 1000
            if tau > 0.5:
                ax.annotate(
                    text=f'{tau:.1f}',
                    xy=((a + b) / 2, 10),
                    xytext=(0, 2),
                    textcoords='offset points',
                    ha='center',
                    va='bottom',
                    xycoords='data',
                    fontsize=fontsize,
                    color=colors['times'],
                )
                ax.annotate(
                    text='',
                    xy=(a - 50, 10),
                    xytext=(b + 50, 10),
                    xycoords='data',
                    fontsize=fontsize,
                    color=colors['times'],
                    arrowprops=dict(
                        arrowstyle='<|-|>',
                        mutation_scale=6,
                        # width=0.1,
                        # headwidth=4,
                        # headlength=2,
                        color=colors['times'] if other else 'black',
                    ),
                )
            elif abs(tau - 0.5) < 0.10:
                ax.annotate(
                    text='',
                    xy=(a, 2 if df[column][a] == 0 else 18),
                    ha='right',
                    va='center',
                    xycoords='data',
                    xytext=(-1, 0),
                    textcoords='offset points',
                    fontsize=fontsize,
                    arrowprops=dict(
                        width=1,
                        headwidth=3,
                        headlength=2,
                        color=colors['times'] if other else 'black',
                    ),
                    color=colors['times'],
                )
                ax.annotate(
                    text=f'{tau:.1f}',
                    xy=(b, 2 if df[column][a] == 0 else 18),
                    ha='left',
                    va='center',
                    xycoords='data',
                    xytext=(3, 0),
                    textcoords='offset points',
                    fontsize=fontsize,
                    arrowprops=dict(
                        width=1,
                        headwidth=3,
                        headlength=2,
                        color=colors['times'] if other else 'black',
                    ),
                    color=colors['times'],
                )
            else:
                ax.annotate(
                    text='',
                    xy=(b, 2 if df[column][a] == 0 else 18),
                    ha='left',
                    va='center',
                    xycoords='data',
                    xytext=(1, 0),
                    textcoords='offset points',
                    fontsize=fontsize,
                    arrowprops=dict(
                        width=1,
                        headwidth=3,
                        headlength=2,
                        color=colors['times'] if other else 'black',
                    ),
                    color=colors['times'],
                )
                ax.annotate(
                    text=f'{tau:.1f}',
                    xy=(a, 2 if df[column][a] == 0 else 18),
                    ha='right',
                    va='center',
                    xycoords='data',
                    xytext=(-3, 0),
                    textcoords='offset points',
                    fontsize=fontsize,
                    arrowprops=dict(
                        width=1,
                        headwidth=3,
                        headlength=2,
                        color=colors['times'] if other else 'black',
                    ),
                    color=colors['times'],
                )


    fig.tight_layout()
    fig.subplots_adjust(bottom=0.25, top=0.99, right=0.995)
    fig.savefig(
        out / f'{name}.pdf',
        transparent=True,
    )

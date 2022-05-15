"""
The purpose of this file is to plot a histogram of the extracted tau values

Developed by Marcel Robitaille on 2021/07/16 Copyright © 2021 QuIN Lab
"""

from glob import glob

import matplotlib.pyplot as plt
import pandas as pd

from tau_extraction import get_taus
from example import Example
from constants import DATA_DIR


def plot_taus():
    for example in glob(str(DATA_DIR / '*_signals.pkl')):
        example = Example(example)
        df = example.read()
        parameters = example.parameters.read()

        for _, trap in parameters.iterrows():
            # Pull times out of dataframes where a transition is detected
            t_ups, t_downs = get_taus(df[f'trap_{trap.trap}'], df.index)

            fig, (ax1, ax2) = plt.subplots(ncols=2, sharey=True)

            hist_y, hist_x, *_ = ax1.hist(
                t_ups,
                bins=20,
                label=fr'$\tau_{{high}}$ (expected: {trap.tau_high})',
                density=True,
            )
            example.with_name(f'trap={trap.trap}_tau_high_histogram.pkl') \
                .write(pd.DataFrame(dict(x=hist_x[1:], y=hist_y)))
            # hist_x = hist_x[1:]
            # params, _ = \
            #     curve_fit(lambda x, t, a: a * np.exp(-x/t), hist_x, hist_y)
            # params, _ = \
            #     curve_fit(lambda x, p: (1-p) ** (x-1) * p, hist_x, hist_y)
            # print(params)
            p = 2 / trap.tau_high
            y = (1 - p) ** (hist_x - 1) * p
            ax1.plot(
                hist_x,
                y,
                '--',
                label='Geometric distribution',
            )
            ax1.set_xlabel('Index')
            ax1.set_ylabel('Count')
            ax1.legend()

            hist_y, hist_x, *_ = ax2.hist(
                t_downs,
                bins=20,
                label=fr'$\tau_{{down}}$ (expected: {trap.tau_low})',
                density=True,
            )
            example.with_name(f'trap={trap.trap}_tau_low_histogram.pkl') \
                .write(pd.DataFrame(dict(x=hist_x[1:], y=hist_y)))

            p = 1 / trap.tau_high
            y = (1 - p) ** hist_x * p
            ax2.plot(
                hist_x,
                y,
                '--',
                label='Geometric distribution',
            )
            ax2.set_xlabel('Index')
            ax2.legend()

            fig.tight_layout()
            fig.savefig(example.with_name(
                f'trap={trap.trap}_tau_distribution.png').path)
            plt.close(fig)


# plot_taus()

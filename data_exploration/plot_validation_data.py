"""
Create simple time series plots of our validation dataset

Developed by Marcel Robitaille on 2022/02/18 Copyright © 2021 QuIN Lab
"""

from glob import glob

import matplotlib.pyplot as plt

from constants import DATA_DIR
from example import Example


def generate_plots():
    for example in glob(str(DATA_DIR / '*_signals.pkl')):
        print(example)

        df = Example(example).read()

        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(df.index, df['full_signal'], lw=0.5)
        ax.plot(df.index, df['rtn_sum'], lw=0.5)
        ax.set_xlim(0, df.index.max())
        ax.set_xlabel('Signal step')
        ax.set_ylabel('Intensity')
        fig.tight_layout()
        fig.savefig(example.with_name('signals.png').path, dpi=800)
        plt.close(fig)

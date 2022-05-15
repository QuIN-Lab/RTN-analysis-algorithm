"""
Plot simple time-series figure of aRTN data from aRTN validation dataset

Developed by Marcel Robitaille on 2022/02/15 Copyright © 2021 QuIN Lab
"""

from multiprocessing import Pool
from glob import glob

import matplotlib.pyplot as plt

from example import Example
from constants import ARTN_DATA_DIR


def plot_artn(f):
    example = Example(f)
    print(example)
    df = example.read()
    print(df)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(df.index, df.full_signal)
    ax.set_xlim(0, df.index.max())
    ax.set_ylabel('Intensity')
    ax.set_xlabel('Signal step')

    fig.savefig(example.with_name('time-series.png').path, dpi=300)
    ax.set_xlim(0, 200_000)
    fig.savefig(example.with_name('time-series_0-200000.png').path, dpi=300)
    plt.close(fig)


with Pool() as p:
    p.map(plot_artn, glob(str(ARTN_DATA_DIR / '*signals.feather')))

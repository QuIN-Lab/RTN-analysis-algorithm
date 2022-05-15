"""
Plot simple time series figures for CNT real data

Developed by Marcel Robitaille on 2022/02/16 Copyright © 2021 QuIN Lab
"""

from glob import glob

import matplotlib.pyplot as plt
import pandas as pd

from constants import CNT_DATA_DIR


for f in glob(str(CNT_DATA_DIR / '*.csv')):
    if 'decomp_data_traps' in f:
        continue
    print(f)
    df = pd.read_csv(f)
    print(df)

    df['time'] = df.newTime
    df['full_signal'] = df.AI * 1e6
    df = df[['time', 'full_signal']]
    df.to_feather(f.replace('.csv', '_signals.feather'))

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(df.time, df.full_signal)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel(r'Current ($\mu{}A$)')
    ax.set_xlim(0, df.time.max())
    fig.savefig(f.replace('csv', 'png'), dpi=300)
    plt.close(fig)

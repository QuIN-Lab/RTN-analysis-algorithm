"""
Plot the metastable plot like Fig. 5 b) in our algorithm paper EDL submission.

Developed by Marcel Robitaille on 2022/03/18 Copyright © 2021 QuIN Lab
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
from rich import print

from figure_constants import LABEL_FONT_SIZE


plt.switch_backend('Qt5Agg')
plt.style.use(['science'])

DATA_DIR = Path.home() / 'OneDrive/02. QuIN_Research/31. Noise-RTN/01. 2021_Algorithm paper/simulated_rtn/2022_01_10_anomalous_metastable'

df = pd.read_feather(DATA_DIR / 'artn_metastable_example=6_signals.feather')
df

plt.close()
fig, ax = plt.subplots(figsize=(1.5, 1.25))

df = df[:35_000]
ax.plot(df['full_signal'], linewidth=0.5)

xticks = np.array([0, 15_000, 30_000])
ax.set_xticks(xticks)
ax.set_xticklabels(xticks // 1_000)
ax.set_xlim(df.index.min(), df.index.max())
ax.set_ylabel('Intensity (a.u.)', fontsize=LABEL_FONT_SIZE)
ax.yaxis.set_label_coords(-0.20, 0.40)
ax.set_xlabel(r'Signal step ($\times10^3$)', fontsize=LABEL_FONT_SIZE, labelpad=0)
# ax.xaxis.set_label_coords(0.45, -0.1.2)
ax.tick_params(axis='both', which='both', direction='in', top=True,
               right=True, labelsize=8, pad=2)
# fig.tight_layout()

plt.savefig('/tmp/overleaf_rtn_metastable_plot.pdf')

"""
Plot simple time series figures of PMOS real data

This was created back when we intended to test our algorithm on PMOS data,
but we have switched to CNT real data for now.

Developed by Marcel Robitaille on 2021/12/13 Copyright © 2021 QuIN Lab
"""

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fastkde import fastKDE


PMOS_REAL_DATA_DIR = Path.home() / 'OneDrive/02. QuIN_Research' / \
    '31. Noise-RTN/03. Algorithm Verification/QuIN-CMOS 28nm_2019_Jun' / \
    '14K_28nmPMOS_June2019/measurements'
f = PMOS_REAL_DATA_DIR / \
    '2019-06-07_z8430_5_Sample_VGm0.6_VDm0.01_SampInt5e-3_NumSampMAX_2.csv'
f = PMOS_REAL_DATA_DIR / \
    '2019-06-07_z8430_5_Sample_VGm0.6_VDm0.01_SampInt5e-3_NumSampMAX_3.csv'
f = PMOS_REAL_DATA_DIR / \
    '2019-06-07_z8430_5_Sample_VGm0.6_VDm0.02_SampInt5e-3_NumSampMAX.csv'
f = PMOS_REAL_DATA_DIR / \
    '2019-06-07_z8430_5_Sample_VGm0.6_VDm0.02_SampInt5e-3_NumSampMAX_3.csv'
df = pd.read_csv(f)
df = df.rename(columns={'Time (s)': 'time', 'IG': 'full_signal'})
print(df)
df['filtered_signal'] = df['full_signal'].rolling(40).mean()
fig, (ax0, ax1) = plt.subplots(
    ncols=2,
    sharey=True,
    gridspec_kw={'width_ratios': [2, 1]},
)
ax0.plot(df['time'], df['full_signal'])
ax0.plot(df.dropna()['time'], df['filtered_signal'].dropna())
ax1.tick_params(axis='both', which='both', direction='in',
                labelleft=False, top=True, right=True)

# min_intensity = min(df['full_signal'])
# max_intensity = max(df['full_signal'])
# delta_intensity = max_intensity - min_intensity
# intensity = np.linspace(
#     min_intensity - delta_intensity,
#     max_intensity + delta_intensity,
#     2000,
# )

# kernel = stats.gaussian_kde(
#     df['filtered_signal'].dropna(),
#     bw_method=0.040,
# )
# density = kernel(intensity)
density, intensity = fastKDE.pdf(
    df['filtered_signal'].dropna().to_numpy(),
    # Power of 2 plus 1
    numPoints=2**10 + 1,
)
ax1.plot(density, intensity)
ylim = np.where(np.asarray(density) > 1e-4)[0]
ylim = [intensity[ylim[0]], intensity[ylim[-1]]]
ylim = [
    ylim[0] - 0.1 * (ylim[1] - ylim[0]),
    ylim[1] + 0.1 * (ylim[1] - ylim[0]),
]
ax1.set_ylim(ylim)
ax0.set_ylabel('Current (?)')
ax0.set_xlabel('Time (s)')
ax1.set_xlabel('Density')
fig.tight_layout()
fig.subplots_adjust(wspace=0)
fig.savefig(str(f).replace('.csv', '-kde.png'), dpi=300)
# plt.show()
plt.close('all')

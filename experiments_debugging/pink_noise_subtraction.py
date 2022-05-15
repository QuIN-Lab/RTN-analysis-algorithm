"""
Attempt to subtract 1/f noise from a noisy RTN signal.
First, fit the noisy RTN signal with the best 1/f fit.
Subtract this in the frequency domain, and take the inverse transform to recover
the time domain.

Developed by Marcel Robitaille on 2022/04/12 Copyright © 2021 QuIN Lab
"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize

from data_exploration.psd import apply_welch


plt.switch_backend('Agg')
mpl.rcParams['agg.path.chunksize'] = 10000


fig, ax = plt.subplots()

fig.subplots_adjust(right=0.97, top=0.92)

ax.set_yscale('log')
ax.set_xscale('log')
ax.grid(True, which='both')
ax.set_xlabel('Frequency~(Hz)')
ax.set_ylabel(r'Power~(nA\textsuperscript{2} / Hz)')
# ax.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.1))

# for column in ('I_D', 'digitized'):

def func(x, c):
    return c/x


df = pd.read_feather('/home/marcel/OneDrive/02. QuIN_Research/31. Noise-RTN/01. 2021_Algorithm paper/simulated_rtn/2022_04_08_pink_noise/1-trap_wn=0.1_example=6_signals.feather')

def noise_column():
    if 'white_noise' in df.columns:
        return 'white_noise'
    if 'pink_noise' in df.columns:
        return 'pink_noise'
    raise NotImplementedError()


def _one_over_f(f, scale):
    """
    Fitting curve
    """

    return scale / f


def find_fitting_parameters(frequencies, powers):
    powers = powers[frequencies != 1]
    frequencies = frequencies[frequencies != 1]
    print(frequencies)

    sigma = (1 - frequencies) ** 2
    params, *_ = curve_fit(_one_over_f, frequencies, powers, sigma=sigma)
    return params


def _plot_best_fit(ax, frequencies, powers):
    params = find_fitting_parameters(frequencies, powers)
    scale, = params
    alpha = 1.0
    best_curve = _one_over_f(frequencies, *params)
    print(best_curve)
    ax.plot(frequencies, best_curve, '-.', label=f'${scale:0.2} / f$')
            # label=f'${scale:0.2} / f^{{{alpha:0.2}}}$')
    return params


frequencies, powers = apply_welch(df, column='full_signal')
ax.plot(frequencies, powers, label='full_signal')

params = _plot_best_fit(ax, frequencies, powers)
ax.plot(frequencies, powers - _one_over_f(frequencies, *params),
        label='full_signal - fitting')

# print(np.fft.fft(df['full_signal']))
# print()
# print()
# print()
# print(np.fft.fft(df['full_signal']))
# print(len(np.fft.fft(df['full_signal'])))
ax.plot(abs(np.fft.fft(df['full_signal'])))
# ax.plot(np.fft.fft(df['full_signal']) - _one_over_f(range(1_000_000), *params))
# print()
# print()
# print()
# ax.plot(_one_over_f(range(1_000_000), *params))
# ifft = np.fft.ifft(powers, n=1_000_000)

frequencies, powers = apply_welch(df, column=noise_column())
ax.plot(frequencies, powers, label=noise_column(), linestyle='--')

frequencies, powers = apply_welch(df, column='rtn_sum')
ax.plot(frequencies, powers, label='rtn_sum')

ax.legend()

fig.savefig('/tmp/pink_noise_subtraction.png', dpi=300)
plt.close(fig)

# %%

plt.close('all')
fig, ax = plt.subplots()

# ax.plot(ifft)
ax.plot(np.fft.ifft(
    np.fft.fft(df['full_signal']) - 
    _one_over_f(range(1_000_000), *params)
))
# ax.plot(df['full_signal'], alpha=0.5)
# ax.set_xlim(0, 10_000)

fig.savefig('/tmp/pink_noise_subtraction_ifft.png', dpi=300)
plt.close(fig)

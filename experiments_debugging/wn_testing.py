"""
I used this code to try many different things to try and improve the white noise
estimation. It generates a heatmap of error to pinpoint outliers.

Developed by Marcel Robitaille on 2021/11/03 Copyright © 2021 QuIN Lab
"""

# pylint: disable=line-too-long,redefined-outer-name
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

DATA_DIR = Path.home() / 'OneDrive/02. QuIN_Research/31. Noise-RTN/01. 2021_Algorithm paper/simulated_rtn/generated_rtn_data4_white_noise_study/data4_All traps_white_noise_study/'

def improved(signal, digitized=None):
    SAMPLE_WIDTH = 30
    NUM_SAMPLES = 100
    samples = [
        signal.loc[i:i+SAMPLE_WIDTH]
        for i in np.linspace(0, len(signal) - SAMPLE_WIDTH - 1, NUM_SAMPLES)
    ]
    if digitized is not None and False:  # pylint: disable=conditional-evals-to-constant
        for i, sample in enumerate(samples):
            fig, ax = plt.subplots()
            ax.set_title(f'{i} {sample.std()}')
            ax.plot(digitized)
            ax.scatter(sample.index, sample)
            ax.set_xlim(sample.index.min(), sample.index.max())
            plt.show()
        plt.close(fig)
        print(samples)
    # ax.sc(signal)

    samples = [
        signal.loc[i:i+SAMPLE_WIDTH].std()
        for i in np.linspace(0, len(signal) - SAMPLE_WIDTH - 1, NUM_SAMPLES)
    ]
    samples = [
        sample for sample in samples if sample <= 3 * min(samples)
    ]
    # mean = np.mean(samples)
    # std = np.std(samples)
    # z_scores = [(x - mean) / std for x in samples]
    # filtered = [x for x, z_score in zip(samples, z_scores) if z_score < 2]
    # return np.mean(sorted(samples)[:len(samples)*3//2])
    return np.median(samples)


def original(signal):
    SAMPLE_WIDTH = 10
    NUM_SAMPLES = 10
    samples = [
        signal.loc[i:i+SAMPLE_WIDTH].std()
        for i in np.linspace(0, len(signal) - SAMPLE_WIDTH - 1, NUM_SAMPLES)
    ]
    return np.median(samples)


n_traps = 3
true_noise_level = 0.1
example=8
df = pd.read_feather(DATA_DIR / f'{n_traps}-trap_wn={true_noise_level:.1f}_example={example}_signals.feather')
est = improved(df['full_signal'], df['rtn_sum'])
print('guess', est)
real = df['white_noise'].std()
print('true', real)
print('err', abs(real - est) / real)
# %%

noise_levels = np.arange(0, 1.1, 0.1)
def generator(n_traps, f):
    for n_traps in np.arange(3) + 1:
        for true_noise_level in noise_levels:
            for example in range(10):
                df = pd.read_feather(DATA_DIR / f'{n_traps}-trap_wn={true_noise_level:.1f}_example={example}_signals.feather')
                est = f(df['full_signal'])
                real = df['white_noise'].std()
                err = 0 if real == est else (-1 if real == 0 else abs(real - est) / real)
                print(' '.join(f'{x:03.2f}' for x in (
                    est,
                    real,
                    err,
                )))
                yield n_traps, true_noise_level, example, err


# for n_traps in np.arange(3) + 1:
# for n_traps in [3]:
original_df = pd.DataFrame(
    generator(n_traps, original),
    columns=['n_traps', 'noise', 'example', 'err'],
)
# %%
improved_df = pd.DataFrame(
    generator(n_traps, improved),
    columns=['n_traps', 'noise', 'example', 'err'],
)
# %%

original_df['algorithm'] = 'original'
improved_df['algorithm'] = 'improved'
df = pd.concat([original_df, improved_df])
print(df.to_string())

fig, axes = plt.subplots(nrows=3, sharex=True, figsize=(6, 6))
df['noise'] = (df['noise'] * 100).apply(int)
df['err'] = df['err'] * 100
for n_traps, ax in zip(np.arange(3) + 1, axes):
    sns.boxplot(
        ax=ax,
        data=df[df['n_traps'] == n_traps],
        y='err',
        x='noise',
        hue='algorithm',
        showfliers=False,
    )
    ax.legend([], [], frameon=False)
    if n_traps == 3:
        ax.set_xlabel('Noise level (%)')
    if n_traps == 1:
        ax.legend(loc='upper right')
    ax.set_ylabel('Error (%)')
    ax.annotate(
        xy=(0.02, 0.9),
        text=f'$N_\\mathrm{{traps}} = {n_traps}$',
        xycoords='axes fraction',
    )
fig.tight_layout()
fig.subplots_adjust(hspace=0)
fig.savefig('/tmp/boxplot.png', dpi=300)
plt.close(fig)


# %%
# fig, ax = plt.subplots()
# ax.set_title(name, fontsize=16)
# ax.tick_params(axis='both', which='major', labelsize=16)

# im = ax.imshow(
#     accs,
#     cmap='RdYlGn_r',
#     vmax=accs.max().max(),
#     # vmin=heatmap.min().min(),
#     vmin=0,
# )

# noises = [f'{x:.1f}' for x in noise_levels]
# xs = range(10)

# ax.set_xticks(range(len(xs)))
# ax.set_yticks(range(len(noises)))
# ax.set_xticklabels(xs, rotation=45, ha='right')
# ax.set_yticklabels(noises, rotation=45)
# ax.set_ylabel('Noise', fontsize=16)
# # ax.set_xlabel(xlabel, fontsize=16)
# # width = heatmap.shape[1] / 10
# # for x in np.arange(0, 9 * width, width) + width - 0.5:
# #     ax.axvline(x, ymin=0, ymax=1, color='white')

# fig.tight_layout()
# extent = im.get_extent()

# for i, j in np.ndindex(accs.shape):
#     value = accs[i][j]
#     red, green, blue, _ = \
#         im.cmap(im.norm(value))

#     text = '{:.1f} %'.format(value * 100)
#     ax.text(
#         j, i,
#         text,
#         color='black' if (red*76.245 + green*149.7 + blue*29.07) > 186
#         else 'white',
#         ha='center',
#         va='center',
#         # fontsize=200 / (len(text) * np.sqrt(max(extent[1:3]))),
#         fontsize=7,
#         rotation=45,
#     )
# fig.savefig(f'/tmp/wn_estimate_heatmaps_ntraps={n_traps}.png', dpi=300)
# plt.close(fig)
# print(accs)

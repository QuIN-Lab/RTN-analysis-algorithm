import re

import matplotlib.pyplot as plt
import numpy as np

from example import Example
from constants import DATA_DIR


plt.style.use('science')

# %%


n_traps = 3
noise = 0.9
example = 9
example = Example(
        DATA_DIR / f'{n_traps}-trap_wn={noise:.1f}_example={example}').signals


def calculate_colors():
    def generator():
        with open('/home/marcel/Nextcloud/Waterloo/QuIN_Lab/31_CMOS_RTN/Articles/2022_03_RNN_Paper_Nature/flowchart-colors.tex', 'r', encoding='utf-8') as f:
            for line in f:
                m = re.match(
                    r'\\definecolor{([a-z]+)}{HTML}{([a-f0-9]{6})}', line)
                assert m is not None
                name, color = m.groups()
                yield name, f'#{color}'
    return dict(generator())


# %%
if __name__ == '__main__':
    # %%
    colors = calculate_colors()

    print(colors)

    # %%
    df = example.read().iloc[:20_000]
    df

    plt.close()
    fig, ax = plt.subplots()
    ax.plot(df.index, df['full_signal'], c=colors['raw'])
    ax.set_xlim(0, df.index.max())
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    ax.axis('off')
    fig.tight_layout()
    fig.savefig('/tmp/data_only_signals.pdf', dpi=300)
    plt.close(fig)

    # %%
    plt.close()
    fig, ax = plt.subplots()
    ax.plot(df.index, df['rtn_sum'], c=colors['digitization'])
    ax.set_xlim(0, df.index.max())
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    ax.axis('off')
    fig.tight_layout()
    fig.savefig('/tmp/data_only_digitization.pdf', dpi=300)
    fig.savefig('/tmp/data_only_digitization.png', dpi=300)
    plt.close(fig)


    # %%

    kde_data = example.kde_data.read().squeeze()

    plt.close()
    fig, ax = plt.subplots(figsize=(3.5, 2))
    ax.plot(kde_data.intensity, kde_data.density, c=colors['kde'], linewidth=2)
    ylim = np.where(np.asarray(kde_data.density) > 1e-5)[0]
    ylim = [
        kde_data.raw_intensity[ylim[0]],
        kde_data.raw_intensity[ylim[-1]],
    ]
    ylim = [
        ylim[0] - 0.1 * (ylim[1] - ylim[0]),
        ylim[1] + 0.1 * (ylim[1] - ylim[0]),
    ]
    ax.set_xlim(ylim)
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    ax.axis('off')
    fig.tight_layout()
    fig.savefig('/tmp/data_only_kde.pdf', dpi=300)
    plt.close(fig)

    # %%

    kde_data = example.kde_data.read().squeeze()
    gmm_fit = example.gmm_fit.read()

    plt.close()
    fig, ax = plt.subplots(figsize=(3.5, 2))
    ax.plot(kde_data.intensity, kde_data.density, '--', c='gray', label='KDE',
            linewidth=2)
    ax.plot(kde_data.intensity, gmm_fit.best_fit, label='GMM', c=colors['gmm'],
            linewidth=2, alpha=0.8)
    ax.legend(fontsize=20)
    ylim = np.where(np.asarray(kde_data.density) > 1e-5)[0]
    ylim = [
        kde_data.raw_intensity[ylim[0]],
        kde_data.raw_intensity[ylim[-1]],
    ]
    ylim = [
        ylim[0] - 0.1 * (ylim[1] - ylim[0]),
        ylim[1] + 0.1 * (ylim[1] - ylim[0]),
    ]
    ax.set_xlim(ylim)
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    ax.axis('off')
    fig.tight_layout()
    fig.savefig('/tmp/data_only_gmm.pdf', dpi=300)
    plt.close(fig)

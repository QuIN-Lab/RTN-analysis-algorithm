"""
Make simple PSD plots

Developed by Marcel Robitaille on 2022/04/06 Copyright © 2021 QuIN Lab
"""

from multiprocessing import Pool
from datetime import datetime
from functools import partial

import click
import numpy as np
import doctool
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from scipy import signal

from example import Example
from constants import DATA_DIR


plt.style.use('science')
plt.switch_backend('Agg')


def apply_welch(df, column='digitized'):
    # sampling_frequency = 1 / dt

    frequencies, powers = signal.welch(
        df[column],
        2,
        nperseg=128,
    )

    # Fold two-sided spectrum
    indices = np.argsort(abs(frequencies))
    powers = powers[indices]
    frequencies = abs(frequencies[indices])

    # There is always a big spike at 0Hz
    powers = powers[frequencies != 0]
    frequencies = frequencies[frequencies != 0]

    return frequencies, powers



def psd(example: Example, output_format='pdf'):
    example = Example(example)
    print(example)
    assert str(example.path).endswith('_signals.feather'), example.path
    df = example.read()

    fig, ax = plt.subplots()

    fig.subplots_adjust(right=0.97, top=0.92)

    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.grid(True, which='both')
    ax.set_xlabel('Frequency~(Hz)')
    ax.set_ylabel(r'Power~(nA\textsuperscript{2} / Hz)')
    ax.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.1))

    # for column in ('I_D', 'digitized'):

    def noise_column():
        if 'white_noise' in df.columns:
            return 'white_noise'
        if 'pink_noise' in df.columns:
            return 'pink_noise'
        raise NotImplementedError()

    for column in ('full_signal', noise_column()):
        frequencies, powers = apply_welch(df, column=column)

        # ax.plot(frequencies, powers,
        #         label=_label_for_column(column, should_apply_rolling_mean))
        ax.plot(frequencies, powers, label=column)

    ax.legend()
    # ax.set_title(example.with_name('').path.name.replace('_', ' ').strip(),
                 # fontsize=9)

    # color = 'green'
    # fig.text(
    #     0.1, 0.96,
    #     f'Marcel Robitaille\n{datetime.now().strftime("%Y/%m/%d")}',
    #     va='top', ha='center',
    #     color=color,
    #     fontsize=6,
    #     bbox=dict(edgecolor=color, facecolor='white', pad=2),
    # )

    # return fig, ax
    figure_filename = example.with_name(f'psd.{output_format}').path
    fig.savefig(figure_filename, dpi=300)
    plt.close(fig)
    return figure_filename


@click.command('plot-psd')
@click.argument('files', type=click.Path(), required=True, nargs=-1)
@click.option('--output-format', type=click.Choice(['pdf', 'png']),
              default='pdf')
@doctool.example(
    help='Plot the power spectral density of an example using Welch\'s method',
    args=[
        DATA_DIR / '1-trap_wn=0.4_example=0_signals.feather',
        '--output-format=png',
    ],
    creates_image=True,
)
def plot_psd_click(files, output_format):
    with Pool() as p:
        return list(p.imap_unordered(
            partial(psd, output_format=output_format),
            files,
        ))[0]


__all__ = ('plot_psd_click',)


if __name__ == '__main__':
    plot_time_lag(Example('/home/marcel/OneDrive/02. QuIN_Research/31. Noise-RTN/01. 2021_Algorithm paper/simulated_rtn/generated_rtn_data4_white_noise_study/data4_All traps_white_noise_study/3-trap_wn=1.0_example=0_signals.feather'))  # noqa

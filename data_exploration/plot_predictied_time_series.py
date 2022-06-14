"""
Make simple plots of the discretized predicted signal output from the final
step.

Developed by Marcel Robitaille on 2022/03/18 Copyright © 2021 QuIN Lab
"""

import re
from multiprocessing import Pool
from pathlib import Path
from functools import partial

import click
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from example import Example
from constants import STEPS
from figure_constants import LABEL_FONT_SIZE

# plt.switch_backend('Qt5Agg')
plt.style.use('science')


figsize = (2, 1.80)
ylim = (-0.1, 1.25)


def plot_time_series_predictions(example, output_format='pdf'):

    print(example)
    signals = example.read()
    # decomp_data = pd.read_csv(file('decomp_data_traps.csv'))
    kde_data = example.kde_data.read().squeeze()
    mean_vec = sorted(kde_data.peaks_intensities.tolist())
    amp_guesses = np.array(mean_vec[1:]) - mean_vec[0]
    df = pd.read_feather(
        Path('../Lu/2022_01_27_artn_generate_with_ratio/raw_data_from_server') /
        example.time_series_predictions.path.name,
    )
    # df = example.time_series_predictions.read()

    is_real_measurement = 'time' in signals.columns

    # Make sure we have the columns we expect before taking the sum
    columns = [f'trap_{i}' for i, _ in enumerate(amp_guesses)]
    for c in columns:
        assert c in df.columns, \
            f'Expected `{c}` in dataframe. Found only {df.columns}.'
    df[columns] = df[columns] \
        .apply(lambda x: x * amp_guesses[int(x.name[-1])])
    df['sum'] = df[columns].sum(axis=1)
    # TODO: Below is required for CNT data
    # df['sum'] = df['sum'].rolling(2).min()
    # df['sum'] += min(mean_vec)

    plt.close()
    fig, ax = plt.subplots(figsize=(5, 1.75))

    if 'time' not in signals.columns:
        signals['time'] = signals.index

    ax.set_title(example.with_name('').path.name.replace('_', ' ').strip())
    signals = signals[:50_000]
    # Not sure why -2 required
    signals['pred'] = df['sum'].dropna().shift(STEPS - 2)
    signals = signals.dropna()
    signals['time'] -= min(signals['time'])

    # x = np.arange(len(signals)) - 63
    ax.plot(signals.time, signals['full_signal'], linewidth=0.5)
    ax.plot(signals.time, signals['pred'], c='red', linewidth=0.5)

    xticks = np.arange(0, signals.time.max() // 1000, 10)
    ax.set_xticks(xticks * 1000)
    ax.set_xticklabels(xticks)
    ax.set_xlim(0, signals.time.max())
    ax.set_ylabel(
        r'$I$ ($\mu$A)' if is_real_measurement else r'Intensity (arb. unit)',
        fontsize=LABEL_FONT_SIZE, labelpad=0.5)
    ax.set_xlabel(
        r'Time $(s)$' if is_real_measurement else r'Signal Step',
        fontsize=LABEL_FONT_SIZE, labelpad=1)
    ax.tick_params(axis='both', which='both', direction='in',
                   top=True, right=True, labelsize=10, pad=2)

    figure_filename = \
            example.with_name(f'predicted_time_series.{outputt_format}').path
    fig.savefig(figure_filename, dpi=300)
    plt.close(fig)
    return figure_filename


@click.command('plot-time-series-predictions')
@click.argument('files', type=click.Path(), required=True, nargs=-1)
@click.option('--output-format', type=click.Choice(['pdf', 'png']),
              default='pdf')
def plot_time_series_predictions_click(files, output_format):
    """
    Make time-series-predictions plots for the examples specified by FILES
    (specify `_signals.feather` files).
    """
    with Pool() as p:
        return list(p.imap_unordered(
            partial(plot_time_series_predictions, output_format=output_format),
            files,
        ))[0]


__all__ = ('plot_time_series_predictions_click',)


if __name__ == '__main__':
    plt.switch_backend('TkAgg')
    from constants import ARTN_DATA_DIR
    plot_time_series_predictions(Example(
        ARTN_DATA_DIR / 'artn_coupled_example=1_signals.feather'
    ))

    # pdfunite ~artn/*_predicted_time_series.pdf
    # 2022_03_17_artn_predicted_time_series_results.pdf

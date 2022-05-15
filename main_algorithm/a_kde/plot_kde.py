"""
Plot the KDE plot like Fig. 2 a) in our algorithm paper EDL submission.

Developed by Marcel Robitaille on 2022/02/18 Copyright © 2021 QuIN Lab
"""

from multiprocessing import Pool
from functools import partial

import click
from matplotlib import ticker
import matplotlib.pyplot as plt
import numpy as np
import doctool

from example import Example
from figure_constants import FILTERED, LABEL_FONT_SIZE, RAW, TICK_PADDING
from mode import Mode
from constants import DATA_DIR


def plot_kde(example, mode, debug, output_format='pdf'):
    plt.switch_backend('TkAgg')
    example = Example(example)
    df = example.read()
    # df = df[df.time < 494]
    # example.with_name('_trimmed=494_signals.feather').write(df)
    # plt.plot(df.time, df.full_signal)
    # plt.show()
    # return
    mode = Mode.from_str(mode=mode, filename=example.path)

    print('plot_kde', example.path.name)
    df = example.read()

    kde_data = example.kde_data.read().squeeze()

    fig, (ax0, ax1) = plt.subplots(
        figsize=(5, 2.05),
        ncols=2,
        dpi=300,
        gridspec_kw={'width_ratios': [2, 1]},
        sharey=True,
    )

    df_fig = df.copy().loc[:30e3]
    df_fig['horizontal_axis'] = df_fig.time if mode == Mode.CNT_REAL_DATA \
        else df_fig.index

    ax0.plot(df_fig.horizontal_axis, df_fig['full_signal'], lw=1, c=RAW)
    ax0.plot(df_fig.horizontal_axis, df_fig['signal_filtered_kde'],
             lw=1, c=FILTERED, linestyle='solid')
    # ax0.plot(df_fig['rtn_sum'], lw=0.3, c='orange')
    ax0.set_xticks([0, 10000, 20000])
    ax0.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _y: x // 1000))
    ax0.set_xlim(0, df_fig.horizontal_axis.max())
    ax0.set_ylabel('Intensity (arb. unit)',
                   fontsize=LABEL_FONT_SIZE, labelpad=1)
    ax0.set_xlabel(
        r'Signal step $(\times 10^3)$',
        fontsize=LABEL_FONT_SIZE,
        labelpad=0,
    )
    # ax0.xaxis.set_minor_locator(AutoMinorLocator(10))
    # ax0.yaxis.set_minor_locator(AutoMinorLocator(10))
    # ax0.tick_params(axis='both', which='both', direction='in', top=True,
    #                 right=True)

    ax1.plot(
        kde_data.raw_density, kde_data.raw_intensity,
        label='Before filtering', c=RAW)
    ax1.plot(
        kde_data.density, kde_data.intensity,
        label='After filtering', c=FILTERED, linestyle='solid')
    ax1.scatter(
        kde_data.peaks_densities,
        kde_data.peaks_intensities,
        marker='s',
        label=f'Peaks: {len(kde_data.peaks_intensities)}' if debug else 'Peaks',
        c='#cc0000',
        s=3,
        zorder=10,
    )

    # for y in set(df['rtn_sum']):
    #     ax1.axhline(y=y, linestyle='--', c='k', lw=0.5)

    ax1.set_xlim(0, max(kde_data.density) * 1.1)
    ylim = np.where(np.asarray(kde_data.raw_density) > 1e-5)[0]
    ylim = [
        kde_data.raw_intensity[ylim[0]],
        kde_data.raw_intensity[ylim[-1]],
    ]
    ylim = [
        ylim[0] - 0.1 * (ylim[1] - ylim[0]),
        ylim[1] + 0.1 * (ylim[1] - ylim[0]),
    ]
    ax1.set_ylim(ylim)
    ax1.set_xlabel(
        r'Density (arb. unit\textsuperscript{-1})',
        fontsize=LABEL_FONT_SIZE,
        labelpad=0,
    )
    ax1.legend(
        loc='upper right',
        prop={'size': 8},
        edgecolor='white',
        labelspacing=0.2,
        borderaxespad=0.2,
    )
    # ax1.xaxis.set_minor_locator(AutoMinorLocator(10))
    # ax1.tick_params(axis='both', which='both', direction='in',
    #                 labelleft=False, top=True, right=True)
    if debug:
        fig.suptitle(f'Window: {kde_data.window}', y=1, va='bottom')

    mf = ticker.ScalarFormatter(useMathText=True)
    mf.set_powerlimits((-2, 2))
    ax0.yaxis.set_major_formatter(mf)
    ax0.tick_params(axis='both', pad=TICK_PADDING)
    ax1.tick_params(axis='both', pad=TICK_PADDING)
    fig.tight_layout()
    fig.subplots_adjust(wspace=0, top=0.99)
    figure_filename = example.with_name(f'kde.{output_format}').path
    fig.savefig(figure_filename, dpi=300)
    plt.show()
    plt.close(fig)
    return figure_filename


@click.command('plot-kde')
@click.argument('files', type=click.Path(), nargs=-1, required=True)
@click.option('--mode', type=click.Choice(Mode.strings), required=True,
              default='auto',
              help='The algorithm mode. If `auto`, determine from filename.')
@click.option('--debug/--no-debug', is_flag=True, default=False,
              help='Add debug information to the figure, like N_peaks')
@click.option('--output-format', type=click.Choice(['pdf', 'png']),
              default='pdf')
@doctool.example(
    help='To generate the KDE plot, run the command:',
    args=[
        DATA_DIR / '1-trap_wn=0.4_example=0_signals.feather',
        '--output-format=png',
    ],
    creates_image=True,
)
def plot_kde_click(files, mode, debug, output_format):
    """
    Make KDE plots for the examples specified by FILES (specify
    `_signals.feather` files).
    """

    for f in files:
        plot_kde(f, mode=mode, output_format=output_format, debug=debug)
    # with Pool() as p:
        # return list(p.imap_unordered(
        #     partial(plot_kde, mode=mode, output_format=output_format,
        #         debug=debug),
        #     files,
        # ))[0]

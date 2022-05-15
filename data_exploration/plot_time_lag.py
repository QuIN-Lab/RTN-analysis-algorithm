"""
Make simple time lag plots

Developed by Marcel Robitaille on 2022/03/17 Copyright © 2021 QuIN Lab
"""

from multiprocessing import Pool
from datetime import datetime
from functools import partial

import click
import numpy as np
import pandas as pd
import doctool
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm

from example import Example
from constants import DATA_DIR


plt.style.use('science')
plt.switch_backend('Agg')


def plot_time_lag(example: Example, bins=200, output_format='pdf', author=None,
        ax=None, fig=None):
    example = Example[pd.DataFrame](example)
    print(example)
    assert str(example.path).endswith('_signals.feather'), example.path
    df = example.read()

    fig, ax = plt.subplots() if ax is None else (None, ax)

    dt = 1
    matrix, *_ = ax.hist2d(
        df.full_signal[:-dt],
        y=df.full_signal[dt:],
        bins=bins,
        norm=LogNorm(),
    )

    ax.set_aspect('equal')

    ax.set_xlabel(r'$I_i$')
    ax.set_ylabel(rf'$I_{{i+{dt}}}$')
    # ax.set_title(example.with_name('').path.name.replace('_', ' ').strip(),
    #              fontsize=9)

    if author:
        color = 'green'
        assert fig, 'Fig must be defined for `author` function'
        fig.text(
            0.7, 0.96,
            f'{author}\n{datetime.now().strftime("%Y/%m/%d")}',
            va='top', ha='center',
            color=color,
            fontsize=6,
            bbox=dict(edgecolor=color, facecolor='white', pad=2),
        )

    if fig is not None:
        fig.tight_layout()

    ticks = ax.get_yticks()
    if (ticks == ticks.astype(int)).all():
        ticks = ticks.astype(int)
    else:
        ax.xaxis.set_major_formatter('{x:.2f}')
        ax.yaxis.set_major_formatter('{x:.2f}')
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticks)
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticks)

    np.save(file=example.with_name(f'time_lag_data.npy').path, arr=matrix)

    if fig is not None:
        figure_filename = example.with_name(f'time_lag.{output_format}').path
        fig.savefig(figure_filename, dpi=600)
        plt.close(fig)
        return figure_filename


def add_tlp(fig, ax, example: Example):
    """
    Take a figure and add a 1:1 TLP in the right plane.
    Create a new axis with the remaining space and return it.
    """

    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width, height = bbox.width, bbox.height
    tlp_width = height / width
    tlp_ax = ax.inset_axes([1 - tlp_width, 0, tlp_width, 1])
    plot_time_lag(example=example, ax=tlp_ax)

    # ax.set_xlim(np.array(ax.get_xlim()) / (1 - tlp_width))

    tlp_ax.yaxis.set_label_position('right')
    tlp_ax.tick_params(
        axis='both', which='both', direction='in',
        labelleft=False, top=True, right=True, labelright=True,
    )

    # Move the label independently of the tick labels
    # Setting labelpad has different results depending on the width of the
    # tick labels
    tlp_ax.yaxis.set_label_coords(1.03, 0.5)
    tlp_ax.xaxis.labelpad = -9

    other_ax = ax.inset_axes([0, 0, 1 - tlp_width, 1])

    ax.set_xticks([])
    ax.set_yticks([])

    return tlp_ax, other_ax


@click.command('plot-time-lag')
@click.argument('files', type=click.Path(), required=True, nargs=-1)
@click.option('--bins', type=int, default=200)
@click.option('--output-format', type=click.Choice(['pdf', 'png']),
              default='pdf')
@click.option('--author', type=str, default=None,
              help='Add your name and the date to the figure.')
@doctool.example(
    help='Plot the time-lag plot for an example',
    args=[
        DATA_DIR / '1-trap_wn=0.4_example=0_signals.feather',
        '--output-format=png',
    ],
    creates_image=True,
)
def plot_time_lag_click(files, bins, output_format, author):
    """
    Make time-lag plots for the examples specified by FILES (specify
    `_signals.feather` files).
    """
    with Pool() as p:
        return list(p.imap_unordered(
            partial(plot_time_lag, bins=bins, output_format=output_format,
                    author=author),
            files,
        ))[0]


# if __name__ == '__main__':
#     plot_time_lag(Example(DATA_DIR / '1-trap_wn=0.2_example=7_signals.feather'))


__all__ = ('plot_time_lag_click',)

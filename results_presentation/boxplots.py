"""
Create boxplots of the algorithm results.

Developed by Marcel Robitaille on 2021/11/05 Copyright © 2021 QuIN Lab
"""

# pylint: disable=redefined-outer-name
import re
from typing import List, Optional, Set, Union, cast
from pathlib import Path

import click
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
from matplotlib import ticker
from matplotlib import pyplot as plt

from example import Example
from constants import Publisher, PUBLISHER
from figure_constants import LABEL_FONT_SIZE
from utils import subfig, pairwise

plt.style.use('science')
# matplotlib.rcParams['legend.frameon'] = 'True'


def isiterable(x):
    try:
        iter(x)
        return True
    except TypeError:
        return False


def even(x):
    return x % 2 == 0



def stripplot(
    ax, y, x, hue, data, palette, edgecolor='gray',
    hue_order: Optional[List[str]]=None,
    size: Optional[float]=None,
    dodge: bool=True,
    width: float=0.8,
    jitter: Union[bool, float]=True,
    color_inside_boxplot: Optional[str]=None,
    **kwargs,
):
    """
    Like seaborn stripplot, but supports the boxplot `width` argument.
    The API should be identical besides this added argument.
    Any differences should be considered a bug and fixed.

    See: https://seaborn.pydata.org/generated/seaborn.stripplot.html
    """

    df = pd.DataFrame()
    # print(df)
    df['y'] = data[y]
    categories = list(sorted(set(data[x]), key=list(data[x]).index))
    # print('categories', categories)
    hues: Set[str] = set(data[hue])
    hue_order = list(hues) if hue_order is None else hue_order
    assert len(hues) == len(hue_order)
    df['hue'] = data[hue].apply(hue_order.index)
    df['category'] = data[x].apply(categories.index)
    df['c'] = df['hue'].apply(lambda x: palette[x]) \
        if isiterable(palette) else palette

    if color_inside_boxplot is not None:
        boxplot_data = df.groupby(['hue', 'category'])['y'] \
            .quantile([0.25, 0.75]).unstack() \
            .rename(columns={0.25: 'Q1', 0.75: 'Q3'})
        boxplot_data = cast(pd.DataFrame, boxplot_data)
        boxplot_data['IQR'] = boxplot_data['Q3'] - boxplot_data['Q1']
        boxplot_data['min'] = boxplot_data['Q1'] - 1.5 * boxplot_data['IQR']
        boxplot_data['max'] = boxplot_data['Q3'] + 1.5 * boxplot_data['IQR']
        df[['boxplot_min', 'boxplot_max']] = df.apply(
            lambda row: boxplot_data.loc[(row.hue, row.category)][['min', 'max']],
            axis=1,
        )
        df['c'] = df.apply(
            lambda row: color_inside_boxplot
            if row.boxplot_min <= row.y <= row.boxplot_max else row.c,
            axis=1,
        )

    print(df)
    df['hue'] = (df['hue'] - (1.5 if even(len(hues)) else 1)) * width / len(hues)
    df['x'] = df['category'] + df['hue'] if dodge else df['category']

    jitter = 0.1 if isinstance(jitter, bool) else float(jitter)
    jitter /= len(hues)
    df['jitter'] = np.random.uniform(low=-jitter, high=+jitter, size=len(df))
    df['x'] += df['jitter']

    ax.scatter(df.x, df.y, c=df['c'], edgecolor=edgecolor, s=size,
               **kwargs)


def generate_boxplot(df, column, ylabel, name):
    df = df.copy()
    assert df['noise'].dtype == np.int64

    # For nature, make the figure vertical and show all 11 noise values
    if PUBLISHER == Publisher.IEEE:
        noise_set = (20, 60, 100)
        df = df[df['noise'].isin(noise_set)]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.set_yscale('log')
    df[column] *= 100
    sns.boxplot(
        ax=ax,
        y=column,
        x='noise',
        hue='n_traps',
        data=df,
        orient='v',
        palette='vlag',
        boxprops=dict(alpha=0.5),
        # showfliers=False,
    )
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        title=r'$N_\mathrm{traps}$',
        loc='upper left',
        fontsize='small',
        prop=dict(size=7),
    )
    ax.set_xlabel(
        r'$Q_\mathrm{wn} = '
        r'\sigma_\mathrm{wn}\ /\min(\Delta_\mathrm{RTN})\ (\%)$',
        fontsize=LABEL_FONT_SIZE,
    )
    ax.set_ylabel(ylabel, fontsize=LABEL_FONT_SIZE)
    fig.tight_layout()

    filename = f'{name}_{column}_boxplot.png'
    filename = re.sub(r'[\\\${}]', '', filename).replace(' ', '_') \
        .replace('mathrm', '')
    fig.savefig(boxplot_out_dir / filename, dpi=300)
    # plt.show()
    plt.close(fig)


def sort(df, column):
    """
    Separate the traps of each example and classify them according to their
    relative socre: a, b, c
    """

    def generator(column):
        for _, group in df.groupby(by=['n_traps', 'noise', 'example']):
            group.sort_values(column, inplace=True)
            for letter, (_, row) in zip(('a', 'b', 'c'), group.iterrows()):
                yield f'{int(row.n_traps)}.{letter}', row['noise'], \
                    row['example'], row[column]
    df = pd.DataFrame(
        generator(column),
        columns=['n_traps', 'noise', 'example', column],
    )
    return df


def generate_sorted_boxplot(df, column, *args, **kwargs):
    df = sort(df, column=column)
    generate_boxplot(df=df, column=column, *args, **kwargs)


def read_data(filename, n_traps):
    df = pd.read_csv(filename)
    df['n_traps'] = n_traps
    return df


# %% Tau

@click.command()
def plot_boxplot_tau():
    tau_df = pd.concat([
        read_data(results_dir / f'{n_traps}-trap_tau_error.csv', n_traps)
        for n_traps in np.arange(3) + 1
    ])

    tau_df['noise'] = (tau_df['noise'] * 100).apply(int)
    tau_df['n_traps'] = tau_df['n_traps'].apply(int)
    sorted_tau_df = sort(tau_df, column='tau_high').merge(
        sort(tau_df, column='tau_low'),
        on=['example', 'noise', 'n_traps'],
        how='outer',
    ).rename(columns={'n_traps': 'class'})
    sorted_tau_df['noise'] = sorted_tau_df['noise'].apply(int)
    sorted_tau_df['example'] = sorted_tau_df['example'].apply(int)
    sorted_tau_df

    # %%

    for column in ['tau_high', 'tau_low']:
        generate_boxplot(
            name={
                'tau_high': r'$\bar{\tau}_\mathrm{{high}}$',
                'tau_low': r'$\bar{\tau}_\mathrm{{low}}$',
            }[column],
            column=column,
            df=tau_df,
            ylabel={
                'tau_high': r'$\epsilon(\bar{\tau}_\mathrm{high})\ (\%)$',
                'tau_low': r'$\epsilon(\bar{\tau}_\mathrm{low})\ (\%)$',
            }[column],
        )

# %% Digitization

@click.command()
def plot_boxplot_digitization():
    digitization_df = pd.concat([
        read_data(
            results_dir / f'{n_traps}-trap_digitization_accuracy.csv',
            n_traps,
        )
        for n_traps in np.arange(3) + 1
        # for n_traps in [3]
    ])
    digitization_df['noise'] = (digitization_df['noise'] * 100).apply(int)
    digitization_df['digitization_error'] = \
        1 - digitization_df['digitization_accuracy']
    digitization_df['digitization_error'].mean()
    digitization_df['digitization_error'].std()
    # generate_boxplot(
    #     name='digitization error',
    #     column='digitization_error',
    #     df=digitization_df,
    #     ylabel='Digitization error',
    # )


# %% Amplitude

@click.command()
def plot_boxplot_amplitude():
    amp_df = pd.read_csv(results_dir / 'amplitude_accuracies.csv')
    amp_df['noise'] = (amp_df['noise'] * 100).apply(int)
    generate_boxplot(
        name='amplitude error',
        column='amplitude_error',
        df=amp_df,
        ylabel=r'$\epsilon(\Delta_\mathrm{RTN})\ (\%)$',
    )

# %% Subplots boxplot

def get_dataframe(df, should_sort, column):
    if not should_sort:
        return df

    def generator(column):
        for _, group in df.groupby(by=['n_traps', 'noise', 'example']):
            group.sort_values(column, inplace=True)
            for letter, (_, row) in zip(('a', 'b', 'c'), group.iterrows()):
                yield f'{int(row.n_traps)}.{letter}', row['noise'], \
                    row['example'], row[column]

    return pd.DataFrame(
        generator(column),
        columns=['n_traps', 'noise', 'example', column],
    )


class Patch:
    """
    Wrapper around an axis to get the noise axis and error axis
    independently of which is the x and y axis.
    This makes it easy to switch between horizontal and vertical.
    """

    def __init__(self, ax, noise_axis, error_axis):
        self.ax = ax
        self._noise_axis = noise_axis
        self._error_axis = error_axis

    def __getattr__(self, attr):
        if attr == 'noise_axis':
            return getattr(self.ax, f'{self._noise_axis}axis')
        if attr == 'error_axis':
            return getattr(self.ax, f'{self._error_axis}axis')
        try:
            return getattr(self.ax, attr)
        except AttributeError:
            attr = attr.replace('noise_axis_', self._noise_axis) \
                .replace('error_axis_', self._error_axis)
            return getattr(self.ax, attr)

# %%
@click.command()
@click.option('--sort/--no-short', 'should_sort',
              help='Whether to separate and sort the traps into classes '
              '(a, b, c).')
@click.argument('files', type=click.Path(), nargs=-1, required=True)
@click.argument('output-file', type=click.Path())
@click.option('--noise-type', type=click.Choice(['white', 'pink']),
              required=True)
def plot_boxplot_subplots(files, output_file, should_sort, noise_type):
    """
    Plot a boxplot with all metrics using subplots
    """
    # %%
    df = pd.concat(Example[pd.DataFrame](f).read() for f in files)
    df['noise'] = df['noise'].apply(lambda n: int(n) if n > 1 else int(n * 100))

    # for palette in ['vlag', 'hls', 'husl', 'Set1', 'Set2', 'Set3']:

    if PUBLISHER == Publisher.IEEE:
        noise_set = (20, 60, 100)
        df = df[df['noise'].isin(noise_set)]

    plt.close('all')

    # For nature, make the figure vertical and show all 11 noise values
    fig, axes = {
        Publisher.IEEE: plt.subplots(ncols=3, nrows=1, figsize=(5, 1.75), sharey=True),
        Publisher.NATURE: plt.subplots(ncols=1, nrows=2, figsize=(6, 3), sharex=True),
    }[PUBLISHER]

    for i, (ax, column) in \
            enumerate(zip(axes, ('amplitude_error', 'digitization_error'))):
        df_sub = get_dataframe(df, should_sort=should_sort, column=column)
        df_sub[column] *= 100
        print(df_sub)

        noise_axis, error_axis = {
            Publisher.IEEE: ('y', 'x'),
            Publisher.NATURE: ('x', 'y'),
        }[PUBLISHER]
        ax = Patch(ax, noise_axis=noise_axis, error_axis=error_axis)
        ax.set_error_axis_scale('log')
        x, y = {
            Publisher.IEEE: (column, 'noise'),
            Publisher.NATURE: ('noise', column),
        }[PUBLISHER]

        palette = ['white', '#8aafc8', '#d9efff', '#e98d83', '#e9a783', '#fdca93'] \
            if should_sort \
            else ['#8aafc8', '#fdca93', '#e98d83']
        width = 0.7
        sns.boxplot(
            ax=ax,
            x=x,
            y=y,
            hue='n_traps',
            data=df_sub,
            orient={
                Publisher.IEEE: 'h',
                Publisher.NATURE: 'v',
            }[PUBLISHER],
            palette=palette,
            # boxprops=dict(alpha=0.8),
            showfliers=False,
            linewidth=0.5,
            width=width,
        )
        stripplot(
            ax=ax,
            x=x,
            y=y,
            hue='n_traps',
            dodge=True,
            data=df_sub,
            # palette=['white'] * 3,
            palette=palette,
            color_inside_boxplot='white',
            size=4,
            linewidth=0.2,
            width=width,
        )
        handles, labels = ax.get_legend_handles_labels()
        # Hack to remove stripplot from legend
        # Relies on their being equally-many boxplot and stripplot entries
        # and that they appear in that order
        handles = handles[:3]
        labels = labels[:3]
        print(column)
        if i == len(axes) - 1:
            ax.set_noise_axis_label(
                {
                    'white': r'$Q_\mathrm{wn}\ (\%)$',
                    'pink': r'$Q_\mathrm{pn}\ (\%)$',
                }[noise_type],
                # r'$Q_\mathrm{wn} = \sigma_\mathrm{wn}\ '
                # r'/\min_\mathrm{traps}(\Delta_\mathrm{RTN})\ (\%)$',
                labelpad=0,
                fontsize=LABEL_FONT_SIZE,
            )
            # yticks = ax.get_yticklabels()
            # ax.set_yticklabels(yticks, rotation=90, va='center')
            ax.tick_params(axis=noise_axis, pad=2)
        else:
            ax.set_noise_axis_label('')
        if column == {
                Publisher.IEEE: 'tau_low',
                Publisher.NATURE: 'amplitude_error',
        }[PUBLISHER]:
            ax.legend(
                handles,
                labels,
                title='Trap' if should_sort else r'$N_\mathrm{traps}$',
                loc={
                    Publisher.IEEE: 'upper right',
                    Publisher.NATURE: 'lower right',
                }[PUBLISHER],
                prop=dict(size=8),
                title_fontsize=9,
                handlelength=0.7,
                labelspacing=0.3,
                handletextpad=0.3,
                borderaxespad=0.1,
            )
        else:
            ax.get_legend().remove()
        ax.set_error_axis_label(
            {
                'tau_high': r'$\epsilon(\bar{\tau}_\mathrm{high})\ (10^x\,\%)$',
                'tau_low': r'$\epsilon(\bar{\tau}_\mathrm{low})\ (10^x\,\%)$',
                'amplitude_error': r'$\epsilon(\Delta_\mathrm{RTN})\ (10^x\,\%)$',
                'digitization_error': r'$\epsilon(\eta)\ (10^x\,\%)$',
            }[column],
            labelpad=1,
            fontsize=LABEL_FONT_SIZE,
        )

        # Hide major ticks, but keep labels
        # Show minor ticks between labels
        # Requested by Prof. Kim
        ax.tick_params(axis='x', which='major', length=0)
        ax.tick_params(axis='x', which='minor', length=4)
        ax.xaxis.set_minor_locator(ticker.FixedLocator(
            [np.mean(pair) for pair in pairwise(ax.get_xticks())]
        ))

        ax.annotate(
            subfig(i + 1),
            xycoords='axes fraction',
            textcoords='offset points',
            ha='left',
            **{
                Publisher.IEEE: dict(xy=(0, 0), xytext=(4, 8)),
                Publisher.NATURE: dict(xy=(0, 1), xytext=(4, -12)),
                # Publisher.NATURE: dict(xy=(0, 1), xytext=(-32, 2 - 8 * i)),
            }[PUBLISHER],
        )

    fig.tight_layout()
    fig.subplots_adjust(**{{
        Publisher.IEEE: 'wspace',
        Publisher.NATURE: 'hspace',
    }[PUBLISHER]: 0})
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_file, dpi=300)
    plt.close(fig)


# %% Artn one subplot boxplot

@click.command()
@click.argument('files', type=click.Path(), nargs=-1, required=True)
@click.argument('output-file', type=click.Path())
def plot_boxplot_artn(files, output_file):
    df = pd.concat(Example[pd.DataFrame](f).read() for f in files)
    if 'digitization_accuracy' in df.columns and \
            'digitization_error' not in df.columns:
        df['digitization_error'] = 1 - df['digitization_accuracy']

    print(df)
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(3, 2.5), sharey=True)
    # fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(1.9, 2.5), sharey=True)

    df_melted = df.melt(
        id_vars={'rtn_type', 'example', 'trap', 'zone'}.intersection(df.columns),
        # value_vars=['amplitude_error', 'tau_high_error', 'tau_low_error'],
        value_vars=['amplitude_error', 'digitization_error'],
    )
    df_melted['value'] *= 100
    ax.set_yscale('log')
    palette = ['white', '#8aafc8', '#d9efff', '#e98d83', '#e9a783', '#fdca93']
    palette = ['#ffd4ec', '#c6ebb4', '#d9efff', '#e98d83', '#e9a783', '#fdca93']

    width = 0.5
    x = sns.boxplot(
        ax=ax,
        y='value',
        x='variable',
        hue='rtn_type',
        data=df_melted,
        orient='v',
        palette=palette,
        # palette='pastel',
        # color='white',
        # palette=['white'] * 3,
        # boxprops=dict(alpha=0.8),
        showfliers=False,
        linewidth=0.5,
        width=width,
        hue_order=['metastable', 'missing_level', 'coupled'],
    )
    print(x)
    stripplot(
        ax=ax,
        y='value',
        x='variable',
        hue='rtn_type',
        # dodge=True,
        data=df_melted,
        # orient='v',
        # color='k',
        palette=palette,
        # palette=['white'] * 3,
        color_inside_boxplot='white',
        # edgecolor=palette,
        # alpha=0.5,
        size=4,
        linewidth=0.2,
        width=width,
        hue_order=['metastable', 'missing_level', 'coupled'],
    )
    handles, labels = ax.get_legend_handles_labels()
    # Hack to remove stripplot from legend
    # Relies on their being equally-many boxplot and stripplot entries
    # and that they appear in that order
    handles = handles[:3]
    labels = labels[:3]
    labels = [l.replace('_', ' ').capitalize() for l in labels]
    ax.set_xlabel('')
    # ax.set_xlabel(
    #     r'RTN type',
    #     # r'$Q_\mathrm{wn} = \sigma_\mathrm{wn}\ '
    #     # r'/\min_\mathrm{traps}(\Delta_\mathrm{RTN})\ (\%)$',
    #     # labelpad=-3,
    #     labelpad=1,
    #     fontsize=LABEL_FONT_SIZE,
    # )
    # xticks = ax.get_xticklabels()
    # ax.set_yticklabels(yticks, rotation=90, va='center')
    ax.tick_params(axis='x', pad=2)
    ax.legend(
        handles,
        labels,
        # title=r'$N_\mathrm{traps}$',
        # title='aRTN type',
        loc='lower right',
        prop=dict(size=8),
        title_fontsize=9,
        handlelength=0.7,
        labelspacing=0.3,
        handletextpad=0.3,
        borderaxespad=0.1,
    )
    # if column == 'tau_high':
    #     ax.set_xlim(0.04, )
    ax.set_ylabel(r'$\epsilon\ (10^x\,\%)$', labelpad=0,
                  fontsize=LABEL_FONT_SIZE)
    ax.set_xticklabels([
        r'$\Delta_\mathrm{RTN}$',
        r'$\eta$',
        # r'$\bar{\tau}_\mathrm{high}$',
        # r'$\bar{\tau}_\mathrm{low}$',
    ])

    # Hide major ticks, but keep labels
    # Show minor ticks between labels
    # Requested by Prof. Kim
    ax.tick_params(axis='x', which='major', length=0)
    ax.tick_params(axis='x', which='minor', length=4)
    ax.xaxis.set_minor_locator(ticker.FixedLocator(
        [np.mean(pair) for pair in pairwise(ax.get_xticks())]
    ))

    # Show error axis in scientific notation
    ax.yaxis.set_major_formatter(
        ticker.LogFormatterExponent(base=10, labelOnlyBase=True))

    fig.tight_layout()
    fig.subplots_adjust(wspace=0)
    fig.savefig(output_file, dpi=300)
    plt.close(fig)


@click.command()
@click.argument('files', type=click.Path(), nargs=-1, required=True)
@click.argument('output-file', type=click.Path())
def plot_boxplot_models(files, output_file):
    """
    Plot a boxplot comparing models with all metrics using subplots
    """
    # %%
    df = pd.concat(Example[pd.DataFrame](f).read() for f in files)
    df['noise'] = df['noise'].apply(lambda x: int(x * 100) if x <= 1 else int(x))

    # for palette in ['vlag', 'hls', 'husl', 'Set1', 'Set2', 'Set3']:

    if PUBLISHER == Publisher.IEEE:
        noise_set = (20, 60, 100)
        df = df[df['noise'].isin(noise_set)]

    plt.close('all')

    fig, axes = plt.subplots(ncols=1, nrows=3, figsize=(6, 5), sharex=True)

    for i, (ax, n_traps) in enumerate(zip(axes, np.arange(3) + 1)):
        # df_sub = get_dataframe(df, should_sort=False, column=column)
        df_sub = df[df.n_traps == n_traps]
        df_sub = df_sub.sort_values(['model_type', 'n_traps', 'noise', 'example'])

        column = 'digitization_error'
        df_sub[column] *= 100
        # print(df_sub)

        noise_axis, error_axis = {
            Publisher.IEEE: ('y', 'x'),
            Publisher.NATURE: ('x', 'y'),
        }[PUBLISHER]
        ax = Patch(ax, noise_axis=noise_axis, error_axis=error_axis)
        ax.set_error_axis_scale('log')
        x, y = {
            Publisher.IEEE: (column, 'noise'),
            Publisher.NATURE: ('noise', column),
        }[PUBLISHER]

        palette = ['#accaea', '#f0b792', '#98daa7', '#f3aba8']
        width = 0.7
        hue_order = [x for x in ['gru', 'lstm', 'wavenet', 'hmm']
                       if x in set(df_sub.model_type)]
        sns.boxplot(
            ax=ax,
            x=x,
            y=y,
            hue='model_type',
            data=df_sub,
            orient={
                Publisher.IEEE: 'h',
                Publisher.NATURE: 'v',
            }[PUBLISHER],
            palette=palette,
            # boxprops=dict(alpha=0.8),
            showfliers=False,
            linewidth=0.5,
            width=width,
            hue_order=hue_order,
        )
        handles, _labels = ax.get_legend_handles_labels()
        labels = hue_order
        # Hack to remove stripplot from legend
        # Relies on their being equally-many boxplot and stripplot entries
        # and that they appear in that order
        # handles = handles[:3 if i > 0 else 4]
        # labels = labels[:3 if i > 0 else 4]
        stripplot(
            ax=ax,
            x=x,
            y=y,
            hue='model_type',
            dodge=True,
            data=df_sub,
            palette=palette,
            color_inside_boxplot='white',
            size=4,
            linewidth=0.2,
            width=width,
            hue_order=labels,
        )
        if i == len(axes) - 1:
            ax.set_noise_axis_label(
                r'$Q_\mathrm{wn}\ (\%)$',
                # r'$Q_\mathrm{wn} = \sigma_\mathrm{wn}\ '
                # r'/\min_\mathrm{traps}(\Delta_\mathrm{RTN})\ (\%)$',
                labelpad=0,
                fontsize=LABEL_FONT_SIZE,
            )
            # yticks = ax.get_yticklabels()
            # ax.set_yticklabels(yticks, rotation=90, va='center')
            ax.tick_params(axis=noise_axis, pad=2)
        else:
            ax.set_noise_axis_label('')

        labels = [x.upper() if x != 'wavenet' else 'WaveNet' for x in labels]
        if i == 0:
            ax.legend(
                handles,
                labels,
                # title='Trap' if should_sort else r'$N_\mathrm{traps}$',
                loc={
                    Publisher.IEEE: 'upper right',
                    Publisher.NATURE: 'lower right',
                }[PUBLISHER],
                prop=dict(size=8),
                title_fontsize=9,
                handlelength=0.7,
                labelspacing=0.3,
                handletextpad=0.3,
                borderaxespad=0.1,
                facecolor='white',
                edgecolor='white',
            )
        else:
            ax.get_legend().remove()
        ax.set_error_axis_label(
            {
                'tau_high': r'$\epsilon(\bar{\tau}_\mathrm{high})\ (10^x\,\%)$',
                'tau_low': r'$\epsilon(\bar{\tau}_\mathrm{low})\ (10^x\,\%)$',
                'amplitude_error': r'$\epsilon(\Delta_\mathrm{RTN})\ (10^x\,\%)$',
                'digitization_error': r'$\epsilon(\eta)\ (10^x\,\%)$',
            }[column],
            labelpad=1,
            fontsize=LABEL_FONT_SIZE,
        )

        # Hide major ticks, but keep labels
        # Show minor ticks between labels
        # Requested by Prof. Kim
        ax.tick_params(axis='x', which='major', length=0)
        ax.tick_params(axis='x', which='minor', length=4)
        ax.xaxis.set_minor_locator(ticker.FixedLocator(
            [np.mean(pair) for pair in pairwise(ax.get_xticks())]
        ))

        ax.annotate(
            subfig(i + 1),
            xycoords='axes fraction',
            textcoords='offset points',
            ha='left',
            **{
                Publisher.IEEE: dict(xy=(0, 0), xytext=(4, 8)),
                Publisher.NATURE: dict(xy=(0, 1), xytext=(4, -12)),
                # Publisher.NATURE: dict(xy=(0, 1), xytext=(-32, 2 - 8 * i)),
            }[PUBLISHER],
        )

    fig.tight_layout()
    fig.subplots_adjust(**{{
        Publisher.IEEE: 'wspace',
        Publisher.NATURE: 'hspace',
    }[PUBLISHER]: 0})
    fig.savefig(output_file, dpi=300)
    plt.close(fig)


__all__ = ('plot_boxplot_tau', 'plot_boxplot_digitization',
           'plot_boxplot_amplitude', 'plot_boxplot_subplots',
           'plot_boxplot_artn', 'plot_boxplot_models')


if __name__ == '__main__':
    @click.group()
    def main():
        pass

    for command in __all__:
        main.add_command(locals()[command])

    main()

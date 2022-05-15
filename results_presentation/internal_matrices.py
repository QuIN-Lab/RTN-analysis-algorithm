"""
Create the internal matrices or headmaps of the algorithm results.

Developed by Marcel Robitaille on 2021/11/05 Copyright © 2021 QuIN Lab
"""

import re
from pathlib import Path

import click
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from utils import run_click
from example import Example


# %%


def sort_examples(df: pd.DataFrame, column: str):
    """
    Sort examples by difficulty

    The example number is arbitrary, so it's better to sort the examples by
    difficulty
    """

    def sort_group(row):
        example_map = row.groupby('example')[column].sum().sort_values() \
            .reset_index().sort_values('example')
        example_map['i'] = example_map.index
        row['new_example'] = row.example.apply(lambda example:
                                               example_map.iloc[example].i)
        return row
    return df.groupby('noise').apply(sort_group)


# %%


def generate_heatmap(name, df: pd.DataFrame, column, xlabel, output_dir: Path,
                     output_format, formatter=None, vmax=None, vmin=None, heatmap=None,
                     image=None, should_sort=False):
    if heatmap is None:
        heatmap = df.pivot(index='x', columns='noise', values=column).values.T
        # image = abs(heatmap)
    else:
        pass
        # image[image == 0] = image.min()
        # image[image == 0] = heatmap[~np.isnan(heatmap)].min()
        # image = np.nan_to_num(image)

    image = heatmap if image is None else image
    print(heatmap)

    fig, ax = plt.subplots(figsize=(10, 7))

    ax.set_title(name, fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=16)

    im = ax.imshow(
        image,
        cmap='RdYlGn_r',
        vmax=vmax if vmax else min(1, image.max().max()),
        # vmin=heatmap.min().min(),
        vmin=vmin,
    )

    # noises = [f'{x:.1f}' for x in sorted(set(df['noise']))]
    if df['noise'].max() <= 1:
        df['noise'] = (df['noise'] * 100).astype(int)
    noises = [int(x) for x in sorted(set(df['noise']))]
    xs = sorted(set(df['x']))

    ax.set_xticks(range(len(xs)))
    ax.set_yticks(range(len(noises)))
    ax.set_xticklabels(xs, rotation=45, ha='right')
    ax.set_yticklabels(noises, rotation=45)
    ax.set_ylabel('Noise', fontsize=16)
    ax.set_xlabel(xlabel, fontsize=16)
    width = heatmap.shape[1] / 10
    # for x in np.arange(0, 9 * width, width) + width - 0.5:
    #     ax.axvline(x, ymin=0, ymax=1, color='white')

    fig.tight_layout()
    extent = im.get_extent()

    for i, j in np.ndindex(heatmap.shape):
        value = heatmap[i][j]
        red, green, blue, _ = im.cmap(im.norm(image[i][j]))

        text = formatter(value, i, j) if formatter \
                else r'{:.1f}\,\%'.format(value * 100)
        ax.text(
            j, i,
            text,
            color='black' if (red*76.245 + green*149.7 + blue*29.07) > 186
            else 'white',
            ha='center',
            va='center',
            fontsize=200 / (len(text) * np.sqrt(max(extent[1:3]))) if len(text) > 1 else 20,
            # fontsize=10,
            rotation=45 if len(text) > 1 else 0,
        )
    filename = f'{name}_heatmap{"_sorted" if should_sort else ""}.{output_format}'
    filename = filename.replace('\\mathrm', '')
    filename = re.sub(r'[\\\${}]', '', filename).replace(' ', '_')
    filename = output_dir / filename
    print(filename)
    # plt.show()
    fig.savefig(filename, dpi=300)
    # subprocess.call([Path.home() / '.local/bin/symmetric_crop', filename])
    plt.close(fig)


# %% Tau


@click.command()
@click.argument('files', type=click.Path(), nargs=-1, required=True)
@click.argument('output-dir', type=click.Path(), required=True)
def plot_heatmap_tau(files, output_dir):
    """
    Generate the Tau heatmap reading tau results from FILES (pandas dataframes,
    supports `.csv` or `.feather`) and save the figures to OUTPUT-DIR.
    """

    output_dir = Path(output_dir)
    # Gaurd against saving the figure to the last `.csv` or `.feather` given
    assert output_dir.is_dir(), 'You may have forgotten to specify OUTPUT'

    df = pd.concat(Example(f).read() for f in files)

    for n_traps, df_sub in df.groupby('n_traps'):
        n_traps = int(n_traps)
        for column in ['tau_high', 'tau_low']:
            df_sub = sort_examples(df=df_sub, column=column)
            df_sub['x'] = df_sub[['new_example', 'trap']] \
                .apply(lambda args: '{}/{}'.format(*map(int, args)), axis=1)
            generate_heatmap(
                name={
                    'tau_high': fr'{n_traps}-trap $\tau_\mathrm{{high}}$ error',
                    'tau_low': fr'{n_traps}-trap $\tau_\mathrm{{low}}$ error',
                }[column],
                column=column,
                df=df_sub,
                xlabel='Example / Trap',
                output_dir=output_dir,
            )


if __name__ == '__main__':
    data_dir = Path.home() / 'OneDrive/02. QuIN_Research/31. Noise-RTN' / \
            '01. 2021_Algorithm paper/rnn_results_tracking' / \
            'Run-8_2021-12-07_RNN_fix_1-trap_low_tau_accuracy'
    run_click(plot_heatmap_tau, *data_dir.glob('*_tau_error.csv'), '/tmp/')


# %% Digitization


@click.command()
@click.argument('files', type=click.Path(), nargs=-1, required=True)
@click.argument('output-dir', type=click.Path(), required=True)
@click.option('--sorted/--unsorted', 'should_sort', is_flag=True,
              help='Whether to sort the rows')
@click.option('--output-format', type=click.Choice(['pdf', 'png']),
              default='pdf')
@click.option('--subtract', default=None, multiple=True)
@click.option('--name-suffix', default='')
def plot_heatmap_digitization(files, output_dir, should_sort, output_format,
                              subtract, name_suffix):
    """
    Generate the digitization heatmap reading tau results from FILES
    (pandas dataframes, supports `.csv` or `.feather`) and save the figures to
    OUTPUT-DIR.
    """

    # Gaurd against saving the figure to the last `.csv` or `.feather` given
    assert output_dir.endswith('/'), 'You may have forgotten to specify OUTPUT'
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    df = pd.concat(Example(f).read() for f in files)
    if subtract:
        subtract_df = pd.concat(Example(f).read() for f in subtract)

    column = 'digitization_error'

    for n_traps, df_sub in df.groupby('n_traps'):
        n_traps = int(n_traps)
        print(df_sub)
        df_sub.sort_values(['noise', 'example'], inplace=True)
        if 'digitization_accuracy' in df.columns and \
                'digitization_error' not in df.columns:
            df_sub['digitization_error'] = 1 - df_sub['digitization_accuracy']

        if subtract:
            subtract_df_sub = subtract_df[subtract_df.n_traps == n_traps]
            if len(subtract_df_sub) == 0:
                continue
            if 'digitization_accuracy' in subtract_df_sub.columns and \
                    'digitization_error' not in subtract_df_sub.columns:
                subtract_df_sub['digitization_error'] = \
                    1 - subtract_df_sub['digitization_accuracy']
            df_sub = pd.merge(df_sub, subtract_df_sub, suffixes=('', '_sub'),
                                 on=['n_traps', 'noise', 'example', 'trap'])
            df_sub['digitization_error'] -= df_sub['digitization_error_sub']

        df_sub = sort_examples(df=df_sub, column=column) \
            if should_sort else \
            df_sub.rename(columns={'example': 'new_example'}, copy=True)

        # df_sub = df_sub[df_sub['trap'] == 0]
        # df_sub.drop(columns='trap', inplace=True)

        df_sub['x'] = df_sub[['new_example', 'trap']] \
                .apply(lambda args: '{}/{}'.format(*map(int, args)), axis=1) \
            if 'trap' in df_sub.columns \
            else df_sub['new_example']

        generate_heatmap(
            name=f'{n_traps}-trap digitization error {name_suffix}',
            column=column,
            df=df_sub,
            xlabel='Example',
            output_dir=output_dir,
            output_format=output_format,
            should_sort=should_sort,
        )


if __name__ == '__main__':
    data_dir = Path.home() / 'OneDrive/02. QuIN_Research/31. Noise-RTN' / \
            '01. 2021_Algorithm paper/rnn_results_tracking' / \
            'Run-8_2021-12-07_RNN_fix_1-trap_low_tau_accuracy'
    run_click(
        plot_heatmap_digitization,
        *data_dir.glob('*_digitization_accuracy.csv'),
        '/tmp/',
    )


# %% Amplitude


@click.command()
@click.argument('files', type=click.Path(), nargs=-1, required=True)
@click.argument('output-dir', type=click.Path(), required=True)
@click.option('--output-format', type=click.Choice(['pdf', 'png']),
              default='pdf')
def plot_heatmap_amplitude(files, output_dir, output_format):
    """
    Generate the amplitude heatmap reading tau results from FILES
    (pandas dataframes, supports `.csv` or `.feather`) and save the figures to
    OUTPUT-DIR.
    """

    print(output_dir)
    output_dir = Path(output_dir)
    # Gaurd against saving the figure to the last `.csv` or `.feather` given
    assert output_dir.is_dir(), 'You may have forgotten to specify OUTPUT'

    df = pd.concat(Example(f).read() for f in files)

    column = 'amplitude_error'

    for n_traps, df_sub in df.groupby('n_traps'):
        # df_sub = sort_examples(df=df_sub, column=column)
        df_sub['x'] = df_sub[['example', 'trap']] \
            .apply(lambda args: '{}/{}'.format(*map(int, args)), axis=1)

        generate_heatmap(
            name=f'{n_traps}-trap amplitude error',
            column=column,
            df=df_sub,
            xlabel='Example / Trap',
            output_dir=output_dir,
            output_format=output_format,
        )


# %% n_traps


@click.command()
@click.argument('files', type=click.Path(), nargs=-1, required=True)
@click.argument('output-dir', type=click.Path(), required=True)
@click.option('--output-format', type=click.Choice(['pdf', 'png']),
              default='pdf')
def plot_heatmap_ntraps(files, output_dir, output_format):
    print(output_dir)
    output_dir = Path(output_dir)
    # Gaurd against saving the figure to the last `.csv` or `.feather` given
    assert output_dir.is_dir(), 'You may have forgotten to specify OUTPUT'

    df = pd.concat(Example(f).read() for f in files)
    df = df[df['trap'] == 0]
    df['noise'] *= 100

    column = 'predicted_n_traps'

    groupby = 'rtn_type' \
        if 'rtn_type' in df.columns and len(set(df['rtn_type'])) > 1 \
        else 'n_traps'
    for n_traps, df_sub in df.groupby(groupby):
        continue
        # print(df_sub)
        # df_sub['x'] = df_sub[['example', 'trap']] \
        #     .apply(lambda args: '{}/{}'.format(*map(int, args)), axis=1)
        df_sub['rate'] = 1 - (df_sub.predicted_n_traps / df_sub.n_traps)
        df_sub['rate'] = df_sub['rate'].apply(lambda x: x if x >= 0 else 1)
        # df_sub = sort_examples(df=df_sub, column=column)
        df_sub['x'] = df_sub['example']
        heatmap = df_sub.pivot(
            index='x',
            columns='noise',
            values='rate',
        ).values.T

        generate_heatmap(
            name=f'{n_traps}-trap ntraps',
            column=column,
            df=df_sub,
            xlabel='Example',
            output_dir=output_dir,
            output_format=output_format,
            image=heatmap,
            formatter='{:.0f}'.format,
        )
    for n_traps, df_sub in df.groupby(groupby):
        df_sub['rate'] = 1 / 10
        test = df_sub.groupby(['predicted_n_traps', 'noise'])['rate'].sum()
        test = test.reset_index()
        test = test.pivot(
            index='predicted_n_traps',
            columns='noise',
            values='rate',
        ).fillna(0)
        print(type(test))
        image = test.copy()
        print(test)
        image[image.index == n_traps] = 1 - image[image.index == n_traps]
        image = image.values.T
        test = test.values.T
        df_sub['x'] = df_sub['predicted_n_traps']
        generate_heatmap(
            name=f'{n_traps}-trap ntraps rate',
            column=column,
            df=df_sub,
            xlabel=r'$N_\mathrm{traps}$',
            output_dir=output_dir,
            output_format=output_format,
            heatmap=test,
            image=image,
        )

        # heatmap = df_sub.pivot(
        #     index='x',
        #     columns='noise',
        #     values='predicted_n_traps',
        # )

__all__ = ('plot_heatmap_amplitude', 'plot_heatmap_digitization',
           'plot_heatmap_tau', 'plot_heatmap_ntraps')

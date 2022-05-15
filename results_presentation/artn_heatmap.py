"""
This file creates heatmaps to show the error in predicting Δ_RTN, τ_high, τ_low,
and digitization for aRTN

Developed by Marcel Robitaille on 2022/02/09 Copyright © 2021 QuIN Lab
"""


import re
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from example import Example

@click.command()
@click.argument('files', type=click.Path(), nargs=-1, required=True)
@click.argument('output-dir', type=click.Path())
@click.argument('column')
@click.option('--output-format', type=click.Choice(['pdf', 'png']),
              default='pdf')
@click.option('--data-type', type=click.Choice(['percent', 'whole']),
              default='percent')
def plot_heatmap_artn(files, output_dir, output_format, column, data_type):
    """
    Generate an aRTN heatmap results from FILES
    (pandas dataframes, supports `.csv` or `.feather`) and save the figures to
    OUTPUT-DIR.
    """

    output_dir = Path(output_dir)

    # Gaurd against saving the figure to the last `.csv` or `.feather` given
    assert output_dir.is_dir(), 'You may have forgotten to specify OUTPUT'

    df = pd.concat(Example(f).read() for f in files)

    print(df)
    assert 'coupled' in set(df['rtn_type'])
    # pylint: disable=redefined-outer-name
    df['trap'] = df['trap'].apply(int)
    fig, axes = plt.subplots(nrows=3, figsize=(6, 6))
    for rtn_type, ax in \
            zip(['coupled', 'missing_level', 'metastable'], axes):
        df_sub = df[df['rtn_type'] == rtn_type]
        columns = 'zone' \
            if len(set(df_sub.trap)) == 1 and 'zone' in df.columns \
            else 'trap'
        heatmap = df_sub.pivot(
            index='example',
            columns=columns,
            values=column,
        ).values.T
        image = abs(heatmap)

        ax.set_title(rtn_type, fontsize=16)
        ax.tick_params(axis='both', which='major', labelsize=16)

        im = ax.imshow(
            image,
            cmap='RdYlGn_r',
            # vmax=min(1, heatmap.max().max()),
        )

        ylabels = sorted(set(df_sub[columns]))
        xlabels = sorted(set(df_sub['example']))

        ax.set_xticks(range(len(xlabels)))
        ax.set_yticks(range(len(ylabels)))
        ax.set_xticklabels(xlabels, rotation=45, ha='right')
        ax.set_yticklabels(ylabels, rotation=45)
        ax.set_ylabel(columns.title(), fontsize=16)
        ax.set_xlabel('Example', fontsize=16)

        extent = im.get_extent()

        for i, j in np.ndindex(heatmap.shape):
            value = heatmap[i][j]
            red, green, blue, _ = \
                im.cmap(im.norm(value))

            text = r'{:.1f}\,\%'.format(value * 100) if data_type == 'percent' \
                else str(value)
            ax.text(
                j, i,
                text,
                color='black' if (red*76.245 + green*149.7 + blue*29.07) > 186
                else 'white',
                ha='center',
                va='center',
                fontsize=200 / (len(text) * np.sqrt(max(extent[1:3]))),
                # fontsize=20,
                # fontsize=10,
                rotation=45,
            )
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.4)

    name = column
    # filename = f'{name}_heatmap{"_sorted" if should_sort else ""}.{output_format}'
    filename = f'artn_{name}_heatmap.{output_format}'
    filename = filename.replace('\\mathrm', '')
    filename = re.sub(r'[\\\${}]', '', filename).replace(' ', '_')
    filename = output_dir / filename
    fig.savefig(output_dir / filename, dpi=300)


# # %% Digitiation

# # These were copied from the artn_debugging.py output
# # when the accuracy calculated after processing RNN was not trustworthy
# metastable = {
#     0: {'trap_0': 0.998507},
#     1: {'trap_0': 0.998441},
#     2: {'trap_0': 0.998512},
#     3: {'trap_0': 0.99852},
#     4: {'trap_0': 0.998663},
#     5: {'trap_0': 0.998495},
#     6: {'trap_0': 0.998226},
#     7: {'trap_0': 0.99843},
#     8: {'trap_0': 0.998425},
#     9: {'trap_0': 0.998342}}

# coupled = {
#     0: {'trap_0': 0.998115, 'trap_1': 0.99868},
#     1: {'trap_0': 0.996152, 'trap_1': 0.998444},
#     2: {'trap_0': 0.98993, 'trap_1': 0.989074},
#     3: {'trap_0': 0.785616, 'trap_1': 0.783687},
#     4: {'trap_0': 0.998278, 'trap_1': 0.99683},
#     5: {'trap_0': 0.992463, 'trap_1': 0.99233},
#     6: {'trap_0': 0.994045, 'trap_1': 0.998857},
#     7: {'trap_0': 0.997139, 'trap_1': 0.996045},
#     8: {'trap_0': 0.997769, 'trap_1': 0.75687},
#     9: {'trap_0': 0.998365, 'trap_1': 0.998805}}

# missing_level = {
#     0: {'trap_0': 0.997553, 'trap_1': 0.997638},
#     1: {'trap_0': 0.998188, 'trap_1': 0.998063},
#     2: {'trap_0': 0.99877, 'trap_1': 0.997402},
#     3: {'trap_0': 0.998001, 'trap_1': 0.996929},
#     4: {'trap_0': 0.997101, 'trap_1': 0.998209},
#     5: {'trap_0': 0.998173, 'trap_1': 0.99729},
#     6: {'trap_0': 0.997726, 'trap_1': 0.997345},
#     7: {'trap_0': 0.996094, 'trap_1': 0.996826},
#     8: {'trap_0': 0.997951, 'trap_1': 0.99841},
#     9: {'trap_0': 0.99798, 'trap_1': 0.998224}}

# # df = pd.read_csv(results_dir / 'digitization_accuracy.csv')
# def generator():
#     for artn_type in ('missing_level', 'metastable', 'coupled'):
#         for example, acc in globals()[artn_type].items():
#             for trap, value in acc.items():
#                 trap = int(trap[-1])
#                 yield artn_type, example, trap, value
# df = pd.DataFrame(
#     generator(),
#     columns=['artn_type', 'example', 'trap', 'digitization_accuracy'],
# )
# df['digitization_error'] = 1 - df['digitization_accuracy']
# # df.to_csv(results_dir / 'digitization_accuracy.csv', index=False)

# create_heatmap(
#     df=df,
#     column='digitization_error',
#     figure_filename='digitization_accuracy_heatmap.png',
# )


# # %% Tau
# df = pd.read_csv('results.csv')

# for column in ('tau_high_error', 'tau_low_error'):
#     create_heatmap(
#         df=df,
#         column=column,
#         figure_filename=f'tau_error_{column}_heatmap.png',
#     )


__all__ = ('plot_heatmap_artn',)

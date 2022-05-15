"""
Plot grid of RTN and aRTN examples
with time-series and time-lag plots.
The 1-trap 20% white noise example will include annotations defining the 3 RTN
parameters.

Currently plots:
Normal white noise RTN: 3x3 grid. 1,2,3 traps on the rows, 20%, 60%, 100% noise on the columns
Normal 1/f noise RTN: 1x3. 1-trap 20%, 2-traps 60%, 3-traps 100%
Anomalous RTN: 1x3 with 3 different aRTN types

Developed by Marcel Robitaille on 2022/01/30 Copyright © 2021 QuIN Lab
"""

import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
from rich import print

from figure_constants import LABEL_FONT_SIZE
from constants import DATA_DIR, ARTN_DATA_DIR, PINK_DATA_DIR
from example import Example
from utils import subfig, nearest_multiple
from data_exploration.plot_time_lag import add_tlp

# %%

# plt.switch_backend('Qt5Agg')
plt.switch_backend('Agg')
plt.style.use(['science'])

RAW = '#999'  # gray


# PADDING = 18
PADDING = 0


# %%


def double_arrow(ax, x1, x2, y1, y2, text, position, color, fontsize):
    mid_x = (x1 + x2) / 2
    mid_y = (y1 + y2) / 2
    kwargs = dict(
        head_width=5 if y1 == y2 else 200,
        head_length=250 if y1 == y2 else 7,
        linewidth=0.8,
        length_includes_head=True,
        color=color,
        zorder=100,
    )
    # For some reason, arrows appear a tiny bit too long
    shrink = 0.90
    ax.arrow(mid_x, mid_y, (mid_x - x1) * shrink, (mid_y - y1) * shrink, **kwargs)
    ax.arrow(mid_x, mid_y, (mid_x - x2) * shrink, (mid_y - y2) * shrink, **kwargs)
    # padding = 0.2
    kwargs = {
        'above': dict(xytext=(0, 12), ha='center', va='bottom'),
        'below': dict(xytext=(0, -12), ha='center', va='top'),
        'right': dict(xytext=(2, -2), ha='left', va='center'),
        'left': dict(xytext=(-2, -10), ha='right', va='center'),
    }[position]
    ax.annotate(
        text=text,
        fontsize=fontsize,
        color=color,
        xy=(mid_x, mid_y),
        textcoords='offset points',
        **kwargs,
    )



def generate_plot(example, ax, fig, df, color, i, annotation=''):
    # tlp_ax, time_ax = add_tlp(example=example, ax=ax, fig=fig)
    time_ax = ax
    tlp_ax = None

    time_ax.plot(df['full_signal'], linewidth=0.5, color=RAW)
    time_ax.set_xlim(0, max(df.index))
    time_ax.plot(df['rtn_sum'], linestyle='dashed', linewidth=1, c=color)
    # time_ax.set_yticks([])

    # time_ax.annotate(
    #     f'{subfig(i)} {annotation}',
    #     xy=(0, 0),
    #     xytext=(2, 6),
    #     xycoords='axes fraction',
    #     textcoords='offset points',
    #     ha='left',
    #     fontsize=9,
    # )

    # xticks = np.array([0, 2_000, 4_000, 6_000, 8_000])
    xticks = np.array([0, 9_000])
    time_ax.set_xticks(xticks)
    time_ax.set_xticklabels(xticks // 1_000)

    time_ax.set_ylabel('Intensity (a.u.)', fontsize=LABEL_FONT_SIZE,
            labelpad=-17)
    # ax.yaxis.set_label_coords(-0.02, 0.45)

    # Use the full dataframe, not the zoomed-in dataframe, to determine the
    # limits
    full_df = example.read()
    quantile = 1e-4
    lo, hi = full_df.full_signal.quantile([quantile, 1 - quantile])
    print(lo, hi)
    padding = (hi - lo) / 5
    top_padding = (hi - lo) / 7
    time_ax.set_ylim(lo - padding, hi + top_padding)

    yticks = [nearest_multiple(10, x) for x in time_ax.get_ylim()]
    time_ax.set_yticks(yticks)
    time_ax.set_yticklabels([f'${x}$' for x in yticks])
    time_ax.set_xlabel(r'Signal step ($\times 10^3$)', fontsize=LABEL_FONT_SIZE,
                  labelpad=-1)
    time_ax.xaxis.set_label_coords(4 / 9, -0.03)
    time_ax.xaxis.set_major_locator(MultipleLocator(1_000))
    time_ax.xaxis.set_minor_locator(MultipleLocator(200))
    # ax.tick_params(axis='both', which='both', direction='in', top=True,
    #                right=True, labelsize=8)

    if tlp_ax is not None:
        tlp_ax.set_xlim(time_ax.get_ylim())
        tlp_ax.set_ylim(time_ax.get_ylim())
        ticks = [nearest_multiple(10, x) for x in time_ax.get_ylim()]

        # How much to offset the TLP ticks from the time-series ticks
        # If they are too close, they could conflict on the vertical edge between
        # plots and the leftmost tick on the bottom of the TLP plot
        # could be mistakenly for a tick of the TLP plot
        offset = 10 if (ticks[1] - ticks[0]) < 100 else 30
        ticks = [ticks[0] + offset, ticks[1] - offset]
        tlp_ax.set_xticks(ticks)
        tlp_ax.set_yticks(ticks)
        tlp_ax.set_xticklabels(f'${x}$' for x in ticks)
        tlp_ax.set_yticklabels(f'${x}$' for x in ticks)

    return df, time_ax


def add_param_definition(ax, df):
    ax.arrow(0, 0, 1, 1)
    transitions = df[df['rtn_sum'].diff() != 0].index
    y = df['rtn_sum'].max() + PADDING
    high_index = 6
    double_arrow(
        ax, transitions[high_index], transitions[high_index + 1], y, y,
        text=r'$\tau_\mathrm{high}$',
        position='above',
        color=color,
        fontsize=15,
    )
    y = -PADDING
    low_index = 5
    double_arrow(
        ax, transitions[low_index], transitions[low_index + 1], y, y,
        text=r'$\tau_\mathrm{low}$',
        position='below',
        color=color,
        fontsize=15,
    )
    x = transitions[3]
    double_arrow(ax, x, x, df['rtn_sum'].min(), df['rtn_sum'].max(),
                 text=r'$\Delta_\mathrm{RTN}$', position='left', color=color,
                 fontsize=12)


# Normal RTN

normal_examples = [
    (
        'white',
        Example(
            DATA_DIR /
            f'{n_traps}-trap_wn={noise:.1f}_example={example_number}',
        ).signals,
        example_number,
        n_traps,
        noise,
    ) for (n_traps, noise, example_number) in [
        (1, 0.2, 7),
        # (1, 0.6, 7),
        # (1, 1.0, 7),
        # (2, 0.2, 3),
        (2, 0.6, 4),
        # (2, 1.0, 3),
        # (3, 0.2, 2),
        # (3, 0.6, 2),
        (3, 1.0, 2),
    ]
]

# 1/f examples

pink_examples = [
    (
        'pink',
        Example(
            PINK_DATA_DIR /
            f'{n_traps}-trap_wn={noise:.1f}_example={example_number}'
        ).signals,
        example_number,
        n_traps,
        noise,
    ) for (n_traps, noise, example_number) in [
        (1, 0.2, 7),
        (2, 0.6, 3),
        (3, 1.0, 2),
    ]
]

# aRTN examples

artn_examples = [
    (
        'artn',
        Example(ARTN_DATA_DIR / f'artn_{artn_type}_example={example_number}')
        .signals,
        example_number,
        artn_type,
    )
    for artn_type, example_number in [
        ('metastable', 6),
        ('missing_level', 0),
        ('coupled', 0),
    ]
]


# examples = [*normal_examples, *pink_examples, *artn_examples]
examples = [*normal_examples]

color = '#0075ff'
plt.close('all')
fig, axes = plt.subplots(ncols=3, nrows=len(examples) // 3, figsize=(9, 1.5))
# fig.subplots_adjust(wspace=0.31, hspace=0.25)
fig.subplots_adjust(wspace=0.23, hspace=0.25)
for (i, ax), (rtn_type, example, example_number, *args) in zip(
    enumerate(axes.flatten()),
    examples,
):
    df = example.read()

    def get_start():
        """
        Different examples have different starting indices
        """

        if rtn_type in ('white', 'pink'):
            n_traps, _noise = args
            return {
                1: 18_000,
                2: 18_000,
                3: 20_000,
            }[n_traps]

        if rtn_type == 'anomalous':
            artn_type, = args
            return {
                'metastable': 15_000,
                'coupled': 0,
                'missing_level': 15_500,
            }[artn_type]

    width = 10_000
    start = get_start()
    df = df.iloc[start:].reset_index(drop=True)[:width]

    def get_annotation(rtn_type):
        if rtn_type in ('white', 'pink'):
            n_traps, noise = args
            noise_type = {
                'white': 'white',
                'pink': '$1/f$',
            }[rtn_type]
            annotation = f'{n_traps} trap{"" if n_traps == 1 else "s"}, ' \
                rf'{int(noise * 100)}\,\% {noise_type} noise'
            return annotation

        if rtn_type == 'artn':
            artn_type, = args
            annotation = artn_type.replace('_', '-').capitalize() + \
                ' aRTN'
            return annotation
        return ''

    df, ax = generate_plot(
        example=example,
        ax=ax,
        fig=fig,
        df=df,
        color=color,
        i=i + 1,
        annotation=get_annotation(rtn_type=rtn_type),
    )
    if i == 0:
        add_param_definition(df=df, ax=ax)

    if rtn_type == 'anomalous' and (artn_type := args[0]) == 'metastable':
        transition = df[df.zone.diff() != 0].index.max()
        ax.axvspan(0, transition, facecolor='red', alpha=0.10)
        ax.axvspan(transition, max(df.index), facecolor='blue', alpha=0.10)
        xytext = (0, -12)
        ax.annotate(
            'Zone A',
            xy=(transition / 2, ax.get_ylim()[1]),
            xytext=xytext,
            textcoords='offset points',
            ha='center',
        )
        ax.annotate(
            'Zone B',
            xy=((transition + max(df.index)) / 2, ax.get_ylim()[1]),
            xytext=xytext,
            textcoords='offset points',
            ha='center',
        )

    # Add a single label per row on the left-hand side
    # if i % 3 == 0:
    #     ax.annotate(
    #         subfig((i // 3) + 1),
    #         xy=(0, ax.get_ylim()[1]),
    #         xytext=(-25, 0),
    #         va='center',
    #         ha='right',
    #         textcoords='offset points',
    #     )
    ax.annotate(
        f'{subfig(i + 1)} {i+1} trap{"s" if i > 0 else ""}',
        xy=(0, 0),
        xytext=(2, 5),
        va='bottom',
        ha='left',
        xycoords='axes fraction',
        textcoords='offset points',
    )


fig.savefig('/tmp/test.pdf', dpi=300)
plt.close(fig)

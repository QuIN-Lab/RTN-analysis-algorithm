"""
Plot the CNT digitization plot like Fig. 5 in our algorithm paper EDL
submission.

Developed by Marcel Robitaille on 2022/02/09 Copyright © 2021 QuIN Lab
"""

import click
import numpy as np
import matplotlib.pyplot as plt

from example import Example
from utils import nearest_multiple
from figure_constants import LABEL_FONT_SIZE
from data_exploration.plot_time_lag import add_tlp

plt.style.use('science')


def plot_cnt_data_prediction(example: Example, do_inset: bool,
                             should_add_tlp: bool):
    # plt.switch_backend('TkAgg')
    example = Example(example)
    assert example.path.name.endswith('_signals.feather')

    signals = example.read()
    kde_data = example.kde_data.read().squeeze()
    mean_vec = sorted(kde_data.peaks_intensities.tolist())
    amp_guesses = np.array(mean_vec[1:]) - mean_vec[0]
    pred = example.time_series_predictions.read()
    if 'trap_1' in pred.columns:
        pred['trap_1'] = pred['trap_1'] & ~pred['trap_0']
    pred = pred.apply(lambda x: x * amp_guesses[int(x.name[-1])])
    pred['sum'] = pred.sum(axis=1)
    pred['sum'] = pred['sum'].rolling(2).min()
    pred['sum'] += min(mean_vec)

    plt.close()
    fig, ax = plt.subplots(figsize=(7, 1.75))

    tlp_ax, ax = add_tlp(example=example, fig=fig, ax=ax) if should_add_tlp \
        else (None, ax)

    signals['pred'] = pred['sum'].dropna().shift(62)
    signals = signals[(signals.time > 15) & (signals.time < 545)]
    signals = signals.dropna()
    signals['time'] -= min(signals['time'])

    # x = np.arange(len(signals)) - 63
    ax.plot(signals.time, signals['full_signal'], linewidth=0.5)
    ax.plot(signals.time, signals['pred'], c='red', linewidth=0.5)

    ax.set_xticks([0, 100, 200, 300, 400, 500])
    ax.set_xlim(0, signals.time.max())
    ax.set_ylabel(r'$I$ ($\mu$A)', fontsize=LABEL_FONT_SIZE, labelpad=0.5)
    ax.set_xlabel(r'Time $(s)$', fontsize=LABEL_FONT_SIZE, labelpad=1)
    ax.tick_params(axis='both', which='both', direction='in',
                   top=True, right=True, labelsize=10, pad=2)

    if do_inset:
        inset = ax.inset_axes([0.61, 0.54, 0.29, 0.46])
        xticks = [195]
        xticks.append(xticks[0] + 5)
        # xticks = [195, 200]
        xlim = [xticks[0] - 0.5, xticks[1] + 0.5]
        signals_sub = signals[
            (signals.time >= xlim[0]) & (signals.time <= xlim[1])]
        inset.set_xticks(xticks)
        inset.set_xticklabels(xticks)
        inset.plot(signals_sub.time, signals_sub['full_signal'], linewidth=0.5)
        inset.plot(signals_sub.time, signals_sub['pred'], c='red',
                   linewidth=0.5)
        inset.set_xlim(xlim)
        inset.set_ylabel(r'$I$ ($\mu$A)', fontsize=10, labelpad=1)
        inset.set_xlabel(r'Time $(s)$', labelpad=-4, fontsize=10)
        inset.tick_params(axis='both', which='both', direction='in',
                          top=True, right=True, labelsize=10, pad=2)

    if tlp_ax is not None:
        tlp_ax.set_xlim(ax.get_ylim())
        tlp_ax.set_ylim(ax.get_ylim())
        print(ax.get_ylim())
        ticks = [nearest_multiple(0.01, x) for x in ax.get_ylim()]
        print(ticks)

        # How much to offset the TLP ticks from the time-series ticks
        # If they are too close, they could conflict on the vertical edge between
        # plots and the leftmost tick on the bottom of the TLP plot
        # could be mistakenly for a tick of the TLP plot
        # offset = 10 if (ticks[1] - ticks[0]) < 100 else 30
        offset = 0.005 if example.path.name.startswith('03') else 0.01
        precision = 3 if example.path.name.startswith('03') else 2
        ticks = [ticks[0] + offset, ticks[1] - offset]
        print(ticks)
        tlp_ax.set_xticks(ticks)
        tlp_ax.set_yticks(ticks)
        tlp_ax.set_xticklabels(f'${x:.{precision}f}$' for x in ticks)
        tlp_ax.set_yticklabels(f'${x:.{precision}f}$' for x in ticks)

        tlp_ax.set_xlabel(r'$I_i\ (\mu{}A)$', labelpad=1)
        tlp_ax.set_ylabel(r'$I_{i+1}\ (\mu{}A)$', labelpad=10)
        tlp_ax.yaxis.set_label_coords(1.2, 0.5)

    # fig.subplots_adjust(bottom=0)
    # fig.tight_layout()
    fig_filename = example.with_name('predicted_time_series.pdf').path
    print(fig_filename)
    fig.savefig(fig_filename, dpi=300)
    # plt.show()
    plt.close(fig)


@click.command('plot-cnt-data-prediction')
@click.argument('files', type=click.Path(), nargs=-1, required=True)
@click.option('--inset/--no-inset', type=bool, required=True)
@click.option('--tlp/--no-tlp', 'should_add_tlp', type=bool, required=True,
              help='Whether to add a time-lag plot')
def plot_cnt_data_prediction_click(files, inset, should_add_tlp):
    """
    Plot time-series predictions for a specific CNT result
    specified by FILES (specify `_signals.feather` files).
    """

    for example in files:
        plot_cnt_data_prediction(example=example, do_inset=inset,
                                 should_add_tlp=should_add_tlp)


__all__ = ('plot_cnt_data_prediction_click',)


if __name__ == '__main__':
    @click.group()
    def main():
        pass

    for command in __all__:
        main.add_command(locals()[command])

    main()

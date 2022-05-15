"""
This file was used to plot many signals from the artn RNN predictions to debug
the poor performance.

It has served its purpose, but I am keeping it in case it is of some use

Developed by Marcel Robitaille on 2022/02/09 Copyright © 2021 QuIN Lab
"""


from pathlib import Path
from itertools import chain, product

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tau_extraction import get_taus
from constants import ARTN_DATA_DIR


RESULTS_DIR = Path.home() / \
    'code/quin/CMOS_RTN/Lu/2022_01_27_artn_generate_with_ratio/' \
    'raw_data_from_server/'

# %%


def norm(data):
    return data / max(data.dropna())


acc = {}


def generator():
    for artn_type, example in product(
            ('metastable',),
            # ('missing_level', 'coupled', 'metastable'),
            # range(10),
            [0],
    ):
        def file(end, directory=ARTN_DATA_DIR):
            # pylint: disable=cell-var-from-loop
            return Path(directory) / f'artn_{artn_type}_example={example}_{end}'

        signals = pd.read_feather(file('signals.feather'))
        print(signals)

        if artn_type == 'metastable':
            signals['trap_0'] = signals['rtn_sum']

        if artn_type == 'coupled':
            true_trap_0_amp = signals[
                (signals['trap_0'] > 0) & (signals['trap_1'] == 0)
            ]['trap_0'].iloc[0]
            true_trap_1_amp = signals[
                (signals['trap_1'] > 0) & (signals['trap_0'] == 0)
            ]['trap_1'].iloc[0]

            signals['trap_0'][signals['trap_0'] > 0] = true_trap_0_amp
            signals['trap_1'][signals['trap_1'] > 0] = true_trap_1_amp
            assert set(signals['trap_0']) == {0, true_trap_0_amp}
            assert set(signals['trap_1']) == {0, true_trap_1_amp}
        pred = pd.read_feather(
            file('time_series_predictions.feather', RESULTS_DIR))
        parameters = pd.read_csv(file('parameters.csv'))
        print(parameters)
        decomp_data = pd.read_csv(file('decomp_data_traps.csv'))
        if artn_type == 'coupled':
            *amp_guesses, fudge = decomp_data['sep'].to_list()
        else:
            amp_guesses = decomp_data['sep'].to_list()
        pred = pred.shift(63)

        if artn_type != 'metastable' and \
                (norm(signals['trap_0'].iloc[:len(pred)]) == pred['trap_0']).sum() + \
                (norm(signals['trap_1'].iloc[:len(pred)]) == pred['trap_1']).sum() < \
                (norm(signals['trap_0'].iloc[:len(pred)]) == pred['trap_1']).sum() + \
                (norm(signals['trap_1'].iloc[:len(pred)]) == pred['trap_0']).sum():
            print('flipping')
            amp_guesses = amp_guesses[::-1]
            pred = pred.rename(columns={
                'trap_0': 'trap_1',
                'trap_1': 'trap_0',
            })

        n_traps = len(pred.columns)
        pred = pred.rolling(17, center=True).median()
        pred = pred.apply(lambda x: x * amp_guesses[int(x.name[-1])]) # pylint: disable=cell-var-from-loop
        pred['sum'] = pred.sum(axis=1)
        if artn_type == 'coupled':
            pred['sum'][(pred['trap_0'] > 0) & (pred['trap_1'] > 0)] = fudge
        # pred['sum'] = pred['sum'].rolling(2).min()
        signals['pred_trap_0'] = pred['trap_0']
        try:
            signals['pred_trap_1'] = pred['trap_1']
        except KeyError:
            pass

        pred['zone'] = pred['sum'].rolling(10000, center=True).median()
        # pred['zone'] = signals['zone']
        pred['truth'] = signals['rtn_sum']
        signals['pred'] = pred['sum']
        signals['pred_zone'] = pred['zone']
        print(signals)
        # print(signals)
        # signals = signals.dropna()
        # trap_0_acc = ((norm(signals['trap_0']) == norm(signals['pred_trap_0'])).sum() / len(signals))
        # trap_1_acc = ((norm(signals['trap_1']) == norm(signals['pred_trap_1'])).sum() / len(signals))
        # acc[example] = {'trap_0': trap_0_acc, 'trap_1': trap_1_acc}
        # print(acc[example])

        if artn_type == 'metastable':
            digitization_accuracy = (
                (norm(signals['trap_0']) ==
                 norm(signals['pred_trap_0'])).sum() / len(signals))
            for zone in ('a', 'b'):
                zone_df = pred[pred['zone'] > 0
                               if zone == 'a' else ~(pred['zone'] > 0)]

                class Grouper:
                    def __init__(self):
                        self.prev = None
                        self.group = 0

                    def __call__(self, x):
                        if self.prev is not None and \
                                abs(self.prev - x.name) > 1:
                            self.group += 1
                        self.prev = x.name
                        return self.group

                zone_df['group'] = zone_df.apply(Grouper(), axis=1)
                tau_high, tau_low = pd.DataFrame([
                    get_taus(group.trap_0, group.index)
                    for _, group in zone_df.groupby('group')
                ]).apply(lambda x: np.mean(list(chain.from_iterable(x))))

                def error(a, b):
                    return abs(a - b) / b

                tau_high = error(tau_high, parameters[f'tau_high_{zone}'][0])
                tau_low = error(tau_low, parameters[f'tau_low_{zone}'][0])
                yield artn_type, example, 0, zone, tau_high, tau_low, \
                    digitization_accuracy
        else:
            for trap in range(n_traps):
                tau_high, tau_low = get_taus(pred[f'trap_{trap}'], pred.index)
                tau_high = np.mean(tau_high)
                tau_low = np.mean(tau_low)

                tau_high_theory = parameters.tau_high[trap]
                tau_low_theory = parameters.tau_low[trap]

                tau_high = abs(tau_high_theory - tau_high) / tau_high_theory
                tau_low = abs(tau_low_theory - tau_low) / tau_low_theory

                digitization_accuracy = (
                    (norm(signals[f'trap_{trap}']) ==
                     norm(signals[f'pred_trap_{trap}'])).sum() / len(signals))

                yield artn_type, example, trap, '-', tau_high, tau_low, \
                    digitization_accuracy

        # continue
        # signals = signals.iloc[:100000]
        plt.close()
        fig, ax = plt.subplots(nrows=3, sharex=True)
        ax[0].plot(signals['full_signal'])
        ax[0].plot(signals['pred'])
        ax[0].set_ylabel('Raw data and\nall traps prediction')
        ax[1].plot(signals['trap_0'])
        ax[1].plot(-signals['pred_trap_0'])
        ax[1].scatter(
            signals.index,
            -signals['pred_trap_0'],
            c=(norm(signals['pred_trap_0']) == norm(signals['trap_0']))
            .apply(lambda same: 'green' if same else 'red'))
        ax[1].set_ylabel('Trap 0 prediction')
        ax[2].plot(pred['zone'], label='Predicted')
        ax[2].plot(signals['zone'], c='red', label='Truth')
        ax[2].legend()
        ax[2].set_ylabel('Zone')
        # ax[2].plot(signals['trap_1'])
        # ax[2].plot(-signals['pred_trap_1'])
        # ax[2].scatter(
        #     signals.index,
        #     -signals['pred_trap_1'],
        #     c=(norm(signals['pred_trap_1']) == norm(signals['trap_1']))
        #     .apply(lambda same: 'green' if same else 'red'))
        fig.tight_layout()
        fig.suptitle(file('').name)
        plt.get_current_fig_manager().window.showMaximized()
        plt.show()
        plt.close(fig)

        # fig, ax = plt.subplots(nrows=2, sharex=True)
        # # ax[0].set_title(basename, fontsize=8)
        # # print('full_signal', len(signals['full_signal']))
        # # print('pred', pred.shape)
        # ax[0].plot(signals['full_signal'])
        # ax[0].plot(signals['pred'])
        # # ax[0].set_xlim(0, 5000)
        # # ax[1].set_xlim(0, 5000)
        # for i in range(n_traps):
        #     ax[1].plot(pred[f'trap_{i}'], label=f'Trap {i}')
        # # ax[1].plot(signals['trap_1'], label='True')
        # # ax[1].plot(signals['pred_trap_1'], label='Pred')
        # ax[1].set_xlim(0, len(pred['sum']))
        # ax[1].legend()
        # ax[1].set_ylabel('Current ($\\mu A$)')
        # ax[0].set_ylabel('Current ($\\mu A$)')
        # ax[1].set_xlabel('Signal step')
        # fig.tight_layout()
        # TODO
        # fig.savefig(
        #     '/tmp/cnt_images/04.LBm1_Set1_1um_500nm_9K_Sampling_1.75V_5'
        #     '.28uA_10sets_Run81-Run90_predicted_time_series.png',
        #      dpi=300,
        # )
        # plt.show()
        # plt.close(fig)

        # df = pd.read_feather('/tmp/artn_correlated_example=0_generated_data_00.feather')

df = pd.DataFrame(
    generator(),
    columns=['artn_type', 'example', 'trap',
             'zone', 'tau_high_error', 'tau_low_error',
             'digitization_accuracy'])
df

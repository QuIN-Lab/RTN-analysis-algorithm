"""
Try fitting Tau histogram instead of taking the mean.
The idea was that this could be less susceptible to outliers and that it would
achieve higher accuracy.

Developed by Marcel Robitaille on 2022/04/11 Copyright © 2021 QuIN Lab
"""

import re
from pathlib import Path
from itertools import product
from multiprocessing import Pool
from glob import glob

import click
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.optimize import curve_fit, minimize
from scipy.special import factorial
from scipy import stats
from scipy.stats import poisson

from example import Example
from constants import DATA_DIR
from tau_extraction import get_taus
from utils import pairwise

plt.switch_backend('Agg')


def func(x, t):
    return np.exp(-x/t) / t


example = Example('')

def error(a, b):
    return abs(a - b) / a * 100


def tau_fitting(example):
    example = Example(example)
    print(example)
    assert example.is_time_series_predictions, example

    predictions = example.read()
    signals = Example(DATA_DIR / example.signals.path.name).read()
    params = Example(DATA_DIR / example.parameters.path.name).read() \
        .sort_values('trap')


    n_traps = len(params)
    fig, axes = plt.subplots(ncols=2, nrows=n_traps,
                             figsize=(7, 0.5 + 2 * n_traps))

    bins = 60
    for ax, (trap, (i, which)) in zip(np.array(axes).flatten(), product(
        range(n_traps),
        enumerate(['tau_high', 'tau_low']),
    )):
        design = params.iloc[trap][which]

        # Pred

        tau = get_taus(predictions[f'trap_{trap}'], predictions.index) \
                [['tau_high', 'tau_low'].index(which)]
        e = error(design, np.mean(tau))
        y, x, _ = ax.hist(tau, bins=bins, density=True, color='red',
                alpha=0.6, label=f'Predictions $({e:.1f}\,\%)$')
        x = np.array([np.mean(a) for a in pairwise(x)])
        popt, pcov = curve_fit(func, x, y, [200])
        fit,  = popt
        e = error(params.iloc[trap][which], fit)
        ax.plot(x, func(x, *popt), c='red',
                label=f'$e^{{x/{fit:.1f}}}/{fit:.1f}\ ({e:.1f}\,\%)$')

        label = re.sub(r'_(.*)', r'_\\mathrm{\1}', which.replace('tau', r'\tau'))
        ax.set_title(fr'trap = {trap}, ${label} = {params.iloc[trap][which]}$')

        # True

        true_tau = get_taus(signals[f'trap_{trap}'], signals.index) \
                [['tau_high', 'tau_low'].index(which)]
        e = error(params.iloc[trap][which], np.mean(true_tau))
        y, x, _ = ax.hist(true_tau, bins=bins, alpha=0.6,
                density=True, color='green', label=f'True $({e:.1f}\,\%)$')

        x = np.array([np.mean(a) for a in pairwise(x)])
        popt, pcov = curve_fit(func, x, y, [200])
        true_fit, = popt
        e = error(params.iloc[trap][which], true_fit)
        ax.plot(x, func(x, *popt), color='green',
                label=f'$e^{{x/{true_fit:.1f}}}/{true_fit:.1f}\ ({e:.1f}\,\%)$')
        ax.legend(fontsize=6)

        ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0),
                            useOffset=False)
        ax.set_xlabel(f'${label}$')
        ax.set_ylabel('Probability')

        yield example, trap, which, design, np.mean(tau), np.mean(true_tau), \
                fit, true_fit

    fig.suptitle(example.with_name('').path.name.replace('_', ' '))
    fig.tight_layout()
    out_dir = Path('/tmp/tau_fitting/')
    out_dir.mkdir(exist_ok=True)
    print(out_dir / example.with_name('tau_fitting.png').path.name)
    fig.savefig(out_dir / example.with_name('tau_fitting.png').path.name,
            dpi=300)
    plt.close(fig)


def wrapper(file):
    """
    Wrapper is required to avoid "generator function cannot be pickled" errors
    This wraps up everything that is yielded from the generator into a list,
    which is later flattened
    """
    return list(tau_fitting(file))


@click.command('tau-fitting')
@click.argument('files', type=click.Path(), nargs=-1, required=True)
def tau_fitting_click(files):
    """
    Plot the tau histogram and the e^(-x/tau) / tau fitting
    for all examples specified by FILES.
    """

    with Pool() as p:
        return list(p.imap_unordered(
            tau_fitting,
            files,
        ))[0]


@click.command('tau-fitting-trends')
@click.argument('files', type=click.Path(), nargs=-1, required=True)
def tau_fitting_trends(files):
    """
    Plot the trends of tau fitting.
    That is, plot many different metrics to compare them.
    """

    with Pool() as p:
        df = pd.DataFrame(chain(*p.imap_unordered(
            wrapper,
            files,
        )), columns=['example', 'trap', 'which', 'design', 'pred_mean',
                    'true_mean', 'pred_fit', 'true_fit'])

    df['pred_mean_vs_design'] = error(df['design'], df['pred_mean'])
    df['true_mean_vs_design'] = error(df['design'], df['true_mean'])
    df['pred_mean_vs_true_mean'] = error(df['true_mean'], df['pred_mean'])

    df['pred_fit_vs_design'] = error(df['design'], df['pred_fit'])
    df['true_fit_vs_design'] = error(df['design'], df['true_fit'])
    df['pred_fit_vs_true_fit'] = error(df['true_fit'], df['pred_fit'])

    sort_col = 'pred_mean_vs_true_mean'
    df = df.sort_values(sort_col)
    df.reset_index(inplace=True, drop=True)

    plt.close('all')
    fig, ax = plt.subplots()

    for column in ['pred_mean_vs_design','pred_mean_vs_true_mean', 'pred_fit_vs_design']:
        ax.plot(df[column], linewidth=0.75, label=column)

    ax.legend()
    ax.set_ylim(0, 100)
    ax.set_xlim(0, len(df))
    ax.set_xlabel(f'Example / trap index sorted by `{sort_col}`')
    ax.set_ylabel('Error %')
    fig.tight_layout()
    fig.savefig('/tmp/trends.png', dpi=300)
    plt.close(fig)


__all__ = ('tau_fitting_click',)


if __name__ == '__main__':
    tau_fitting_trends(glob('/home/marcel/OneDrive/02. QuIN_Research/31. Noise-RTN/01. 2021_Algorithm paper/rnn_results_tracking/Run-7_2021-12-02_RNN_fix_noise_truth_estimate_mismatch/raw_data_from_server/*_predicted_time_series.feather'))

"""
Main file for the second step of the algorithm.

Developed by Marcel Robitaille on 2022/03/18 Copyright © 2021 QuIN Lab
"""

import itertools
import math
from multiprocessing import Pool
from operator import itemgetter
from functools import partial

import click
import numpy as np
import pandas as pd
from colorama import Fore, Style
from timer import timer
from toolz.curried import curry
import doctool
from tqdm import tqdm

from mode import Mode
from example import Example
from utils import print_filename_on_exception
from .plot_gmm import plot_gmm
from .define_mixture_model import define_mixture_model


def is_power_of_2(x):
    if x == 0:
        return False
    return math.log2(x).is_integer()


@curry
def index_of_closest(a, b):
    return np.abs(a - b).argmin()


def rtn_levels_to_trap_amplitudes(means):
    """
    Extract the individual trap amplitudes from the list of RTN levels
    """

    n_peaks = len(means)
    base = means.pop(0)
    traps = []

    # Exclude base which is already removed
    for i in np.arange(n_peaks)[1:]:
        # If it's a power of two, we should take the next peak as this
        # trap's amplitude
        if is_power_of_2(i):
            traps.append(means.pop(0) - base)
        # Otherwise, figure out which peak to remove (which is the sum
        # of other traps)
        # Use the binary representation to determine which traps to add
        # up. For example, if i=0b110, then we should sum the amplitudes
        # of trap 1 and 2 but not 0
        else:
            target = base + sum(
                trap
                for j, trap in enumerate(traps) if i & 1 << j
            )
            means.pop(index_of_closest(np.asarray(means), target))

    return traps


def fit_and_score(indices, mode, n_traps, mean_vec, kde_data, example,
                  output_format, debug, print=print):
    dens = np.array(kde_data.density)
    bins = np.array(kde_data.intensity)

    # TODO: This assumes the missing level is one of two traps alone
    # That is, it assumes that one of the two traps is only active when the
    # other trap is active
    # This is the only case of missing level aRTN I have seen in literature,
    # but I am not 100% sure others are not possible
    if mode == Mode.MISSING_LEVEL:
        assert len(mean_vec) == 3
        mean_vec.insert(2, mean_vec[2] - mean_vec[1] + mean_vec[0])
        # TODO: It would be nice to make this a computed property on some class
        n_peaks = len(mean_vec)
        n_traps = int(np.ceil(np.log2(n_peaks)))

    print()
    print('indices', indices)
    for i in sorted(indices, reverse=True):
        mean_vec[i] += 1
        mean_vec.insert(i, mean_vec[i] - 1)

    seeds = mean_vec.copy()
    separations = np.array(rtn_levels_to_trap_amplitudes(mean_vec.copy()))
    print('separations', separations)
    normalized = np.array(mean_vec) - mean_vec[0]
    if mode == Mode.COUPLED:
        coupling_factor = dict(
            max=1.3,
            min=0.7,
            vary=True,
            value=normalized[3] / (normalized[1] + normalized[2]),
        )
    else:
        coupling_factor = dict(value=1, vary=False)
    model, parameters = define_mixture_model(
        base={'value': mean_vec[0], 'vary': True},
        # TODO: Determine seed sigma from raw signal
        sig={
            'value': 0.005 if mode == Mode.CNT_REAL_DATA else 1,
            'vary': True,
        },
        coupling_factor=coupling_factor,
        n_traps=n_traps,
        s_min=0,
        s_max=max(bins),
        separations=[{'value': x, 'vary': True} for x in separations],
        probabilities=[{'value': 0.5, 'vary': True}
                       for _ in range(n_traps)],
        mode=mode,
    )
    model.eval(parameters, x=bins)
    fit = model.fit(dens, parameters, x=bins)
    suffix = ','.join(map(str, indices))
    if debug:
        plot_gmm(
            example=example,
            fit=fit,
            n_traps=n_traps,
            suffix=f'_indices={suffix}_mode={mode.name}',
            seeds=seeds,
            output_format=output_format,
        )

    # Find difference between fitted curve and KDE curve
    # Only consider the peak locations.
    # Otherwise, the wrong permutation can get a very high score if one of
    # the Gaussians covers a lot of white noise
    centers = [
        fit.params[f'g{i:0{n_traps}b}_center'].value
        for i in range(2 ** n_traps)
    ]
    bins_indices = [
        index_of_closest(bins, x)
        for x in centers + kde_data.peaks_intensities.tolist()
    ]
    height_differences = np.array([
        dens[i] - fit.best_fit[i]
        for i in bins_indices
    ])
    mse = (height_differences ** 2).sum()

    # If one of the gaussians is way way off (so far off that the KDE value
    # there is 0), make the error infinity
    # Basically force it to not use this permutation
    if (dens[bins_indices] == 0).any() and mode != Mode.MISSING_LEVEL:
        mse = np.inf

    return fit, indices, mode, n_traps, mse


@print_filename_on_exception
def decomposition(example, mode, output_format, debug, alone):
    # if '3-trap_example=1' in example or '3-trap_example=4' in example:
    #     return
    # print()
    # if example.sep_error.path.exists():
    #     return

    # discretization_data = pd.Series(example.discretization_data.read())
    example = Example(example)
    mode = Mode.from_str(mode=mode, filename=example.path)
    print('decomposition', example.path.name, 'mode', mode)

    kde_data = example.kde_data.read().squeeze()

    # We don't have labels for real data
    parameters = None if mode == Mode.CNT_REAL_DATA \
        else  example.parameters.read()

    mean_vec = sorted(kde_data.peaks_intensities.tolist())
    dens = np.array(kde_data.density)
    bins = np.array(kde_data.intensity)

    first_non_zero = bins[(dens > dens.max() / 20).tolist().index(True)]
    print('first non zero', first_non_zero)
    print('first peak to first non zero', mean_vec[0] - first_non_zero)
    n_peaks = len(mean_vec)
    n_traps = int(np.ceil(np.log2(n_peaks)))

    # CNT real data has some examples with background trends/drifts
    # These are falsely detected as >3 traps, which should be skipped
    # We should not limit our algorithm to <=3 traps, but these data have been
    # manually checked
    if mode == Mode.CNT_REAL_DATA and n_traps > 3:
        print(Fore.RED, 'Found more than 3 traps!!!!! Skipping',
              Style.RESET_ALL)
        return

    if mode in {Mode.MUTUALLY_EXCLUSIVE, Mode.CNT_REAL_DATA}:
        print(mean_vec)
        assert len(mean_vec) == 3
        mean_vec.append(mean_vec[2] + mean_vec[1] - mean_vec[0])
        # TODO: It would be nice to make this a computed property on some class
        n_peaks = len(mean_vec)
        n_traps = int(np.ceil(np.log2(n_peaks)))

    if mean_vec[0] - first_non_zero > 34 and n_peaks < 2 ** n_traps:
        print('inserting first non zero')
        mean_vec.insert(0, first_non_zero)
        # TODO: It would be nice to make this a computed property on some class
        n_peaks = len(mean_vec)
        n_traps = int(np.ceil(np.log2(n_peaks)))

    # We cannot perform this wrongness test for coupled
    # Coupled aRTN would by definition have very high wrongness
    if mode != Mode.COUPLED and n_peaks == 4 and parameters is not None:
        # Wrongness is a measure of how different the detected peaks are from
        # theory.
        # The sum of multiple peaks is compared to higher level peaks
        # This is used to detect a hidden extra trap
        wrongness = abs(mean_vec[1] + mean_vec[2] - mean_vec[3] - mean_vec[0])
        wrongness /= max(mean_vec) - min(mean_vec)
        print('wrongness', wrongness)
        if wrongness > 0.03:
            if len(parameters) == 2:
                print(f'{Fore.RED}WARNING: Assuming 3 traps but only 2!'
                      f'{Style.RESET_ALL}{example.path.name} {wrongness}')
            print(f'{Fore.RED}bumping{Style.RESET_ALL}', example.path.name)
            n_traps += 1

    print('mean_vec', mean_vec)

    # Cannot do anything if the KDE only finds one peak
    # Common for 1/f noise
    if len(mean_vec) < 2:
        example.decomp_data_traps.write(pd.DataFrame(columns=['trap', 'sep']))
        try:
            example.gmm_fit.path.unlink()
        except FileNotFoundError:
            pass
        example.with_name('timer_decompose.txt').write(str(0))
        return None

    with timer() as t:
        kwargs = dict(
            mode=mode,
            n_traps=n_traps,
            mean_vec=mean_vec.copy(),
            kde_data=kde_data,
            example=example,
            output_format=output_format,
            debug=debug,
        )
        combinations = list(itertools.combinations_with_replacement(
            np.arange(n_peaks),
            r=2 ** n_traps - n_peaks,
        ))
        if alone:
            with Pool() as p:
                possibilities = list(tqdm(p.imap_unordered(
                    partial(
                        fit_and_score,
                        **kwargs,
                        # print=tqdm.write,
                        print=print,
                    ),
                    combinations,
                ), total=len(combinations)))
        else:
            possibilities = [fit_and_score(
                indices=indices,
                **kwargs,
            ) for indices in combinations]

        fit, indices, mode, n_traps, _ = min(possibilities, key=itemgetter(4))

    print('winner', indices)
    amplitudes = [fit.params[f'sep_trap{i}'].value for i in range(n_traps)]
    if mode == Mode.COUPLED:
        amplitudes.append((amplitudes[0] + amplitudes[1]) *
                          fit.params['coupling_factor'].value)
    example.decomp_data_traps.write(pd.DataFrame({
        'trap': i,
        'sep': amp,
    } for i, amp in enumerate(amplitudes)))

    example.gmm_fit.write(fit)

    example.with_name('timer_decompose.txt').write(str(t.elapse))

    return plot_gmm(
        example=example,
        fit=fit,
        n_traps=n_traps,
        seeds=mean_vec.copy(),
        output_format=output_format,
    )


@click.command('process-gmm')
@click.argument('files', type=click.Path(), required=True, nargs=-1)
@click.option(
    '--mode', '-m', type=click.Choice(Mode.strings), required=True,
    default='auto',
    help='Algorithm mode. If `auto`, detect automatically from filename.')
@click.option('--output-format', type=click.Choice(['pdf', 'png']),
              default='pdf')
@click.option('--debug/--no-debug', is_flag=True, default=False)
@doctool.example(
    help='To process the GMM and generate the GMM plot, run the command:',
    args=[
        DATA_DIR / '1-trap_wn=0.4_example=0_signals.feather',
        '--output-format=png',
    ],
    creates_image=True,
)
def process_gmm_click(files, mode, output_format, debug):
    """
    Run GMM on all examples specified by FILES (specify `_signals.feather`
    files).
    """

    alone = len(files) == 1 and False

    with Pool() as p:
        consume = map if alone else p.imap_unordered
        return list(consume(
            partial(decomposition, mode=mode, output_format=output_format,
                    debug=debug, alone=alone),
            files,
        ))[0]


__all__ = ('process_gmm_click', )

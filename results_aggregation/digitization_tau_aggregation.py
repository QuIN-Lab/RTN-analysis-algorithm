import re
from typing import cast
from pathlib import Path
from multiprocessing import Pool
from itertools import permutations
from operator import eq, ne
from functools import lru_cache, partial

import click
import pandas as pd

from example import Example
from constants import console
from utils import run_click, error
from tau_extraction import get_taus_mean

# %%

def parse_filename(filename):
    if m := re.search(r'(\d)-trap_wn=([0-9\.]+)_example=(\d+)', filename):
        n_traps, noise, example_number = \
                (f(x) for x, f in zip(m.groups(), (int, float, int)))
        return 'normal', n_traps, noise, example_number

    if m := re.search(r'artn_(metastable|coupled|missing_level)_example=(\d+)',
                      filename):
        artn_type, example_number = m.groups()
        n_traps = 1 if artn_type == 'metastable' else 2
        # TODO: This may not always be true, but then it should be in
        # the filename, in which case this regex would not match
        noise = 0.2
        return str(artn_type), n_traps, noise, int(example_number)

    raise NotImplementedError(
        f'Filename did not match any known pattern {filename}')

def generator(example, filter_size):
    example = Example[pd.DataFrame](example)
    console.print(example)
    assert example.is_time_series_predictions(), \
        'Please specify `_time_series_predictions.feather` files. ' \
        f'Received `{example}`'

    rtn_type, _true_n_traps, noise, example_number = \
        parse_filename(str(example.path))

    pred = example.read()
    pred = pred if filter_size == 0 else cast(
        pd.DataFrame,
        pred.rolling(window=filter_size, center=True).mean().round())
    signals = example.signals.read()
    n_traps = len(pred.columns)

    # Metastable case
    if 'trap_0' not in signals.columns and 'zone' in signals.columns:
        signals['trap_0'] = signals['rtn_sum']
        signals.drop(columns=['zone', 'a', 'b'], inplace=True)

    # Remove all extra columns besides `trap_x`
    signals = signals[[c for c in signals.columns if c.startswith('trap_')]]
    true_n_traps = len(signals.columns)
    signals = cast(pd.DataFrame, (signals > 0).astype(int))
    pred = cast(pd.DataFrame, pred.rename('pred_{}'.format, axis='columns'))

    # Shift prediction index by 63
    # to account for the step size
    if len(pred) < len(signals):
        console.print(
            '[WARNING] Detected that predictions are shorter than signals. '
            'Assuming batched recurrent data. '
            'Shifting predictions by 63 (STEPS - 1).',
            style='yellow',
        )
        pred.index += 63
    df = cast(pd.DataFrame, pd.concat((signals, pred), axis=1).dropna())

    @lru_cache
    def digitization_error(pred_trap, true_trap, operator):
        return operator(
            df[f'pred_trap_{pred_trap}'],
            df[f'trap_{true_trap}'],
        ).sum() / len(signals)

    # Match the predicted traps with the true traps
    # The algorithm does not necessarily output the traps in the same order as
    # our design values
    options = [
        # We want to return a list of the accuracy for each predicted trap
        # so here we should loop through the predicted n_traps
        [dict(
            rtn_type=rtn_type,
            n_traps=true_n_traps,
            noise=noise,
            example=example_number,
            true_trap=true_trap,
            pred_trap=pred_trap,
            is_flipped=is_flipped,
            digitization_error=digitization_error(
                pred_trap=pred_trap,
                true_trap=true_trap,
                operator=(eq if is_flipped else ne),
            ))
         for pred_trap, true_trap in zip(range(n_traps), true_traps)]
        for is_flipped in (False, )
        # We try each permutation of the true traps
        # and take the minimum
        for true_traps in permutations(range(true_n_traps))
    ]
    results = pd.DataFrame(min(
        options,
        key=lambda rows: sum(row['digitization_error'] for row in rows),
    ))

    def calculate_tau_errors(row):
        true_trap = row['true_trap']
        pred_trap = row['pred_trap']
        pred_signal = df[f'pred_trap_{pred_trap}']
        true_signal = df[f'trap_{true_trap}']

        pred_tau_high, pred_tau_low = get_taus_mean(pred_signal, df.index)
        true_tau_high, true_tau_low = get_taus_mean(true_signal, df.index)

        return error(pred=pred_tau_high, truth=true_tau_high), \
            error(pred=pred_tau_low, truth=true_tau_low)

    results[['tau_high_error', 'tau_low_error']] = \
        results[['true_trap', 'pred_trap']].apply(
            calculate_tau_errors,
            axis=1,
            result_type='expand',
        )

    return results


@click.command()
@click.argument('files', nargs=-1, required=True, type=click.Path())
@click.argument('output', required=True, type=click.Path())
@click.option(
    '--filter', '-f', 'filter_size', type=int, default=18,
    help='Size of the filter to use on the digitized raw output. '
    'This step used to be done in the RNN code, '
    'but it\'s better to do it here. We used to use 18. '
    'Use `0` to disable.',
)
def aggregate_digitization_tau_error(files, output, filter_size):
    """
    Calculate the digitization and tau error for a set of time-series
    predictions specified by FILES (files must end in
    `_time_series_predictions.feather`). Save the resulting dataframe to OUTPUT
    (supports `.csv`, `.feather`).
    The reason to combine these two like this is that they are both calculated
    from the time-series predictions of the digitization step
    and to facilitate some housekeeping.
    """
    # The housekeeping mentionned in the docstring
    # is the trap matching.
    # The algorithm does not necessarily output the traps in the same order as
    # our design values, so trap-matching is required to calculated digitization
    # and tau error based on the correct trap's design parameters.
    # It is better to do this only once and output the results together.
    # The alternative would be to have a separate command for the tau
    # aggregation that has to calculate the digitization error for each
    # permutation of traps, take the minimum as the correct trap mapping,
    # and calcualte the tau error based on that mapping.

    output = Example(output)
    assert not output.path.is_dir()
    assert output.path.suffix in {'.csv', '.feather'}

    # Prevent shooting yourself in the foot
    # Since FILES takes a variable number of arguments, if you forget to specify
    # OUTPUT, the last file you intended for FILES will be overwritten with the
    # results
    assert not output.is_time_series_predictions(), \
            'Maybe you forgot to specify OUTPUT?'

    print(len(files))
    with Pool() as p:
        df = pd.concat(p.imap_unordered(
            partial(generator, filter_size=filter_size),
            files,
        ))

    df.rename(columns={'true_trap': 'trap'}, inplace=True)
    df.sort_values(['rtn_type', 'n_traps', 'noise', 'example', 'trap'],
                   inplace=True)
    output.write(df)


if __name__ == '__main__':
    data_dir = Path.home() / 'OneDrive/02. QuIN_Research/31. Noise-RTN' / \
        '01. 2021_Algorithm paper/rnn_results_tracking' / \
        'Run-9_2022_04_22_RNN_LSTM/'

    run_click(
        aggregate_digitization_tau_error,
        *(data_dir / 'raw_data_from_server').glob('*_predictions.feather'),
        data_dir / 'digitization_error.csv',
    )


__all__ = ('aggregate_digitization_tau_error',)

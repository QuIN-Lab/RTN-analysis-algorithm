from multiprocessing import Pool
from pathlib import Path
from functools import partial

import click
import pandas as pd
from timer import timer
from hmmlearn.hmm import GaussianHMM

from example import Example
from utils import consume


def run_hmm(example: Example, results_dir: Path):
    example = Example(example)
    assert example.path.name.endswith('_signals.feather'), example
    print(example)

    signals = example.read()

    full_signal = signals.full_signal.to_numpy().reshape(-1, 1)

    with timer() as t:
        model = GaussianHMM(n_components=2, n_iter=100, algorithm='viterbi')
        model.fit(full_signal)
        pred = model.predict(full_signal)
    pred = pd.DataFrame(pred, columns=['trap_0'])
    result = example.with_parent(results_dir)
    result.time_series_predictions.write(pred)
    result.with_name('hmm_timer.txt').write(str(t.elapse))


@click.command('hmm')
@click.argument('files', required=True, type=click.Path(), nargs=-1)
@click.argument('results-dir', required=True, type=click.Path())
def hmm_click(files, results_dir):
    """
    Predict the underlying RTN component using HMM
    for all signals specified by FILES (provide `_signals.feather` files).
    Write the time-series predictions to RESULTS_DIR.
    This uses HMM, not fHMM, so it only really works for 1-trap data.
    """

    results_dir = Path(results_dir)
    for f in files:
        run_hmm(f, results_dir=results_dir)
    # with Pool() as p:
    #     consume(p.imap_unordered(
    #         partial(run_hmm, results_dir=results_dir),
    #         files,
    #     ))


__all__ = ('hmm_click',)

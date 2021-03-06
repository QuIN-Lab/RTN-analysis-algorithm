import re
from pathlib import Path

import click
import doctool
import numpy as np
import pandas as pd
import numpy as np

from mode import Mode
from utils import auto_n_traps
from constants import console, DATA_DIR, CNT_DATA_DIR
from example import Example
from normalize_reformat import reformat_real_data
from .main import train_and_score_worker, make_predictions, \
    add_model_layer_type_options, ModelType, DefineModelProtocol


@click.command()
@click.option(
    '--mode', '-m', 'mode_str', required=True, type=click.Choice(Mode.strings),
    default='auto',
    help='Algorithm mode. If `auto`, detect automatically from filename.')
@click.option('--retrain/--no-retrain', required=True, is_flag=True,
              help='Wether to retrain or to load weights from the saved .h5 file')
@click.argument('files', required=True, type=click.Path(), nargs=-1)
@click.argument('results-dir', required=True, type=click.Path())
@click.option('--n-traps', '-n', type=click.Choice(['1', '2', '3', 'auto']),
              default='auto', required=True,
              help='The number of traps. If `auto`, determine from filename.'
              'Only required for simulated data to calculate accuracy.')
@click.option('--skip-existing/--no-skip-existing', is_flag=True, default=False,
              help='Skip training if the time_series_predictions file exists.')
@doctool.example(
    help='The RNN is run in a similar way, although there are many more options (like GRU vs. LSTM).',
    args=[DATA_DIR / '1-trap_wn=0.4_example=0_signals.feather',],
)
@doctool.example(
    help='The mode must be guessed when running on real measurements.',
    args=['--mode=missing-level',
          CNT_DATA_DIR / '01.LBm1_Set1_1um_500nm_3.7K_Sampling_2.0V_6.1uA_9sets_Run101-Run109_signals.feather',],
)
@add_model_layer_type_options
def train(files, results_dir, mode_str, retrain, n_traps,
          define_model: DefineModelProtocol, model_type: ModelType,
          recurrent_layer: str, skip_existing: bool):
    """
    Train the RNN model for the examples specified by FILES (specify
    `_signals.feather` files).
    Save the results (trained model, full predictions, accuracies) in
    RESULTS_DIR.

    For simulated signals, the default options should be sufficient.
    All required information will be extracted from the filename.

    For real measurements, please specify `--mode` based on
    your assessment of the aRTN case.

    This command has replaced the previous `train-artn` and `train-ctn` commands
    (cc7af0e1).
    """

    # Don't use multiprocessing pool like I do for many of the step 1/2 commands
    # Training one model is already maxing out the GPU

    console.print('[bold red]Starting Training:[/bold red] '
                  f'retrain={retrain}, model_type={model_type}, '
                  f'recurrent_layer={recurrent_layer}')
    for i, filename in enumerate(sorted(files)):
        console.print()
        console.print()
        console.print()
        console.print(f'Training file {i} of {len(files)}', style='red')

        print(filename)
        # m = re.search(r'wn=([0-9\.]+)', filename)
        # noise = float(m.group(1))
        # if noise > 0.2:
        #     console.log('[WARNING] Skipping file with noise > 20%')
        #     continue
        mode = Mode.from_str(mode=mode_str, filename=filename)
        train_and_score_worker(
            filename=filename,
            mode=mode,
            results_dir=Path(results_dir),
            retrain=retrain,
            n_traps=n_traps,
            define_model=define_model,
            model_type=model_type,
            skip_existing=skip_existing,
        )


@click.command()
@click.option(
    '--mode', '-m', 'mode_str', required=True, type=click.Choice(Mode.strings),
    default='auto',
    help='Algorithm mode. If `auto`, detect automatically from filename.')
@click.option('--n-traps', '-n', type=click.Choice(['1', '2', '3', 'auto']),
              default='auto', required=True,
              help='The number of traps. If `auto`, determine from filename.')
@click.argument('files', required=True, type=click.Path(), nargs=-1)
@click.argument('results-dir', required=True, type=click.Path())
@add_model_layer_type_options
def output_time_series(results_dir, mode_str, n_traps, files,
                       define_model: DefineModelProtocol,
                       model_type: ModelType):
    """
    Output time series data using pre-trained models
    for the examples specified by FILES (specify `_signals.feather` files).
    Save the results in RESULTS_DIR.
    """

    for i, filename in enumerate(files):
        print(f'Processing file {i} of {len(files)}')

        mode = Mode.from_str(mode=mode_str, filename=filename)
        n_traps = auto_n_traps(n_traps=n_traps, mode=mode, filename=filename)

        assert filename.endswith('_signals.feather'), filename

        example = Example(filename)
        signals = example.read()

        model = define_model(n_traps=n_traps)
        model.load_weights(results_dir / example.tf_model.path.name)

        feature_t, _label_t = reformat_real_data(signals, detected_traps=n_traps)
        pred = make_predictions(model=model, feature_t=feature_t,
                                model_type=model_type)

        # Not necessarily 'trap_0', 'trap_1' in that order!
        columns = [c for c in signals.columns if re.fullmatch(r'trap_\d', c)] \
            [:np.shape(pred)[0]]
        print('np.shape(pred)', np.shape(pred))
        if len(columns) == 0:
            columns = [f'trap_{i}' for i in range(np.shape(pred)[1])]
        console.print(type(pred))
        console.print(np.shape(pred))
        pred = pd.DataFrame(pred, columns=columns)
        pred.to_feather(results_dir / example.time_series_predictions.path.name)


__all__ = ('train', 'output_time_series')

"""
Main file for the third step of our algorithm.

Developed by Marcel Robitaille on 2022/03/18 Copyright © 2021 QuIN Lab
"""

from pathlib import Path
from itertools import product
from functools import partial, wraps
from typing import Union, Optional, Protocol, cast
from enum import Enum, auto

import click
import numpy as np
import pandas as pd
import tensorflow as tf
from timer import timer
from tensorflow import keras
from tensorflow.keras.layers import GRU, LSTM, Bidirectional, Dense, Input, \
    Lambda, Conv1D, Multiply, Add, Activation
from sklearn.model_selection import train_test_split

from mode import Mode
from normalize_reformat import reformat_real_data, reformat_generated_data
from constants import STEPS, BATCH_SIZE, console
from white_noise_estimate import white_noise_estimate
from utils import has_labels
from example import Example
from .generate_training_data import generate_data


# Constants

# Each rtn has 2 states
CLASSES = 2
FEATURES = 1


class ModelType(Enum):
    WAVENET = auto()
    RNN = auto()


def reverse(x):
    reverse_aim = x.shape[1]
    repet_num = x.shape[0]
    seq_lenths = [reverse_aim] * repet_num

    x = tf.reverse_sequence(x, seq_lenths, seq_axis=1, batch_axis=0)
    return x


def multi_output_rnn(x, n_traps, filters):
    return tf.reshape(x, [BATCH_SIZE, n_traps, CLASSES * filters])


class DefineModelProtocol(Protocol):
    def __call__(self, n_traps: int) -> keras.Model: ...


def define_model_rnn(n_traps: int, filters: int = 128,
                     RecurrentLayer: Union[GRU, LSTM]=GRU):
    """
    Define the inputs and outputs for the RNN model
    """

    inputs = Input(
        [STEPS, FEATURES],
        batch_size=BATCH_SIZE,
        name='inputs',
    )
    x = Bidirectional(RecurrentLayer(filters, return_sequences=True))(inputs)
    x = Bidirectional(RecurrentLayer(filters, return_sequences=True))(x)
    x = Bidirectional(RecurrentLayer(filters, return_sequences=False))(x)

    out = Dense(n_traps * CLASSES * filters, activation='relu')(x)
    out = Lambda(
        multi_output_rnn,
        arguments=dict(n_traps=n_traps, filters=filters),
    )(out)
    outputs = Dense(
        2,
        activation='softmax',
        name='outputs',
    )(out)

    model = keras.Model(inputs=inputs, outputs=outputs)

    opt = tf.keras.optimizers.Adam(learning_rate=0.00001)
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy'],
    )

    return model


def multi_output_wavenet(x, n_traps, filters):
    return tf.reshape(x, [-1, STEPS, n_traps, CLASSES * filters])


def define_model_wavenet(
    n_traps: int,
    filters: int = 32,
    nb_stacks: int = 1,
    dilation_depth: int = 6,
    use_bias: bool = False,
):
    """
    Define the inputs/outputs/structure of the wavenet model.

    Adapted from:
    https://git.uwaterloo.ca/QuINLab/Projects/Noise/rtn-lu-wavenet/-/blob/main/wavenet_aps_3.py
    """

    def residual_block(x, current_dilation_depth):
        original_x = x
        tanh_out = Conv1D(
            filters=filters,
            kernel_size=2,
            padding='causal',
            dilation_rate=2 ** current_dilation_depth,
            use_bias=use_bias,
            activation='tanh',
        )(x)

        sigm_out = Conv1D(
            filters=filters,
            kernel_size=2,
            padding='causal',
            dilation_rate=2 ** current_dilation_depth,
            use_bias=use_bias,
            activation='sigmoid',
        )(x)

        x = Multiply()([tanh_out, sigm_out])

        res_x = Conv1D(
            filters=filters,
            kernel_size=1,
            use_bias=use_bias,
            padding='same',
        )(x)

        skip_x = Conv1D(
            filters=filters,
            kernel_size=1,
            use_bias=use_bias,
            padding='same',
        )(x)
        res_x = Add()([original_x, res_x])

        return res_x, skip_x

    input_f = keras.Input(
        shape=(STEPS, FEATURES),
        batch_size=None,
        name='inputs',
    )
    out_f = input_f
    out_f = Conv1D(
        filters=filters,
        kernel_size=2,
        padding='causal',
        use_bias=use_bias,
        dilation_rate=1,
    )(out_f)

    out = Add()([
        residual_block(out_f, current_dilation_depth=current_dilation_depth)[1]
        for current_dilation_depth in range(0, dilation_depth + 1)
        for _ in range(nb_stacks)
    ])

    out = Activation('relu')(out)
    out = Conv1D(
        128,
        kernel_size=1,
        activation='relu',
        padding='same',
        use_bias=use_bias,
    )(out)

    out = Dense(n_traps * CLASSES * filters, activation='relu')(out)
    out = Lambda(
        multi_output_wavenet,
        arguments=dict(n_traps=n_traps, filters=filters),
    )(out)
    out = Dense(2, activation='softmax', name='outputs')(out)
    model = keras.Model([input_f], out)

    opt = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(
        loss='sparse_categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy'],
    )

    return model


def make_predictions(model, feature_t, model_type: ModelType, *args, **kwargs):
    pred = model.predict(feature_t, batch_size=BATCH_SIZE)

    # For some reason, the shape is different for wavenet
    # I'm sorry, I do not know why
    pred = {
        ModelType.RNN: pred[:, :, :],
        ModelType.WAVENET: pred[:, -1, :],
    }[model_type]
    pred = np.argmax(pred, axis=-1)

    return pred


def map_files(f, *args, **kwargs):
    parameters_list = Example[pd.DataFrame]('./update_parameters.csv').read()
    for true_n_traps, true_noise_level, example in product(
        np.arange(3) + 1,
        np.arange(0, 1.1, 0.1),
        range(10),
    ):
        console.print(
            f'./data/{true_n_traps}-trap_wn={true_noise_level:.1f}_'
            f'example={example}_signals.feather',
            style='bright')
        filename = f'./data/{true_n_traps}-trap_wn={true_noise_level:.1f}_' \
            f'example={example}_signals.feather'
        example = Example[pd.DataFrame](filename)
        signals = example.read()
        seed_data = parameters_list[
            (parameters_list['noise'].apply('{:.1f}'.format) ==
             f'{true_noise_level:.1f}') &
            (parameters_list['example'] == example) &
            (parameters_list['n_traps'] == true_n_traps)
        ]
        predicted_n_traps = len(seed_data.dropna())
        console.print('Predicted n_traps', predicted_n_traps)
        f(
            true_noise_level=true_noise_level,
            example=example,
            signals=signals,
            n_traps=predicted_n_traps,
            filename=filename,
            seed_data=seed_data,
            *args,
            **kwargs,
        )


def map_pretrained(f, results_dir, define_model):
    """
    Like map_files, but load the pretrained model and add it as an
    argument
    """

    def wrapper(true_noise_level, example, n_traps, *args, **kwargs):
        model = define_model(n_traps=n_traps)
        model.load_weights(
            results_dir /
            f'model_noise_{true_noise_level:.1f}_example_{example}.h5')

        return f(
            true_noise_level=true_noise_level,
            example=example,
            model=model,
            n_traps=n_traps,
            *args,
            **kwargs,
        )

    return map_files(f=wrapper)


def train_and_score_worker(
    filename: str,
    mode: Mode,
    results_dir: Path,
    retrain: bool,
    define_model: DefineModelProtocol,
    model_type: ModelType,
    skip_existing: bool,
    *args,
    seed_data: Optional[pd.DataFrame]=None,
    signals: Optional[pd.DataFrame]=None,
    **kwargs
):
    """
    Main function

    Train an RNN model for a single RTN example and score the predictions
    """

    # Read data
    results_dir.mkdir(exist_ok=True, parents=True)

    example = Example[pd.DataFrame](filename)
    result = example.with_parent(results_dir)
    assert example.path.name.endswith('_signals.feather'), example

    if skip_existing and result.time_series_predictions.path.exists():
        console.print(f'[WARNING] Skipping existing file {example}',
                      style='bold red')
        return

    console.print(filename, style='bright')
    signals = cast(pd.DataFrame, example.read() if signals is None else signals)

    seed_data = cast(
        pd.DataFrame,
        example.decomp_data_traps.read() if seed_data is None else seed_data,
    )

    # Remove DC offset
    # TODO: Use GMM smallest Gaussian center, which should be more precise
    # This requires to save that information to disk
    signals['full_signal'] -= \
        min(example.kde_data.read().squeeze().peaks_intensities)

    console.print(mode)
    if mode == Mode.METASTABLE:
        assert signals.columns[:3].to_list() == ['zone', 'a', 'b']
        signals['trap_0'] = signals['rtn_sum']
        noise_column = next(c for c in ('white_noise', 'pink_noise')
                            if c in signals.columns)
        signals = signals[['trap_0', 'rtn_sum', noise_column, 'full_signal']]
        signals = cast(pd.DataFrame, signals)

    # No labels for real data
    ground_truth_parameters = example.parameters.read() if has_labels(signals) \
        else None
    console.print(ground_truth_parameters)
    console.print('Real amplitudes:', ground_truth_parameters, style='red')

    amp_key = 'guess' if 'guess' in seed_data.columns else 'sep'
    amp_guesses = seed_data.sort_values(by='trap')[amp_key].dropna().to_list()  # type: ignore

    if not amp_guesses:
        return

    n_traps = len(amp_guesses)

    console.print('Guess amplitudes:', amp_guesses, style='red')

    noise_estimate = white_noise_estimate(signals['full_signal']) \
        / min(amp_guesses)

    model = define_model(n_traps=n_traps)
    model.summary()

    # If given retrain flag (the default), generate training data and train the
    # model
    # Otherwise, load the existing weigths (useful to reproduce some data from
    # the saved .h5)
    if retrain:
        # generate data
        with timer() as t:
            generated_dfs = generate_data(
                noise=noise_estimate,
                amplitudes=amp_guesses,
                mode=Mode.MUTUALLY_EXCLUSIVE if mode == Mode.CNT_REAL_DATA else mode,
                filename_template=
                result.with_name('generated_data_{:02d}.feather').path,
            )
            print(amp_guesses)
            print(n_traps)
            print(generated_dfs[0])
        result.with_name('timer_rnn_generate.txt').write(str(t.elapse))

        # train and test model with generated data:
        console.print('generated data')

        with timer() as t:
            # read data
            feature_g, label_g = reformat_generated_data(
                n_traps=n_traps,
                generated_dfs=generated_dfs,
            )
            assert feature_g.max() == 1, feature_g.max()
            feature_g = feature_g[:, :, :]

            # For some reason, the shape is different for wavenet
            # I'm sorry, I do not know why
            label_g = {
                ModelType.RNN: label_g[:, -1, :],
                ModelType.WAVENET: label_g[:, :, :],
            }[model_type]

            X_train, X_val, y_train, y_val = \
                train_test_split(feature_g, label_g, test_size=0.2)

            history = model.fit(
                {'inputs': X_train},
                {'outputs': y_train},
                batch_size=BATCH_SIZE,
                validation_data=({'inputs': X_val}, {'outputs': y_val}),
                validation_batch_size=BATCH_SIZE,
                epochs=20,
            )
        result.with_name('timer_rnn_train.txt').write(str(t.elapse))

        np.save(result.with_name('history.npy').path, np.array(history.history))
        model.save(result.with_name('model.h5').path)
    else:
        model.load_weights(result.with_name('model.h5').path)

    # Load truth data
    feature_t, _label_t = reformat_real_data(
        signals=signals,
        detected_traps=sorted(seed_data.dropna()['trap']),  # type: ignore
    )

    # read label, we only care the final time step output, therefore,
    # we choose final time step truth value as label.
    with timer() as t:
        pred = make_predictions(model=model, feature_t=feature_t,
                                model_type=model_type)
    result.with_name('timer_rnn_predict.txt').write(str(t.elapse))

    # Save time-series predictions
    df = pd.DataFrame(pred, columns=[f'trap_{i}' for i in range(n_traps)])
    df.to_feather(result.time_series_predictions.path)


def add_model_layer_type_options(f):
    """
    Add click options for the model type (RNN vs. Wavenet)
    and layer type (GRU vs. LSTM), create the `define_model` function
    based on the model type and add additional options using `functools.partial`
    if appropriate.
    """

    @wraps(f)
    @click.option('--recurrent-layer', '-l', type=click.Choice(['gru', 'lstm']),
                  required=True, default='lstm',
                  help='Type of recurrent layer to use (RNN only)')
    @click.option('--model-type', 'model_type_str',
                  type=click.Choice(['rnn', 'wavenet']),
                  required=True, default='rnn',
                  help='Model structure to use')
    def wrapper(recurrent_layer, model_type_str: str, *args, **kwargs):
        RecurrentLayer = {
            'gru': GRU,
            'lstm': LSTM,
        }[recurrent_layer]

        model_type: ModelType = getattr(ModelType, model_type_str.upper())
        define_model = {
            ModelType.WAVENET: define_model_wavenet,
            ModelType.RNN: partial(define_model_rnn, RecurrentLayer=RecurrentLayer),
        }[model_type]

        return f(define_model=define_model, model_type=model_type,
                 recurrent_layer=recurrent_layer, *args, **kwargs)

    return wrapper

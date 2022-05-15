"""
Main file for the first step of the algorith. Process the KDE, apply the rolling
filter, etc.

Developed by Marcel Robitaille on 2022/02/18 Copyright © 2021 QuIN Lab
"""

import re
import math
from multiprocessing import Pool
from functools import partial

import click
import pandas as pd
import numpy as np
from scipy import stats
from scipy.signal import find_peaks
from timer import timer
import doctool

from example import Example
from mode import Mode
from utils import consume
from white_noise_estimate import white_noise_estimate
from .filter_peak import filter_peak


def process_kde(example, mode, input_column='full_signal'):
    example = Example[pd.DataFrame](example)
    assert example.path.stem.endswith('_signals'), \
        'Expected the feather file to end with `_signals.feather` ' \
        f'but received `{example.path.name}. ' \
        'This check is to ensure that the correct file is provided. ' \
        'Please rename the file or ensure that you are using the correct file.'

    mode = Mode.from_str(mode=mode, filename=example.path)

    df = example.read()

    # Manually cut a measurement problem with real data
    # There is some kind of background trend or drift
    if example.path.name == \
            '04.LBm1_Set1_1um_500nm_9K_Sampling_1.75V_5.28uA_10sets_' \
            'Run81-Run90_signals.feather':
        df = df[df.time < 575]
    if example.path.name == \
            '01.LBm1_Set1_1um_500nm_3.7K_Sampling_2.0V_6.1uA_9sets_' \
            'Run101-Run109_time_series_predictions.feather':
        df = df[df.time < 494]

    wn_est = white_noise_estimate(df[input_column])
    print('wn_est', wn_est)
    transformed_wn_estimate = math.atan(wn_est / 4) / (math.pi / 2)
    window = int(transformed_wn_estimate * 30)
    if mode == Mode.CNT_REAL_DATA:
        window = 0
    print('kde', example.path.name, f'window={window}')

    def compute_kde(intensity, bw_method=0.02):
        kernel = stats.gaussian_kde(
            df['signal_filtered_kde'].dropna(),
            bw_method=bw_method,
        )
        density = kernel(intensity)

        prominence = max(density) / (140 if mode == Mode.CNT_REAL_DATA else 300)
        peaks, properties = find_peaks(
            pd.Series(density).rolling(window=10, center=True).mean(),
            # Put an extremely small number because if no value is provided,
            # the properties dictionary does not contain these keys
            height=0.00000000001,
            width=0.000000000001,
            prominence=prominence,
            distance=10,
        )

        return density, intensity, peaks, properties

    with timer() as t:
        df['signal_filtered_kde'] = df[input_column] if window == 0 \
            else df[input_column].rolling(window=window).mean()

        min_intensity = min(df[input_column])
        max_intensity = max(df[input_column])
        delta_intensity = max_intensity - min_intensity
        intensity = np.linspace(
            min_intensity - delta_intensity,
            max_intensity + delta_intensity,
            2000,
        )

        kernel = stats.gaussian_kde(
            df[input_column],
            bw_method=0.020,
        )
        raw_density = kernel(intensity)

        density, intensity, peaks, properties = compute_kde(
            intensity=intensity.copy(),
        )

        peaks_df = pd.DataFrame(properties)
        peaks_df['intensity'] = peaks
        print(peaks_df)
        peaks_df.rename(
            mapper=lambda x: re.sub(r's$', '', str(x).replace('peak_', '')),
            inplace=True,
            axis=1,
        )
        print('before filtering', len(peaks_df))
        peaks_df = peaks_df[peaks_df.apply(
            filter_peak,
            density=density,
            raw_density=raw_density,
            axis=1,
        )]
        print('after filtering\n', peaks_df)
        print(len(peaks_df))
        peaks = peaks_df['intensity']

        # If one less than 2 traps or one or two less than 3 traps
        if len(peaks) in {3, 6, 7}:
            print('looking for extra peaks')
            other_peaks, properties = find_peaks(
                pd.Series(density).rolling(window=10, center=True).mean(),
                prominence=[
                    max(density) / 2500,
                    peaks_df['prominence'].min() - 1e-6,
                ],
                height=0.00000000001,
                width=0.000000000001,
                threshold=0.000001,
                distance=10,
            )
            peaks_df = pd.DataFrame(properties)
            peaks_df['intensity'] = other_peaks
            print(peaks_df)
            peaks_df.rename(
                mapper=lambda x: re.sub(r's$', '', str(x).replace('peak_', '')),
                inplace=True,
                axis=1,
            )
            print(other_peaks)
            peaks_df = peaks_df[peaks_df.apply(
                filter_peak,
                density=density,
                raw_density=raw_density,
                axis=1,
            )]
            new_peaks = list(set(peaks).union(set(peaks_df['intensity'])))
            if len(new_peaks) < int(2 ** np.ceil(np.log2(len(peaks)))):
                peaks = new_peaks
            else:
                print('Found too many. Exiting')
            print(peaks)
            print('new length', len(peaks))

    example.kde_data.write(pd.Series(dict(
        raw_density=raw_density,
        raw_intensity=intensity.copy(),
        density=density,
        intensity=intensity,
        peaks_intensities=intensity[peaks],
        peaks_densities=density[peaks],
        window=window,
    )).to_frame().T)
    example.write(df)

    example.timer_kde.write(str(t.elapse))


@click.command('process-kde')
@click.argument('files', type=click.Path(), nargs=-1, required=True)
@click.option('--mode', type=click.Choice(Mode.strings), required=True,
              default='auto',
              help='The algorithm mode. If `auto`, determine from filename.')
@click.option('--input-column', default='full_signal',
              help='The column of the input dataframe to consider as the raw, '
              'noisy input signal.')
@doctool.example(
    name='Process single file',
    args=['~/OneDrive/02. QuIN_Research/31. Noise-RTN/01. 2021_Algorithm paper/simulated_rtn/2021_07_23_generated_normal_rtn_data_4_white_noise_study/1-trap_wn=0.0_example=0_signals.feather'],
    help='To run the KDE step on a file, execute this command '
    '(the path depends on where you saved the data on your computer).',
)
@doctool.example(
    name='Process multiple files',
    args=['~/OneDrive/02. QuIN_Research/31. Noise-RTN/01. 2021_Algorithm paper/simulated_rtn/2021_07_23_generated_normal_rtn_data_4_white_noise_study/*_signals.feather'],
    help='''
    You can also process multiple examples in parallel by specifying a pattern
    rather than a single file. The example below uses the wildcard character `*`
    to match any file in the white noise study folder ending in
    `_signals.feather`.
    ''',
)
def process_kde_click(files, mode, input_column):
    """
    Run KDE step for the examples specified by FILES
    (specify `_signals.feather` or `_signals.csv` files).
    """

    with Pool() as p:
        consume(p.imap_unordered(
            partial(process_kde, mode=mode, input_column=input_column),
            files,
        ))


__all__ = ('process_kde_click',)

"""
Generate metastable aRTN data for validating our algorithm on anomalous examples

Developed by Marcel Robitaille on 2022/02/18 Copyright © 2021 QuIN Lab
"""

import random

import pandas as pd

from example import Example
from constants import ARTN_DATA_DIR, VALIDATION_SIGNAL_DURATION
from .generate_rtn_component import generate_rtn_component
from .add_noise import add_noise
from .result import DataGenerationResult


def generate_metastable(amplitudes, noise, signal_length, **kwargs):
    amplitude, = amplitudes

    tau_high_a = random.randint(50, 1_000)
    tau_low_a = random.randint(50, 1_000)
    zone_a = random.randint(5000, 100_000)
    zone_b = random.randint(5000, 100_000)
    tau_high_b = tau_low_a * (1 + random.uniform(-0.1, 0.1))
    tau_low_b = tau_high_a * (1 + random.uniform(-0.1, 0.1))

    parameters = pd.Series(dict(
        amplitude=amplitude,
        tau_high_a=tau_high_a,
        tau_low_a=tau_low_a,
        zone_a=zone_a,
        zone_b=zone_b,
        tau_high_b=tau_high_b,
        tau_low_b=tau_low_b,
    )).to_frame().T

    df = pd.DataFrame({
        'zone': generate_rtn_component(
            tau_low=zone_a,
            tau_high=zone_b,
            value_high=1,
            value_low=0,
            sig_len=signal_length,
        ),
        'a': generate_rtn_component(
            tau_low=tau_low_a,
            tau_high=tau_high_a,
            value_high=1,
            value_low=0,
            sig_len=signal_length,
        ),
        'b': generate_rtn_component(
            tau_low=tau_low_b,
            tau_high=tau_high_b,
            value_high=1,
            value_low=0,
            sig_len=signal_length,
        ),

    })
    df['rtn_sum'] = df['trap_0'] = df.zone * df.a - (df.zone - 1) * df.b

    df *= amplitude
    # plt.scatter(df.index, df.rtn_sum, c=df.zone)
    # plt.plot(df.rtn_sum)
    # plt.show()
    # up_transitions = df.index[(df.rtn.diff() > 0) & (df.zone == 1)].tolist()
    # down_transitions = df.index[(df.rtn.diff() < 0) & (df.zone == 1)].tolist()

    # # Calculate taus
    # t_ups_a = list(_get_taus(down_transitions, up_transitions))
    # t_downs_a = list(_get_taus(up_transitions, down_transitions))

    # up_transitions = df.index[(df.rtn.diff() > 0) & (df.zone == 0)].tolist()
    # down_transitions = df.index[(df.rtn.diff() < 0) & (df.zone == 0)].tolist()

    # # Calculate taus
    # t_ups_b = list(_get_taus(down_transitions, up_transitions))
    # t_downs_b = list(_get_taus(up_transitions, down_transitions))

    # def generator():
    #     for x in t_ups_a:
    #         yield 'a','up', x

    #     for x in t_ups_b:
    #         yield 'b','up', x

    #     for x in t_downs_a:
    #         yield 'a','down', x

    #     for x in t_downs_b:
    #         yield 'b','down', x
    # taus = pd.DataFrame(generator(), columns=['zone', 'which', 'tau'])
    # print(taus)
    # sns.boxplot(y='tau', x='which', hue='zone', data=taus, orient='v',
    #             showfliers=False)
    # plt.show()

    # Generate white noise
    df, noise = add_noise(df=df, noise=noise)

    # Add white noise to RTN signal
    df['full_signal'] = df['rtn_sum'] + df['white_noise']

    return DataGenerationResult(df, parameters, noise)


def save_metastable(example_number, noise, out_dir=ARTN_DATA_DIR,
                    signal_length=VALIDATION_SIGNAL_DURATION, **kwargs):

    amplitude = random.randint(10, 100)

    df, parameters, noise = generate_metastable(
        amplitudes=[amplitude],
        noise=noise,
        signal_length=signal_length,
    )

    example = Example(
        out_dir /
        f'artn_metastable_wn={noise:.2f}_example={example_number:02d}_'
        'signals.feather'
    )
    example.write(df)
    example.parameters.write(parameters)

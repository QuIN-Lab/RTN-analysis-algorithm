"""
Generate a 2-level RTN component

Developed by AJ Malcolm on 2020/11/02 Copyright © 2021 QuIN Lab
"""

import numpy as np

from constants import rng


def generate_rtn_component(
    tau_low,
    tau_high,
    sig_len=10000,
    value_high=0.5,
    value_low=-0.5,
):

    p_up = 1 / tau_high
    p_dn = 1 / tau_low

    rtn_signal = []
    state = np.random.binomial(1, tau_high / (tau_high + tau_low))
    while True:
        if bool(state):
            t_up = rng.geometric(p_up)
            if sig_len < (len(rtn_signal) + t_up):
                rtn_signal.extend((sig_len - len(rtn_signal)) * [value_high])
                break
            rtn_signal.extend(t_up * [value_high])
            state = 0
        else:
            t_dn = rng.geometric(p_dn)
            if sig_len < (len(rtn_signal) + t_dn):
                rtn_signal.extend((sig_len - len(rtn_signal)) * [value_low])
                break
            rtn_signal.extend(t_dn * [value_low])
            state = 1
    return np.asarray(rtn_signal)

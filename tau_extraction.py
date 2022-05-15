"""
Extract the list of high and low dwell times from a discretized 2-level RTN
component signal.

Developed by Marcel Robitaille on 2022/02/09 Copyright © 2021 QuIN Lab
"""

from typing import List, Tuple

import numpy as np


def _get_stay_times_from_transition_sequences(A, B):
    # pylint: disable=inconsistent-return-statements
    """
    Loop through two sets of points
    and yield the difference between them
    A and B should be times of up transitions and down transitions
    The order depends whether you want to calculate tau_up or tau_down
    """

    A = A.copy()
    B = B.copy()

    if not A or not B:
        return []

    if A[0] < B[0]:
        A.pop(0)

    for a, b in zip(A, B):
        yield a - b
    return StopIteration


def get_taus(digitized_sequence, time) -> Tuple[List[float], List[float]]:
    """
    Get a list of high times and low times from a digitized 2-level
    signal
    """

    diff = digitized_sequence.diff()
    up_transitions = time[diff > 0].tolist()
    down_transitions = time[diff < 0].tolist()
    t_ups = list(_get_stay_times_from_transition_sequences(
        down_transitions, up_transitions))
    t_downs = list(_get_stay_times_from_transition_sequences(
        up_transitions, down_transitions))
    return t_ups, t_downs


def get_taus_mean(digitized_sequence, time) -> Tuple[float, float]:
    t_ups, t_downs = get_taus(digitized_sequence=digitized_sequence, time=time)
    return np.mean(t_ups), np.mean(t_downs)

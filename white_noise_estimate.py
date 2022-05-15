"""
Estimate the white noise in a signal by taking a small sample many places. Like
this, we hope to get some statistics from the signal without catching any of the
switching events.

Developed by Marcel Robitaille on 2022/03/18 Copyright © 2021 QuIN Lab
"""

import numpy as np


def white_noise_estimate(signal):
    SAMPLE_WIDTH = 30
    NUM_SAMPLES = 100

    samples = [
        signal.loc[i:i+SAMPLE_WIDTH].std()
        for i in np.linspace(0, len(signal) - SAMPLE_WIDTH - 1, NUM_SAMPLES)
    ]
    samples = [
        sample for sample in samples if sample <= 3 * min(samples)
    ]
    return np.median(samples)

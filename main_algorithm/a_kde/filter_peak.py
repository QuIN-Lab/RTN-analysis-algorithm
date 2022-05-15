"""
See docstring of `filter_peaks` below.

Developed by Marcel Robitaille on 2022/02/18 Copyright © 2021 QuIN Lab
"""


def filter_peak(peak, density, raw_density):
    """
    Firlter out artificial peaks caused by (we think) ringing
    They usually are right next to a big, valid peak.

    The first check removes all those where the raw density is practially zero.
    This is clearly just artificial.

    It gets trickier near valid peaks, where the raw density starts ramping up.
    Therefore, we check if the peak is kind of near a low raw density
    and near a high filtered density
    """

    RAW_LOOKAROUND = 7
    FILTERED_LOOKAROUND = 45
    DENSITY_MAX_DIVISOR = 25
    HEIGHT_DIVISOR = 10
    # A peak must be less than this before it is even considered for rippling
    # filtering
    HEIGHT_THRESHOLD = 0.0035
    PROMINENCE_THRESHOLD = 0.002

    i = int(peak['intensity'])
    # TODO: Figure out if we really need this check or if the below check covers
    # it
    if raw_density[i] < 0.0002:
        print(f'peak {i} raw_density very small')
        return False

    density_near = density[i-FILTERED_LOOKAROUND-1:i+FILTERED_LOOKAROUND]
    raw_density_near = raw_density[i-RAW_LOOKAROUND-1:i+RAW_LOOKAROUND]
    raw_density_threshold = max(
        density.max() / DENSITY_MAX_DIVISOR,
        peak['prominence'] / HEIGHT_DIVISOR,
    )
    print('Filtering with raw density threshold:', raw_density_threshold)
    peak_condition = peak['height'] < HEIGHT_THRESHOLD
    prom_condition = peak['prominence'] < PROMINENCE_THRESHOLD
    raw_condition = (raw_density_near < raw_density_threshold).any()
    den_condition = (density_near > density[i] * 4.9).any()
    print(peak_condition, prom_condition, raw_condition, den_condition)
    if peak_condition and prom_condition and raw_condition and den_condition:
        # if (raw_density_near < 0.001).any():
        print(peak)
        print('other reason')
        print(density.max())
        print(raw_density_near.min())
        print(density.max() / raw_density_near.min())
        print('density near max', density_near.max())
        return False

    return True

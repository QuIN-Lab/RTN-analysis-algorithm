"""
Define the mixture model and system of equations used for the second step.

Developed by Marcel Robitaille on 2022/03/18 Copyright © 2021 QuIN Lab
"""

from functools import reduce
from operator import add

from lmfit import Parameters
from lmfit.models import GaussianModel
import numpy as np

from mode import Mode


def define_mixture_model(
    n_traps,
    s_min,
    s_max,
    sig,
    base,
    separations,
    probabilities,
    # Coupling factor (coupled aRTN only) is the ratio of the true
    # highest-intensity peak to the sum of the other peaks
    # For instance, if the highest peak comes 20% before the sum of the other
    # two peaks, then this factor is 0.8
    coupling_factor,
    mode,
):
    """
    Set up the lmfit model for the GMM
    """

    parameters = Parameters()
    gaussians = [
        GaussianModel(prefix=f'g{i:0{n_traps}b}_')
        for i in range(2 ** n_traps)
    ]

    parameters.add('sig', **sig)
    parameters.add('base', **base)
    parameters.add('epsilon', max=1e-9, min=-1e-9, vary=True)
    parameters.add('coupling_factor', **coupling_factor)

    for i, s in enumerate(separations):
        parameters.add(f'sep_trap{i}', min=s_min, max=s_max, **s)

    for i, p in enumerate(probabilities):
        parameters.add(f'p_trap{i}', min=0.001, max=0.999, **p)

    for i, gaussian in enumerate(gaussians):
        parameters.update(gaussian.make_params())

        bits = list(reversed(
            np.unpackbits(np.array([i], dtype='>u1').view(np.uint8))
        ))[:n_traps]

        seps = [f'+ sep_trap{j}' for j, b in enumerate(bits) if b]
        if gaussian.prefix == 'g11_':
            seps = [x + '*coupling_factor' for x in seps]
        parameters[f'{gaussian.prefix}center'].set(
            expr='base' + ''.join(seps),
            vary=True,
        )

        # TODO: Find a better way to do this
        if mode == Mode.MISSING_LEVEL and i == 2:
            parameters[f'{gaussian.prefix}amplitude'] \
                .set(value=0, vary=False)
        else:
            parameters[f'{gaussian.prefix}amplitude'] \
                .set(expr=' * '.join(
                    f'(1 - p_trap{i})' if b else f'p_trap{i}'
                    for i, b in enumerate(bits)
                ) + (' + epsilon' if i == len(gaussians) - 1 else ''))

        parameters[f'{gaussian.prefix}sigma'].set(expr='sig', min=0)
        parameters[f'{gaussian.prefix}height'].set(min=0)

    model = reduce(add, gaussians)
    return model, parameters

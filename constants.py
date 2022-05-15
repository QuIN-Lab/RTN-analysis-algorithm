"""
Constants used by the algorithm that should be available globally.

Developed by Marcel Robitaille on 2022/01/26 Copyright © 2021 QuIN Lab
"""

from pathlib import Path
from enum import Enum, auto
from typing import Union

from numpy.random import default_rng
from rich.console import Console
from environs import Env, EnvError


# Number of examples to generate for each number of traps
NUM_VALIDATION_EXAMPLES = 10

# Number of points in signal
VALIDATION_SIGNAL_DURATION = 1_000_000

NUM_TRAINING_DATA_TO_GENERATE = 20
TRAINING_SIGNAL_LENGTH = 50_000  # points
BATCH_SIZE = 1000
KERNEL_SIZE = 3
STEPS = 64

rng = default_rng()
console = Console(record=True)


env = Env()
env.read_env()


class MockPath:
    """
    A kind of null Path that does not throw an error on `/` but does when you
    try to get the string value.
    Used so we can do `@doctool.example(args=[DATA_DIR / 'something'])`
    without an error running the main code, but it does throw an error when
    updating the docs (getting path string value).
    May be useful in other situations too.
    """
    def __truediv__(self, _other):
        return self
    def __str__(self):
        raise RuntimeError('`BASE_DIR` is not defined in `.env`.')


def get_data_dir(path: str) -> Union[Path, MockPath]:
    """
    Helper to get dir relative to env('BASE_DIR') or MockPath if .env not set
    """

    try:
        return env.path('BASE_DIR') / path
    except EnvError:
        return MockPath()


DATA_DIR = get_data_dir('2021_07_23_generated_normal_rtn_data_4_white_noise_study')
PINK_DATA_DIR = get_data_dir('2022_04_08_pink_noise')
ARTN_DATA_DIR = get_data_dir('2022_01_10_generated_anomalous_data_1_wn=20_examples=30')
CNT_DATA_DIR = get_data_dir('30. CNT real data')

# List of usable CNT example numbers
CNT_EXAMPLES = [1, 3, 4, 5, 6, 8]

LOG_DIR = Path('./logs/')
LOG_DIR.mkdir(parents=True, exist_ok=True)


class Publisher(Enum):
    NATURE = auto()
    IEEE = auto()

PUBLISHER = Publisher.NATURE

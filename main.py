"""
Entry point for everything. Loads all the click commands.

Developed by Marcel Robitaille on 2022/03/15 Copyright © 2021 QuIN Lab
"""

import traceback
from datetime import datetime

import click

from main_algorithm import a_kde, b_gmm, c_rnn, d_hmm
import results_presentation
import data_generation
import data_exploration
import experiments_debugging
import results_aggregation
from constants import LOG_DIR, console


@click.group()
def main():
    pass


def load_all(module):
    for command in module.__all__:
        main.add_command(getattr(module, command))


load_all(a_kde)
load_all(b_gmm)
load_all(c_rnn)
load_all(d_hmm)
load_all(results_presentation)
load_all(data_generation)
load_all(data_exploration)
load_all(experiments_debugging)
load_all(results_aggregation)


if __name__ == '__main__':
    now = datetime.today().strftime('%Y-%m-%d-%H-%M-%S')
    try:
        main()  # pylint: disable=no-value-for-parameter
    except SystemExit:
        pass
    except:  # noqa pylint: disable=bare-except
        traceback.print_exc()
        console.print_exception()
    console.save_text(str(LOG_DIR / f'{now}.log'))

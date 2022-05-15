"""
Utility class to quickly get the different variants of a file.
For example, quickly get the truth parameters if you have the time series noisy
signal.

Developed by Marcel Robitaille on 2022/03/18 Copyright © 2021 QuIN Lab
"""

import json
import re
import copy
from pathlib import Path
from typing import TypeVar, Generic

import pandas as pd
import lmfit
from tensorflow import keras


T = TypeVar('T', pd.DataFrame, str, lmfit.model.ModelResult, keras.Model)
S = TypeVar('S', bound='Example')


# TODO: Fix global
variants = set()


def variant(method):
    variants.add(method.__name__)
    return method

class Example(Generic[T]):
    filetypes = {'feather', 'csv', 'pkl', 'json', 'txt', 'lmfit', 'h5'}

    def __init__(self, path):
        self.re = re.compile(
            fr'({"|".join(variants)}|)\.'
            fr'({"|".join(self.filetypes)})$'
        )

        # Support passing an instance of Example to Example
        # That should just do nothing
        # This is useful for "duck typing"
        # Functions should be able to take "anything that can be converted to
        # Example" and not "exactly an instance of Example"
        self.path = path.path if isinstance(path, Example) else \
                Path(path).expanduser()
        self.path = self.path.resolve()

    def read(self) -> T:
        # I looked into asserting the generic variant of the Example,
        # but this is not possible right now.
        # There is discussion of a method, but it's undocumented and
        # didn't work for me.
        # Let's use `type: ignore` for now.
        if self.path.suffix == '.json':
            with open(self.path, 'r', encoding='utf-8') as f:
                return json.load(f)

        if self.path.suffix == '.txt':
            with open(self.path, 'r', encoding='utf-8') as f:
                return f.read()  # type: ignore

        if self.path.suffix == '.lmfit':
            return lmfit.model.load_modelresult(self.path)  # type: ignore

        ft = self.path.suffix[1:].replace('pkl', 'pickle')
        return getattr(pd, f'read_{ft}')(self.path)

    def write(self, df):
        if self.path.suffix == '.json':
            with open(self.path, 'w', encoding='utf-8') as f:
                return json.dump(df, f)

        if self.path.suffix == '.txt':
            with open(self.path, 'w', encoding='utf-8') as f:
                return f.write(df)

        if self.path.suffix == '.lmfit':
            return lmfit.model.save_modelresult(df, self.path)

        ft = self.path.suffix[1:].replace('pkl', 'pickle')
        kwargs = {}
        if self.path.suffix == '.csv':
            kwargs['index'] = False
        return getattr(df, f'to_{ft}')(self.path, **kwargs)

    @property
    @variant
    def signals(self) -> 'Example[pd.DataFrame]':
        return self.with_name('signals.feather')

    @property
    @variant
    def parameters(self) -> 'Example[pd.DataFrame]':
        return self.with_name('parameters.csv')

    @property
    @variant
    def decomp_data(self) -> 'Example[pd.DataFrame]':
       return self.with_name('decomp_data.csv')

    @property
    @variant
    def decomp_data_traps(self) -> 'Example[pd.DataFrame]':
       return self.with_name('decomp_data_traps.csv')

    @property
    @variant
    def discretization_data(self) -> 'Example[pd.DataFrame]':
       return self.with_name('discretization_data.feather')

    @property
    @variant
    def kde_data(self) -> 'Example[pd.DataFrame]':
       return self.with_name('kde_data.feather')

    @property
    @variant
    def sep_error(self) -> 'Example[pd.DataFrame]':
       return self.with_name('sep_error.feather')

    @property
    @variant
    def timer_kde(self) -> 'Example[str]':
       return self.with_name('timer_kde.txt')

    @property
    @variant
    def timer_decompose(self) -> 'Example[str]':
       return self.with_name('timer_decompose.txt')

    @property
    @variant
    def time_series_predictions(self) -> 'Example[pd.DataFrame]':
       return self.with_name('time_series_predictions.feather')

    @property
    @variant
    def predicted_time_series(self) -> 'Example[pd.DataFrame]':
       return self.with_name('predicted_time_series.feather')

    @property
    @variant
    def gmm_fit(self) -> 'Example[lmfit.model.ModelResult]':
       return self.with_name('gmm_fit.lmfit')

    @property
    @variant
    def tf_model(self) -> 'Example[keras.Model]':
       return self.with_name('tf_model.h5')

    def with_name(self, name) -> 'Example':
        return Example(
            self.re.sub(name, str(self.path))
            if self.re.search(str(self.path)) else
            f'{self.path}_{name}'
        )

    def with_parent(self: S, parent) -> S:
        res = copy.deepcopy(self)
        res.path = parent / res.path.name
        return res

    def __repr__(self):
        return f'Example({self.path.name})'

    def is_time_series_predictions(self):
        """
        Return whether an example is the time series predictions.
        Abstract this away because there are two conventions for naming this
        file.
        TODO: Consolidate on only one convention.
        """

        return self.path.name.endswith('_time_series_predictions.feather') or \
            self.path.name.endswith('_predicted_time_series.feather')

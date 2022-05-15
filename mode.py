"""
Enum of the running modes of the algorithm.

Developed by Marcel Robitaille on 2022/01/26 Copyright © 2021 QuIN Lab
"""

import re


class Symbol:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return f'Symbol({self.name})'


class Mode:
    WN_STUDY = Symbol('WN_STUDY')
    CNT_REAL_DATA = Symbol('CNT_REAL_DATA')
    METASTABLE = Symbol('METASTABLE')
    COUPLED = Symbol('COUPLED')
    MISSING_LEVEL = Symbol('MISSING_LEVEL')
    MUTUALLY_EXCLUSIVE = Symbol('MUTUALLY_EXCLUSIVE')

    string_map = {
        'normal': 'WN_STUDY',
        'metastable': 'METASTABLE',
        'coupled': 'COUPLED',
        'missing-level': 'MISSING_LEVEL',
        'mutually-exclusive': 'MUTUALLY_EXCLUSIVE',
    }

    @staticmethod
    def _get_from_filename(filename):
        filename = str(filename)
        if 'metastable' in filename:
            return Mode.METASTABLE
        if 'coupled' in filename or 'correlated' in filename:
            return Mode.COUPLED
        if re.search(r'missing[\s\-_]level', filename, re.IGNORECASE):
            return Mode.MISSING_LEVEL
        if 'cnt' in filename.lower():
            return Mode.CNT_REAL_DATA
        return Mode.WN_STUDY

    @classmethod
    def from_str(cls, mode, filename):
        if isinstance(mode, Symbol):
            return mode
        if mode == 'auto':
            return cls._get_from_filename(filename=filename)
        return getattr(cls, cls.string_map[mode])

    @classmethod
    @property
    def strings(cls):
        return list(cls.string_map.keys()) + ['auto']

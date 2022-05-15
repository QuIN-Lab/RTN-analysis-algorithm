from collections import namedtuple


DataGenerationResult = namedtuple(
    'DataGenerationResult',
    ['dataframe', 'parameters', 'noise'],
)


__all__ = ('DataGenerationResult',)

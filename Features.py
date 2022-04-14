import pandas as pd
import talib
from talib import abstract
from tqdm import tqdm


class Features:
    def __init__(self, df, close='close'):
        self.df = df
        self.close = close

    def add_indicator(self, indicator, spesific_indicators=None, **extra_params):
        indicator_function = talib.abstract.Function(indicator, **extra_params)
        output_length = len(indicator_function.__dict__['_Function__outputs'])
        out_keys = indicator_function.__dict__['_Function__outputs'].keys()
        if output_length == 1:
            self.df[indicator] = self.df.groupby('ticker')[self.close].transform(lambda x: indicator_function(x, **extra_params))
        else:
            if spesific_indicators is None:
                for out_cnt, out in enumerate(out_keys):
                    self.df[out] = self.df.groupby('ticker')[self.close].transform(lambda x: indicator_function(x, **extra_params)[out_cnt])
            else:
                for out_cnt, out in enumerate(out_keys):
                    if out in spesific_indicators:
                        self.df[out] = self.df.groupby('ticker')[self.close].transform(lambda x: indicator_function(x, **extra_params)[out_cnt])

    def get_extra_indicators(self, indicator_lst):
        for indicator in tqdm(indicator_lst):
            indicator_function = abstract.Function(indicator)
            output_length = len(indicator_function.__dict__['_Function__outputs'])
            out_keys = indicator_function.__dict__['_Function__outputs'].keys()
            if output_length == 1:
                self.df[indicator] = self.df.groupby('ticker')[self.close].transform(lambda x: indicator_function(x))
            # else:
                for out_cnt, out in enumerate(out_keys):
                    self.df[out] = self.df.groupby('ticker')[self.close].transform(lambda x: indicator_function(x)[out_cnt])
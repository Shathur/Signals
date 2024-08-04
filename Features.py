import pandas as pd
import talib
from talib import abstract
from tqdm import tqdm


class Features:
    def __init__(self, df, close="close"):
        self.df = df
        self.close = close
        self.initial_features = df.copy().columns
        self.added_features = None
        self.quantile_features = None

    def add_indicator(self, indicator, specific_indicators=None, **extra_params):
        indicator_function = talib.abstract.Function(indicator, **extra_params)
        output_length = len(indicator_function.__dict__["_Function__outputs"])
        out_keys = indicator_function.__dict__["_Function__outputs"].keys()
        if output_length == 1:
            self.df[indicator] = self.df.groupby("ticker")[self.close].transform(
                lambda x: indicator_function(x, **extra_params)
            )
        else:
            if specific_indicators is None:
                for out_cnt, out in enumerate(out_keys):
                    self.df[out] = self.df.groupby("ticker")[self.close].transform(
                        lambda x: indicator_function(x, **extra_params)[out_cnt]
                    )
            else:
                for out_cnt, out in enumerate(out_keys):
                    if out in specific_indicators:
                        self.df[out] = self.df.groupby("ticker")[self.close].transform(
                            lambda x: indicator_function(x, **extra_params)[out_cnt]
                        )
        # update list of available features
        self.added_features = list(set(self.df.columns) - set(self.initial_features))

    def get_extra_indicators(self, indicator_lst):
        for indicator in tqdm(indicator_lst):
            indicator_function = abstract.Function(indicator)
            output_length = len(indicator_function.__dict__["_Function__outputs"])
            out_keys = indicator_function.__dict__["_Function__outputs"].keys()
            if output_length == 1:
                self.df[indicator] = self.df.groupby("ticker")[self.close].transform(
                    lambda x: indicator_function(x)
                )
            else:
                for out_cnt, out in enumerate(out_keys):
                    self.df[out] = self.df.groupby("ticker")[self.close].transform(
                        lambda x: indicator_function(x)[out_cnt]
                    )
        # update list of available features
        self.added_features = list(set(self.df.columns) - set(self.initial_features))

    def get_quantiles(self):
        # keep the quantiles of our indicators
        for indicator in self.added_features:
            self.df[indicator + "_quantile"] = self.df.groupby("friday_date")[
                indicator
            ].transform(
                lambda x: pd.qcut(
                    x=x, q=[0, 0.25, 0.5, 0.75, 1], labels=False, duplicates="drop"
                )
            )
        # update list of available features
        self.added_features = list(set(self.df.columns) - set(self.initial_features))
        self.quantile_features = [f for f in self.df.columns if f.endswith("quantile")]

    def int_transform(self, features_lst=None):
        # save data as int8 to save space before the great chunk of added features
        if features_lst is None:
            data_types = {col: "int8" for col in self.quantile_features}
            self.df = self.df.astype(data_types)
        else:
            data_types = {col: "int8" for col in features_lst}
            self.df = self.df.astype(data_types)

    def remove_indicator(self, indicators_lst):
        # remove list of indicators
        self.df.drop(columns=indicators_lst, inplace=True)
        self.added_features = list(set(self.added_features) - set(indicators_lst))


import os
import sys
from dataclasses import dataclass
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta, FR
import pickle
import gc
from .cross_validation import (
    cv_split_creator,
    TimeSeriesSplitGroups,
    TimeSeriesSplitGroupsPurged,
    RandomSplits,
    WindowedGroups,
)
from .utils import check_enough_tickers, start_end_date, check_if_live
from .train import train_val, submit_signal
from .predictions import get_predictions
from setup_env_variables import setup


class CV_scheme:
    RandomSplits = RandomSplits
    TimeSeriesSplitGroups = TimeSeriesSplitGroups
    TimeSeriesSplitGroupsPurged = TimeSeriesSplitGroupsPurged
    WindowedGroups = WindowedGroups


class config:
    # possible target names
    # target_20d
    # target_20d_raw_return
    # target_20d_factor_neutral
    # target_20d_factor_feat_neutral
    TARGET_NAME = "target_20d_factor_feat_neutral"
    PREDICTION_NAME = "prediction"
    SAVE_TO_DRIVE = True
    SAVE_FOLDER = "./"
    VISUALIZE = False


@dataclass
class Trainer:
    """
    Encapsules our whole training pipeline.
    Built around train.py
    :: param: data_path: str
    :: param: df: pd.DataFrame()
    :: param: tour_df: pd.DataFrame()
    :: param: last_friday: int
    :: param: feature_names: list
    :: param: target_name: str
    :: param: model_params: dict
    :: param: fit_params: dict
    :: param: col:str
    :: param: cv_scheme: CV_scheme
    :: param: n_splits: int
    :: param: is_string: bool
    :: param: extra_constructor_params: dict
    :: param: extra_params: dict
    :: param: return_col: bool
    """

    data_path: str
    df: pd.DataFrame()
    tour_df: pd.DataFrame()
    last_friday: int
    feature_names: list
    num_tour_weeks: int
    target_name: str
    model_type: str
    model_params: dict
    fit_params: dict
    col: str
    cv_scheme: CV_scheme
    n_splits: int
    is_string: bool
    extra_constructor_params: dict
    extra_params: dict
    return_col: bool

    def get_data(self):
        self.df = pd.read_parquet(self.data_path)

    def get_live_data(self):
        self.last_friday = check_enough_tickers(self.df, True)
        self.live_df = self.df[self.df[self.col] == self.last_friday]

    def get_no_live_data(self):
        self.last_friday = check_enough_tickers(self.df, True)
        print(f"last_friday inside get_no_live_data: {self.last_friday}")
        df_wo_live = self.df[self.df[self.col] < self.last_friday]
        # when we approach the end of our dataframe many targets are not yet filled
        # so we drop na values regarding only our features
        # there are some nans in the targets though and we need to
        # get rid of them. So we are ignoring the 2 lines above
        df_wo_live.dropna(inplace=True)
        # the line below doesnt work because live has some nan targets
        # so for training we need to keep only non live features
        # self.df = pd.concat([df_wo_live, self.df[self.df[self.col]==self.last_friday]])
        self.df = df_wo_live
        # print(f"df_w_live: {self.df[self.col].unique().tolist().sort()[-10:-1]}")

    def get_more_features(self):
        self.feature_names = [
            f for f in self.df.columns if (("lag" in f) or ("diff" in f))
        ]

    def get_vol_features(self):
        target_features = [t for t in self.df.columns.tolist() if "target" in t]
        vol_non_features = ["friday_date", "ticker", "bloomberg_ticker", "data_type"]
        self.feature_names = [
            f for f in self.df.columns if f not in target_features + vol_non_features
        ]

    def get_combine_vol_features(self, features_filepath):
        with open(os.path.abspath(features_filepath), "rb") as file:
            imp_feats_20 = pickle.load(file)
        imp_feats_20 = imp_feats_20 + ["ticker", "date", "friday_date"]
        # load the more dataframe
        # for this iteration of cobmine vol we just need the more
        more = pd.read_parquet(
            os.path.join(
                os.path.dirname(__file__),
                os.pardir,
                "Data",
                "Signals",
                "more_full_features.parquet",
            )
        )
        temp_20 = more.loc[:, imp_feats_20]
        # load df
        self.get_data()
        # keep only common dates to save memory
        self.df = self.df[self.df["friday_date"].isin(temp_20["friday_date"].unique())]
        # merge
        self.df = self.df.merge(temp_20, how="left", on=["friday_date", "ticker"])
        print(f"merged features: {self.df.columns.tolist()}")
        print(f"merged num of features: {len(self.df.columns.tolist())}")
        # now that we have the combined features we seperate live and no_live
        self.get_live_data()
        self.get_no_live_data()
        # free memory
        # del temp_20
        # gc.collect()
        # keep combined features
        targets_columns = [t for t in self.df.columns if t.startswith("target")]
        standard_columns = [
            "data_type",
            "date",
            "friday_date",
            "ticker",
            "bloomberg_ticker",
        ]
        drops = targets_columns + standard_columns
        # keeps = [f for f in self.df.columns if ((('quantile' in f) or ('diff' in f)) and (f not in drops))] # + imp_feats_20
        keeps = [f for f in self.df.columns if f not in drops]
        # feature_names = [f for f in extra_volatility_quantiles.columns.values.tolist() if f not in drops]
        self.feature_names = keeps

    def get_CV_split(self):
        return cv_split_creator(
            df=self.df,
            col=self.col,
            cv_scheme=self.cv_scheme,
            n_splits=self.n_splits,
            is_string=self.is_string,
            extra_constructor_params=self.extra_constructor_params,
            extra_params=self.extra_params,
            return_col=self.return_col,
        )

    def get_start_end_date(self):
        print(start_end_date(self.df, "friday_date"))
        print(start_end_date(self.df, "friday_date"))

    def test_dates(self):
        if (self.num_tour_weeks > 1) and (self.num_tour_weeks % 1 == 0):
            tour_period = [
                int(
                    (datetime.now() + relativedelta(weekday=FR(-int(i)))).strftime(
                        "%Y%m%d"
                    )
                )
                for i in np.arange(1, self.num_tour_weeks)
            ]
        else:
            raise ValueError("self.num_tour_weeks needs to be a positive integer")
        return tour_period

    def get_train_val_test(self, tour_period):
        """
        method to be used if num_tour_weeks is not None
        tour period can be the return of self.test_dates()
        If you don't run this method, self.df is already
        defined and you will be running everything without a tour_df
        ::param: tour_period: list of dates -- must be in the correct format
                                               different for each self.df
        """
        self.tour_df = self.df[self.df["friday_date"].isin(tour_period)]
        self.df = self.df[
            ~self.df["friday_date"].isin(self.tour_df["friday_date"].unique())
        ]

    def train(self, save_to_drive, save_folder):
        cv_split_data = self.get_CV_split()
        for count, cv_split in enumerate(cv_split_data):
            print(f"split {count}: {cv_split}")
            print(
                f"unique validation dates: {self.df.iloc[cv_split[1]]['friday_date'].nunique()}"
            )
        if self.tour_df.empty:
            print(f"no tour date so no unique tour dates")
        else:
            print(f"unique tour dates: {self.tour_df['friday_date'].nunique()}")
        metrics = train_val(
            df=self.df,
            feature_names=self.feature_names,
            target_name=self.target_name,
            pred_name=config.PREDICTION_NAME,
            cv_split_data=cv_split_data,
            model_type=self.model_type,
            model_params=self.model_params,
            fit_params=self.fit_params,
            date_col=self.col,
            tour_df=self.tour_df,
            save_to_drive=save_to_drive,
            save_folder=save_folder,
            visualize=config.VISUALIZE,
        )
        return metrics

    def get_validation_sub(self, metrics):
        validation_predictions = np.mean(metrics[2][2], axis=0)
        validation_sub = self.tour_df.copy()
        validation_sub["signal"] = validation_predictions
        return validation_sub

    def get_live_sub(self, save_folder):
        # live sub
        # self.live_df.loc[self.df['friday_date'] == self.last_friday, 'data_type'] = 'live'
        # live_sub = self.df.query('data_type == "live"').copy()
        self.live_df.loc[:, ["data_type"]] = "live"
        live_sub = self.live_df
        live_sub["signal"] = get_predictions(
            df=live_sub[self.feature_names],
            num_models=len(os.listdir(save_folder)),
            folder_name=save_folder,
        )
        return live_sub

    def prepare_sub_for_submission(self, sub):
        """prepare dataframe for submission"""
        sub = check_if_live(sub, "friday_date")
        # select necessary columns
        sub = sub[["ticker", "friday_date", "data_type", "signal"]]
        # rename ticker
        sub = sub.rename(columns={"ticker": "bloomberg_ticker"})
        # until now date was an integer. That was accomodating to our needs
        # now we need to transform it into the appropriate format
        date_col = [c for c in sub.columns.tolist() if "date" in c]
        sub[date_col[0]] = pd.to_datetime(
            sub[date_col[0]], format="%Y%m%d"
        ).dt.strftime("%Y-%m-%d")
        return sub

    def create_submit_sub(
        self,
        validation_sub: pd.DataFrame(),
        live_sub: pd.DataFrame(),
        submit: bool,
        submit_diagnostics: bool,
        submit_reverse: bool,
        submit_reverse_diagnostics: bool,
        upload_name: str,
        model_name: str,
        model_name_reverse: str,
    ):
        # setup env variables
        setup()
        # load keys from global environment
        public_key = os.getenv("PUBLIC_ID")
        secret_key = os.getenv("SECRET_KEY")
        # concat valid and test
        sub = pd.concat([validation_sub, live_sub], ignore_index=True)
        # prepare sub for submission
        sub = self.prepare_sub_for_submission(sub)
        print(f"Our sub before submission is: {sub}")
        # submit
        submit_signal(
            sub,
            public_key,
            secret_key,
            submit,
            submit_diagnostics,
            model_name,
            upload_name=upload_name,
        )
        # submit reverse
        if (submit_reverse) or (submit_reverse_diagnostics):
            reverse_sub = sub
            reverse_sub["signal"] = reverse_sub.groupby("friday_date")["signal"].rank(
                pct=True, method="first", ascending=False
            )
            reverse_sub.reset_index(drop=True, inplace=True)
            submit_signal(
                sub=reverse_sub,
                public_id=public_key,
                secret_key=secret_key,
                submit=submit_reverse,
                submit_diagnostics=submit_reverse_diagnostics,
                slot_name=model_name_reverse,
                upload_name=upload_name,
            )

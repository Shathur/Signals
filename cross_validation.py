import pandas as pd
from warnings import simplefilter

simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
import numpy as np

from sklearn.model_selection._split import _BaseKFold, indexable, _num_samples

from sklearn.metrics import log_loss, accuracy_score


class RandomSplits(_BaseKFold):
    def __init__(self, n_splits=None):
        self.n_splits = n_splits
        super().__init__(n_splits, shuffle=False, random_state=None)

    def split(self, X, y=None, groups=None):
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        group_lst = np.unique(groups)
        n_groups = len(group_lst)

        indices = np.arange(n_samples)

        cutoff_eras = n_groups // self.n_splits
        np.random.shuffle(group_lst)

        for i in range(self.n_splits):
            yield (
                indices[
                    groups.isin(
                        group_lst[i * cutoff_eras : i * cutoff_eras + cutoff_eras]
                    )
                ],
                indices[
                    groups.isin(
                        group_lst[i * cutoff_eras : i * cutoff_eras + cutoff_eras]
                    )
                ],
            )


class TimeSeriesSplitGroups(_BaseKFold):
    """
    Code kindly provided by Michael Oliver in the Numer.ai forum
    https://forum.numer.ai/t/era-wise-time-series-cross-validation/791
    """

    def __init__(self, n_splits=None):
        super().__init__(n_splits, shuffle=False, random_state=None)

    def split(self, X, y=None, groups=None):
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        n_splits = self.n_splits
        n_folds = n_splits + 1
        group_list = np.unique(groups)
        n_groups = len(group_list)
        if n_folds > n_groups:
            raise ValueError(
                (
                    "Cannot have number of folds ={0} greater"
                    " than the number of samples: {1}."
                ).format(n_folds, n_groups)
            )
        indices = np.arange(n_samples)
        test_size = n_groups // n_folds
        test_starts = range(test_size + n_groups % n_folds, n_groups, test_size)
        test_starts = list(test_starts)[::-1]
        for test_start in test_starts:
            yield (
                indices[groups.isin(group_list[:test_start])],
                indices[groups.isin(group_list[test_start : test_start + test_size])],
            )


class TimeSeriesSplitGroupsPurged(_BaseKFold):
    """
    Code kindly provided by Michael Oliver in the Numer.ai forum
    https://forum.numer.ai/t/era-wise-time-series-cross-validation/791
    """

    def __init__(self, n_splits=None, embg_grp_num=None):
        if n_splits > 1:
            super().__init__(n_splits, shuffle=False, random_state=None)
        else:
            self.n_splits = n_splits
        self.embg_grp_num = embg_grp_num

    def split(self, X, y=None, groups=None):
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        n_splits = self.n_splits
        embg_grp_num = self.embg_grp_num
        n_folds = n_splits + 1
        group_list = np.unique(groups)
        n_groups = len(group_list)
        if n_folds > n_groups:
            raise ValueError(
                (
                    "Cannot have number of folds : {0} greater"
                    " than the number of samples: {1}."
                ).format(n_folds, n_groups)
            )
        indices = np.arange(n_samples)
        test_size = n_groups // n_folds
        test_starts = range(test_size + n_groups % n_folds, n_groups, test_size)
        test_starts = list(test_starts)[::-1]
        if n_splits > 1:
            for test_start in test_starts:
                yield (
                    indices[groups.isin(group_list[: test_start - embg_grp_num])],
                    indices[
                        groups.isin(
                            group_list[
                                test_start
                                + embg_grp_num : test_start
                                + embg_grp_num
                                + test_size
                            ]
                        )
                    ],
                )
        elif n_splits == 1:
            yield (
                indices[
                    groups.isin(group_list[: int(0.8 * len(group_list)) - embg_grp_num])
                ],
                indices[
                    groups.isin(group_list[int(0.8 * len(group_list)) + embg_grp_num :])
                ],
            )
        else:
            raise ValueError(f"Invalid split value of {n_splits}")


class PurgedKfold(_BaseKFold):
    """
    Extend KFold class to work with labels that span intervals
    The train is purged of observations overlapping test-label intervals
    Test set is assumed contiguous (shuffle=False), w/o training samples in

    Code was adapted from Advances in Financial ML by Marcos Lopez De Prado.
    Snippet 7.3

    t1 : must be a pd.Series
    X and t1 must have the same index values
    """

    def __init__(self, n_splits=None, t1=None, pctEmbargo=None):
        super(PurgedKfold, self).__init__(n_splits, shuffle=False, random_state=None)
        self.t1 = t1
        self.pctEmbargo = pctEmbargo

    def split(self, X, y=None, groups=None):
        indices = np.arange(X.shape[0])
        mbrg = int(X.shape[0] * self.pctEmbargo)
        test_starts = [
            (i[0], i[-1] + 1)
            for i in np.array_split(np.arange(X.shape[0]), self.n_splits)
        ]

        for i, j in test_starts:
            t0 = self.t1.index[i]
            test_indices = indices[i:j]
            maxt1idx = self.t1.index.searchsorted(self.t1[test_indices].max())
            train_indices = self.t1.index.searchsorted(self.t1[self.t1 <= t0].index)
            if maxt1idx < X.shape[0]:
                train_indices = np.concatenate(
                    (train_indices, indices[maxt1idx + mbrg :])
                )
            yield train_indices, test_indices


class WindowedGroups(_BaseKFold):
    def __init__(self, n_splits=None):
        super().__init__(n_splits, shuffle=False, random_state=None)

    def split(self, X, y=None, groups=None, window_length=4):
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        group_lst = np.unique(groups)
        n_groups = len(group_lst)

        indices = np.arange(n_samples)

        eras = range(n_groups - window_length)
        eras = list(eras)
        for i in eras[:]:
            yield (
                indices[groups == group_lst[i]],
                indices[groups == group_lst[i + window_length]],
            )


def cvscore(
    clf,
    X,
    y,
    sample_weight,
    scoring="neg_log_loss",
    t1=None,
    cv=None,
    cvGen=None,
    pctEmbargo=None,
):
    """
    Alternative to cross_val_score for the PurgedKFold class
    cross_val_score will give different results because it passes weights to the fit method, but
    not to the log_loss method.

    scoring : Must belong in ['neg_log_loss', 'accuracy']
    """
    if cvGen is None:
        cvGen = PurgedKfold(n_splits=cv, t1=t1, pctEmbargo=pctEmbargo)
    score = []

    for train, test in cvGen.split(X=X):
        fit = clf.fit(
            X=X.iloc[train, :],
            y=y.iloc[train],
            sample_weight=sample_weight.iloc[train].values,
        )
        if scoring == "neg_log_loss":
            prob = fit.predict_proba(X.iloc[test, :])
            score_ = -log_loss(
                y.iloc[test],
                prob,
                sample_weight=sample_weight.iloc[test].values,
                labels=clf.classes_,
            )
        else:
            pred = fit.predict(X.iloc[test, :])
            score_ = accuracy_score(
                y.iloc[test], pred, sample_weight=sample_weight.iloc[test].values
            )

        score.append(score_)

    return np.array(score)


def cv_split_creator(
    df,
    col,
    cv_scheme=TimeSeriesSplitGroups,
    n_splits=4,
    is_string=False,
    extra_constructor_params={},
    extra_params={},
    return_col=False,
):

    # add another column with date id to feed the cv splitter
    if col + "_No" not in df.columns:
        if is_string:
            dateno_values = [int("".join(i for i in x if i.isdigit())) for x in df[col]]
            # dateno_values need to be pd.Series or pd.DataFrame
            dateno_values = pd.Series(dateno_values)
            df.insert(loc=1, column=col + "_No", value=dateno_values)
        else:
            dateno_values = df[col]
            df.insert(loc=1, column=col + "_No", value=dateno_values)
    else:
        dateno_values = df[col + "_No"]

    # create TimeSeriesGroupSplit object and use .split to create our folds
    time_group_splitter = cv_scheme(
        n_splits=n_splits, **extra_constructor_params
    ).split(df, groups=dateno_values, **extra_params)

    # keep the data in list format
    cv_split_data = list(time_group_splitter)

    if not return_col:
        return cv_split_data
    else:
        return cv_split_data, dateno_values

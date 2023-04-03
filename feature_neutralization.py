import pandas as pd
import numpy as np

# to neutralize a column in a df by many other columns on a per-era basis
def neutralize(
    df,
    columns,
    extra_neutralizers=None,
    proportion=1.0,
    normalize=True,
    era_col="era"
):
    """
    ::param: df: pd.DataFrame()
    ::param: columns: the columns we need to neutralize
    ::param: extra_neutralizers: columns to neutralize against
    ::param: proportion: number between 0 and 1
    ::param: normalize: defaults to True
    ::param: era_col: our time column. Defaults to 'era'
    """
    if extra_neutralizers is None:
        extra_neutralizers = []
    unique_eras = df[era_col].unique()
    computed = []
    for u in unique_eras:
        print(u, end="\r")
        df_era = df[df[era_col] == u]
        scores = df_era[columns].values
        if normalize:
            scores2 = []
            for x in scores.T:
                x = (pd.Series(x).rank(method="first").values - .5) / len(x)
                scores2.append(x)
            scores = np.array(scores2).T
            extra = df_era[extra_neutralizers].values
            exposures = np.concatenate([extra], axis=1)
        else:
            exposures = df_era[extra_neutralizers].values

        scores -= proportion * exposures.dot(
            np.linalg.pinv(exposures.astype(np.float32)).dot(scores.astype(np.float32)))

        scores /= scores.std(ddof=0)

        computed.append(scores)

    return pd.DataFrame(np.concatenate(computed),
                        columns=columns,
                        index=df.index)

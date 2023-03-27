import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta, TH, FR
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import pickle


def start_end_date(df, date_col='date'):
    """

    :param date_col: the name of the column that measures time
    :param df: dataframe that must contain one column named 'date' in the datetime format
    :return: start date and end date as strings
    """
    if df[date_col].dtypes == 'datetime64[ns]':
        start_date = str(df[date_col].iloc[0].date().month) + '-' + str(df[date_col].iloc[0].date().day) + '-' + str(
            df[date_col].iloc[0].date().year)
        end_date = str(df[date_col].iloc[-1].date().month) + '-' + str(df[date_col].iloc[-1].date().day) + '-' + str(
            df[date_col].iloc[-1].date().year)
    else:
        start_date = df[date_col].iloc[0]
        end_date = df[date_col].iloc[-1]

    return start_date, end_date


def weeks_between_dates(d1, d2):
    """

    :param d1: first date in datetime format
    :param d2: second date in datetime format
    :return: weeks in integer format
    """
    monday1 = (d1 - timedelta(days=d1.weekday()))
    monday2 = (d2 - timedelta(days=d2.weekday()))

    weeks = (monday2 - monday1).days // 7

    return weeks


def plot_feature_importances(feature_names, model):
    plt.figure(figsize=(15, 3))
    plt.bar(feature_names, model.feature_importances_)
    plt.xticks(rotation=70)
    plt.show()


# From Jason Rosenfeld's notebook
# https://twitter.com/jrosenfeld13/status/1315749231387443202?s=20

def score(df, target_name, pred_name):
    """
    Takes df and calculates spearm correlation from pre-defined cols
    """
    # method="first" breaks ties based on order in array
    return np.corrcoef(
        df[target_name],
        df[pred_name].rank(pct=True, method="first")
    )[0, 1]


def run_analytics(era_scores, plot_figures=False):
    if plot_figures:
        print(f"Mean Correlation: {era_scores.mean():.4f}")
        print(f"Median Correlation: {era_scores.median():.4f}")
        print(f"Standard Deviation: {era_scores.std():.4f}")
        print('\n')
        print(f"Mean Pseudo-Sharpe: {era_scores.mean() / era_scores.std():.4f}")
        print(f"Median Pseudo-Sharpe: {era_scores.median() / era_scores.std():.4f}")
        print('\n')
    try:
        hit_rate = era_scores.apply(lambda x: np.sign(x)).value_counts()[1] / len(era_scores)
    except Exception as e:
        print('Exception_Error: there should be at least 3 unique validation date_col')

    if plot_figures:
        print(f'Hit Rate (% positive eras): {hit_rate:.2%}')
        print('\n')

    if plot_figures:
        era_scores.rolling(10).mean().plot(kind='line', title='Rolling Per Era Correlation Mean', figsize=(15,4))
        plt.axhline(y=0.0, color="r", linestyle="--")
        plt.show()

        era_scores.cumsum().plot(title='Cumulative Sum of Era Scores', figsize=(15,4))
        plt.axhline(y=0.0, color="r", linestyle="--")
        plt.show()

    return hit_rate


def features_targets_correlations(df, feature_cols, time_col, target, visualize=False):
    # calculate correlations between targets and features
    feature_corrs = (df.groupby(time_col).apply(lambda x: x[feature_cols].corrwith(x[target])))
    average_feature_corrs = feature_corrs.mean()
    average_feature_corrs_df = pd.DataFrame(average_feature_corrs, columns=['avg'])
    average_feature_corrs_df.sort_values(inplace=True, ascending=False, by='avg')

    if visualize:
        fig, ax = plt.subplots(1, 1, figsize=(20, 10))
        plt.xticks(rotation=90)
        sns.barplot(
            x='index',
            y='avg',
            data=average_feature_corrs_df.reset_index(),
            ax=ax
        )

    return average_feature_corrs_df


def rescale(prediction, targets_lst, feature_range=(0.000001, 0.999999), plot=True):
    for t in targets_lst:
        print(np.max(prediction[t]))
        print(np.min(prediction[t]))
        scaler = MinMaxScaler(feature_range=feature_range)
        prediction[t] = scaler.fit_transform(np.array(prediction[t]).reshape(-1, 1))

    if plot:
        for target in targets_lst:
            prediction[target].hist(bins=30)
            plt.show()

    print('Borders after rescaling')
    for t in targets_lst:
        print(np.max(prediction[t]))
        print(np.min(prediction[t]))

    return prediction


def save_obj(obj, name):
    """
    save and load dtypes object for reading objects(csvs, lists, models etc.)
    """
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


def check_enough_tickers(df, use_last_available=False, date_col='friday_date'):
    # check if for some reason last friday is missing and use another date instead
    last_friday = datetime.now() + relativedelta(weekday=FR(-1))
    last_friday = int(last_friday.strftime('%Y%m%d'))

    if len(df[df[date_col]==last_friday])<1000:
        last_friday_num_tickers = len(df[df[date_col]==last_friday])
        print(f'Last Friday had {last_friday_num_tickers} tickers')
        last_friday = datetime.now() + relativedelta(weekday=TH(-1))
        last_friday = int(last_friday.strftime('%Y%m%d'))
        print(f'We will be using last Thursday instead {last_friday}')
    else:
        print(f'last friday with enough tickers: {len(df[df[date_col]==last_friday])}')
    if use_last_available:
        dates_lst = df[date_col].unique().tolist()
        dates_lst.sort()
        last_friday = dates_lst[-1]
        print(f'We are using the last available date: {last_friday}')
    return last_friday


def check_if_live(
    sub,
    date_col
):
    print(f'Today: {int(datetime.now().strftime("%Y%m%d"))}')
    dates_lst = sub[date_col].unique()
    dates_lst.sort()
    print(f'Last available day {dates_lst[-1]}')
    sub.loc[sub[date_col]==dates_lst[-1],'data_type'] = 'live'
    return sub


def get_era_idx(df, col='era'):
    """
    get the indices of each era in a list format
    
    returns: [era1_idx_lst, era2_idx_lst, ... . eran_idx_lst]
    """
    era_lst = df[col].unique()
    era_idx = [df[df[col] == x].index for x in era_lst]
    return era_idx


def corr_score(df, pred_name, target_name='target', group_name='era'):
    # Check the per-era correlations on the validation set (out of sample)
    correlations = df.groupby(group_name).apply(lambda x: score(x, pred_name, target_name))
    return correlations


def sharpe_score(correlations):
    # Check the "sharpe" ratio on the validation set
    sharpe = correlations.mean() / correlations.std(ddof=0)
    return sharpe


def spearman(y_true, y_pred):
    return spearmanr(y_pred, y_true).correlation

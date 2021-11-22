import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt


def start_end_date(df):
    """

    :param df: dataframe that must contain one column named 'date' in the datetime format
    :return: start date and end date as strings
    """
    start_date = str(df['date'].iloc[0].date().month) + '-' + str(df['date'].iloc[0].date().day) + '-' + str(
        df['date'].iloc[0].date().year)
    end_date = str(df['date'].iloc[-1].date().month) + '-' + str(df['date'].iloc[-1].date().day) + '-' + str(
        df['date'].iloc[-1].date().year)
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
        df[[pred_name]].rank(pct=True, method="first")
    )[0, 1]


def run_analytics(era_scores):
    print(f"Mean Correlation: {era_scores.mean():.4f}")
    print(f"Median Correlation: {era_scores.median():.4f}")
    print(f"Standard Deviation: {era_scores.std():.4f}")
    print('\n')
    print(f"Mean Pseudo-Sharpe: {era_scores.mean() / era_scores.std():.4f}")
    print(f"Median Pseudo-Sharpe: {era_scores.median() / era_scores.std():.4f}")
    print('\n')
    hit_rate = era_scores.apply(lambda x: np.sign(x)).value_counts()[1] / len(era_scores)
    print(f'Hit Rate (% positive eras): {hit_rate:.2%}')

    # era_scores.rolling(10).mean().plot(kind='line', title='Rolling Per Era Correlation Mean', figsize=(15,4))
    # plt.axhline(y=0.0, color="r", linestyle="--"); plt.show()

    # era_scores.cumsum().plot(title='Cumulative Sum of Era Scores', figsize=(15,4))
    # plt.axhline(y=0.0, color="r", linestyle="--"); plt.show()

    return hit_rate

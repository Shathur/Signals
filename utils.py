from datetime import timedelta


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
    :return: weeks
    """
    monday1 = (d1 - timedelta(days=d1.weekday()))
    monday2 = (d2 - timedelta(days=d2.weekday()))

    weeks = (monday2 - monday1).days // 7

    return weeks

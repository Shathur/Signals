def start_end_date(df):
    start_date = str(df['date'].iloc[0].date().month) + '-' + str(df['date'].iloc[0].date().day) + '-' + str(df['date'].iloc[0].date().year)
    end_date = str(df['date'].iloc[-1].date().month) + '-' + str(df['date'].iloc[-1].date().day) + '-' + str(df['date'].iloc[-1].date().year)
    return start_date, end_date

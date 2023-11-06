import pandas as pd


def get_agg_dataframe(this_df):
    this_df['Value ($)'] = this_df['Value ($)'].astype(str).str.replace(',', '').astype('int64')
    this_df['Date'] = pd.DatetimeIndex(this_df['Date']).month

    Insider = this_df.groupby('Insider Trading').agg({'Value ($)': ['min', 'max', 'mean']})
    Relationship = this_df.groupby('Relationship').agg({'Value ($)': ['min', 'max', 'mean']})
    Transaction = this_df.groupby('Transaction').agg({'Value ($)': ['min', 'max', 'mean']})
    Month = this_df.groupby('Date').agg({'Value ($)': ['min', 'max', 'mean']})
    return [Insider, Relationship, Transaction, Month]
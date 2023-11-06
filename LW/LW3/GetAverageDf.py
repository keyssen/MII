import pandas as pd

from LW.LW1.GetDF import get_df


def get_average_df ():
    df = get_df()
    average_insider = df['Insider Trading'].value_counts().idxmax()
    average_relationship = df['Relationship'].value_counts().idxmax()
    average_date = pd.to_datetime(df['Date']).mean().date()
    average_transaction = df['Transaction'].value_counts().idxmax()
    average_cost = round(df['Cost'].mean(), 2)
    df['Shares'] = df['Shares'].str.replace(',', '').astype(int)
    average_shares = int(df['Shares'].mean())
    df['Value ($)'] = df['Value ($)'].str.replace(',', '').astype('int64')
    average_value = int(df['Value ($)'].mean())
    df['Shares Total'] = df['Shares Total'].str.replace(',', '').astype(int)
    average_sharesTotal = int(df['Shares Total'].mean())
    average_SEC = str(pd.to_datetime(df['SEC Form 4'], format='%b %d %I:%M %p').mean().strftime("%b %d %I:%M %p"))
    for i in range(10):
        new_row = {
            'Insider Trading': average_insider,
            'Relationship': average_relationship,
            'Date': average_date,
            'Transaction': average_transaction,
            'Cost': average_cost,
            'Shares': average_shares,
            'Value ($)': average_value,
            'Shares Total': average_sharesTotal,
            'SEC Form 4': average_SEC
        }
        df.loc[len(df)] = new_row
    return df
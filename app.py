from flask import Flask, render_template, request
import pandas as pd
import copy
import matplotlib.pyplot as plt
from io import BytesIO
import base64

app = Flask(__name__)
df = pd.read_csv('TSLA.csv')
new_df = copy.deepcopy(df)
chart_df = copy.deepcopy(df)

def get_column_description(df):
    result = pd.DataFrame(
        {'columns': df.columns, 'types': df.dtypes, 'empty': df.isnull().sum(), 'fill': df.notnull().sum()})
    list = result.values.tolist()
    list.sort(key=lambda x: x[0])
    return list

def get_df():
    return pd.read_csv('TSLA.csv')

def format_number(num):
    return '{:,}'.format(int(num)).replace(',', ' ')

def get_average_df ():
    df = get_df()
    average_insider = df['Insider Trading'].value_counts().idxmax()
    average_relationship = df['Relationship'].value_counts().idxmax()
    average_date = pd.to_datetime(df['Date']).mean().date()
    average_transaction = df['Transaction'].value_counts().idxmax()
    average_cost = round(df['Cost'].mean(), 2)
    df['Shares'] = df['Shares'].str.replace(',', '').astype(int)
    average_shares = int(df['Shares'].mean())
    df['Value ($)'] = df['Value ($)'].astype(str).str.replace(',', '').astype('int64')
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

def get_chart(first_dataFrame, second_dataFrame):
    join_dataFrame = pd.concat([first_dataFrame, second_dataFrame], axis=0)
    plt.plot(join_dataFrame)
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plt.clf()
    graph_url = base64.b64encode(img.getvalue()).decode()
    return graph_url

def get_agg_dataframe(this_df):
    this_df['Value ($)'] = this_df['Value ($)'].astype(str)
    this_df['Value ($)'] = this_df['Value ($)'].str.replace(',', '').astype('int64')
    this_df['Date'] = pd.DatetimeIndex(this_df['Date']).month

    Insider = this_df.groupby('Insider Trading').agg({'Value ($)': ['min', 'max', 'mean']})
    Relationship = this_df.groupby('Relationship').agg({'Value ($)': ['min', 'max', 'mean']})
    Transaction = this_df.groupby('Transaction').agg({'Value ($)': ['min', 'max', 'mean']})
    Month = this_df.groupby('Date').agg({'Value ($)': ['min', 'max', 'mean']})
    return [Insider, Relationship, Transaction, Month]

@app.route('/', methods=['GET', 'POST'])
def table():
    column_descriptions = get_column_description(df)
    if request.method != "POST":
        return render_template('index.html',
                               tables=[df.iloc[0:15, 0:9].to_html(classes="container table table-bordered mystyle")],
                               column_descriptions=column_descriptions)

    minCol = int(request.form["minCol"])
    maxCol = int(request.form["maxCol"])
    minRow = int(request.form["minRow"])
    maxRow = int(request.form["maxRow"])

    return render_template('index.html',
                           tables=[df.iloc[minRow:maxRow + 1, minCol - 1:maxCol].to_html(classes="container table table-bordered mystyle")],
                           column_descriptions=column_descriptions)

@app.route('/aggTables', methods=['GET', 'POST'])
def agg_ables():
    agg_tables = get_agg_dataframe(get_df())
    result_list = [item.map(format_number).to_html(classes="container table table-bordered mystyle") for item in agg_tables]
    return render_template('aggTables.html',
                           agg_tables = result_list)

@app.route('/updateAggTables', methods=['GET', 'POST'])
def update_agg_tables():
    agg_tables = get_agg_dataframe(get_average_df())
    result_list = [item.map(format_number).to_html(classes="container table table-bordered mystyle") for item in agg_tables]
    return render_template('aggTables.html',
                           agg_tables = result_list)

@app.route('/chart', methods=['GET', 'POST'])
def chart():
    agg_tables = get_agg_dataframe(get_df())
    update_agg_tables = get_agg_dataframe(get_average_df())

    fig, axs = plt.subplots(4, 1)
    fig.set_figwidth(15)
    fig.set_figheight(30)

    for index, (item1, item2) in enumerate(zip(agg_tables, update_agg_tables)):
        join_agg_table = pd.concat([item1, item2], axis=0)
        axs[index].plot(join_agg_table)
        axs[index].set_title(join_agg_table.index.name)
        # axs[index].set_yscale('log')
        axs[index].legend(['Value ($) min', 'Value ($) max', 'Value ($) mean'])

    plt.tight_layout()
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.clf()
    return render_template('chart.html',
                           image=image_base64)


if __name__ == "__main__":
    app.run(debug=True, host="localhost", port=int("5000"))

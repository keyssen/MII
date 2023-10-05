from flask import Flask, render_template, request
import pandas as pd
import copy

app = Flask(__name__)
df = pd.read_csv('TSLA.csv')
new_df = copy.deepcopy(df)

def get_column_description(df):
    result = pd.DataFrame(
        {'columns': df.columns, 'types': df.dtypes, 'empty': df.isnull().sum(), 'fill': df.notnull().sum()})
    list = result.values.tolist()
    list.sort(key=lambda x: x[0])
    return list

def format_number(num):
    return '{:,}'.format(int(num)).replace(',', ' ')

@app.route('/', methods=['GET', 'POST'])
def table():
    column_descriptions = get_column_description(df)

    new_df['Value ($)'] = new_df['Value ($)'].astype(str)
    new_df['Value ($)'] = new_df['Value ($)'].str.replace(',', '').astype('int64')
    new_df['Date'] = pd.DatetimeIndex(df['Date']).month


    Insider = new_df.groupby('Insider Trading').agg({'Value ($)': ['min', 'max', 'mean']})
    Insider = Insider.applymap(format_number)
    Relationship = new_df.groupby('Relationship').agg({'Value ($)': ['min', 'max', 'mean']})
    Relationship = Relationship.applymap(format_number)
    Transaction = new_df.groupby('Transaction').agg({'Value ($)': ['min', 'max', 'mean']})
    Transaction = Transaction.applymap(format_number)
    Month = new_df.groupby('Date').agg({'Value ($)': ['min', 'max', 'mean']})
    Month = Month.applymap(format_number)
    # print(Insider)
    # print(Relationship)
    # print(Transaction)
    # print(Month)
    if request.method != "POST":
        return render_template('index.html',
                               tables=[df.iloc[0:15, 0:9].to_html(classes="container table table-bordered mystyle")],
                               Insider=[Insider.to_html(classes="container table table-bordered mystyle")],
                               Relationship=[Relationship.to_html(classes="container table table-bordered mystyle")],
                               Transaction=[Transaction.to_html(classes="container table table-bordered mystyle")],
                               Month=[Month.to_html(classes="container table table-bordered mystyle")],
                               column_descriptions=column_descriptions)

    minCol = int(request.form["minCol"])
    maxCol = int(request.form["maxCol"])
    minRow = int(request.form["minRow"])
    maxRow = int(request.form["maxRow"])

    return render_template('index.html',
                           tables=[df.iloc[minRow:maxRow + 1, minCol - 1:maxCol].to_html(classes="container table table-bordered mystyle")],
                           Insider=[Insider.to_html(classes="container table table-bordered mystyle")],
                           Relationship=[Relationship.to_html(classes="container table table-bordered mystyle")],
                           Transaction=[Transaction.to_html(classes="container table table-bordered mystyle")],
                           Month=[Month.to_html(classes="container table table-bordered mystyle")],
                           column_descriptions=column_descriptions)


if __name__ == "__main__":
    app.run(debug=True, host="localhost", port=int("5000"))

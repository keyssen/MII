import io

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

def create_bar_chart(task_title, task_data1, task_data2, y_label):
    age_groups = task_data1.index
    min_value = task_data1[y_label]['min'].values
    mean_value = task_data1[y_label]['mean'].values
    max_value = task_data1[y_label]['max'].values
    min_value2 = task_data2[y_label]['min'].values
    mean_value2 = task_data2[y_label]['mean'].values
    max_value2 = task_data2[y_label]['max'].values

    x = range(len(age_groups))
    width = 0.2

    plt.bar(x, min_value, width, alpha=0.5, label='Minimum ' + y_label)
    plt.bar([val + width for val in x], mean_value, width, alpha=0.5, label='Mean ' + y_label)
    plt.bar([val + width * 2 for val in x], max_value, width, alpha=0.5, label='Maximum ' + y_label)
    plt.bar([val + width * 3 for val in x], min_value2, width, alpha=0.5, label='New Minimum ' + y_label)
    plt.bar([val + width * 4 for val in x], mean_value2, width, alpha=0.5, label='New Mean ' + y_label)
    plt.bar([val + width * 5 for val in x], max_value2, width, alpha=0.5, label='New Maximum ' + y_label)

    plt.ylabel(y_label)
    plt.xticks([val + width for val in x], age_groups)
    plt.legend()
    plt.title(task_title)

@app.route('/chart', methods=['GET', 'POST'])
def chart():
    agg_tables = get_agg_dataframe(get_df())
    update_agg_tables = get_agg_dataframe(get_average_df())

    task_images = []

    for index, (item1, item2) in enumerate(zip(agg_tables, update_agg_tables)):
        plt.figure(figsize=(20, 6))
        plt.subplot(1, 1, 1)

        create_bar_chart(item1.index.name, item1, item2, 'Value ($)')
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format="png")
        img_buffer.seek(0)
        img_base64 = base64.b64encode(img_buffer.read()).decode("utf-8")

        task_images.append((f"data:image/png;base64,{img_base64}"))
        plt.clf()

    join = pd.concat([agg_tables[0], update_agg_tables[0]], axis=0)
    plt.boxplot(join)
    plt.yscale('log')
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format="png")
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.read()).decode("utf-8")

    # Добавляем изображения в список задач с номером задачи
    task_images.append((f"data:image/png;base64,{img_base64}"))
    plt.clf()

    return render_template('chart.html',
                           task_images=task_images)


if __name__ == "__main__":
    app.run(debug=True, host="localhost", port=int("5000"))

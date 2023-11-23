import io

import numpy as np
from flask import Flask, render_template, request, session
import pandas as pd
import copy
import matplotlib.pyplot as plt
import base64

from LW.LW1.GetColumnDescription import get_column_description
from LW.LW1.GetDF import get_df
from LW.LW2.GetAggDataframe import get_agg_dataframe
from LW.LW3.CreateBarChart import create_bar_chart
from LW.LW3.FormatNumber import format_number
from LW.LW3.GetAverageDf import get_average_df
from LW.LW4.SiteSearch import SiteSearch
from LW.LW5.LinearRegression import linear_regression
from LW.LW6.decisionView import decision_View
from LW.LW7.clasterization import clasterization

app = Flask(__name__)
app.secret_key = 'super secret key'
app.config['SESSION_TYPE'] = 'filesystem'
df = pd.read_csv('TSLA.csv')
new_df = copy.deepcopy(df)
chart_df = copy.deepcopy(df)
search_engine = SiteSearch()
search_engine.add("https://www.kaggle.com/datasets/ankanhore545/100-highest-valued-unicorns", ["Company", "Valuation", "Country", "State", "City", "Industries", "Founded Year", "Name of Founders", "Total Funding", "Number of Employees"])
search_engine.add("https://www.kaggle.com/datasets/ilyaryabov/tesla-insider-trading", ["Insider Trading", "Relationship", "Date", "Transaction", "Cost", "Shares", "Value", "Shares Total", "SEC Form 4"])
search_engine.add("https://www.kaggle.com/datasets/sameepvani/nasa-nearest-earth-objects", ["NASA", "est_diameter_min", "est_diameter_max", "relative_velocity", "miss_distance", "orbiting_body", "sentry_object", "absolute_magnitude", "hazardous"])





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
def agg_tables():
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

    task_images = []

    for index, (item1, item2) in enumerate(zip(agg_tables, update_agg_tables)):
        plt.figure(figsize=(20, 6))
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

@app.route('/linearRegression', methods=['GET'])
def linearRegression():
    plt.figure(figsize=(20, 6))
    listMessages = list(linear_regression())
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format="png")
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.read()).decode("utf-8")
    linear_image = (f"data:image/png;base64,{img_base64}")
    plt.clf()
    return render_template('LW5.html', linear_image=linear_image, listMessages=listMessages)

@app.route('/findURL', methods=['POST'])
def findURL():
    session['word'] = request.form["word"]
    if (search_engine.contains(session.get('word'))):
        links = search_engine.find_url(session.get('word'))
        return render_template('findURL.html', condition = True)
    return render_template('findURL.html')

@app.route('/findURL', methods=['GET'])
def get_page_findURL():
    return render_template('findURL.html')

@app.route('/true_findURL', methods=['GET'])
def true_findURL():
    links = search_engine.find_url(session.get('word'))
    return render_template('findURL.html', links = links)


@app.route('/decisionTree', methods=['GET'])
def decisionTree():
    plt.figure(figsize=(20, 6))
    accuracy = decision_View()
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format="png")
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.read()).decode("utf-8")
    decision_tree_image = (f"data:image/png;base64,{img_base64}")
    plt.clf()
    return render_template('LW6.html', decision_tree_image=decision_tree_image, accuracy=accuracy)

@app.route('/clast', methods=['GET', 'POST'])
def clast():
    plt.figure(figsize=(20, 6))
    data = get_df()
    data['Transaction'] = data['Transaction'].replace({'Sale': 1, 'Option Exercise': 2})
    data['Transaction'] = data['Transaction']
    data['Value ($)'] = data['Value ($)'].str.replace(',', '').astype('int64')
    clast = clasterization(data.iloc[:, [3, 6]].to_numpy(), data["Transaction"].nunique())
    clast.clast()
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format="png")
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.read()).decode("utf-8")
    clast_image = (f"data:image/png;base64,{img_base64}")
    plt.clf()
    return render_template('LW7.html', clast_image=clast_image,)



if __name__ == "__main__":
    app.run(debug=True, host="localhost", port=int("5000"))

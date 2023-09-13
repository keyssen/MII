from flask import Flask, render_template, request
import pandas as pd
import numpy as np
app = Flask(__name__)

def get_column_description(df):
    print(df.describe())
    result = pd.DataFrame({'types': df.dtypes,'empty': df.isnull().sum(), 'fill': df.notnull().sum()})
    return result
@app.route('/', methods=['GET', 'POST'])
def table():
    params = {
        "data": table,
        "start_index": 0
    }

    df = pd.read_csv('TSLA.csv')
    # length = df.shape
    # print(df.shape[0]) # строк
    # print(df.shape[1])  # столбцов


    column_descriptions = get_column_description(df)

    types = df.dtypes
    df.describe()
    if request.method == "POST":
        minCol = int(request.form["minCol"])
        maxCol = int(request.form["maxCol"])
        minRow = int(request.form["minRow"])
        maxRow = int(request.form["maxRow"])
        return render_template('index.html', tables=[df.iloc[minRow:maxRow+1, minCol-1:maxCol].to_html(classes="container table"),
        column_descriptions.to_html(classes="container table")])
    else:
        return render_template('index.html', tables=[df.to_html(classes="container table"),
        column_descriptions.to_html(classes="container table")])

@app.route('/about')
def about():
    return "about"

if __name__ == "__main__":
    app.run(debug=True, host="localhost", port=int("5000"))
from flask import Flask, render_template, request
import pandas as pd

app = Flask(__name__)
df = pd.read_csv('TSLA.csv')

def get_column_description(df):
    result = pd.DataFrame({'columns': df.columns, 'types': df.dtypes,'empty': df.isnull().sum(), 'fill': df.notnull().sum()})
    list = result.values.tolist()
    list.sort(key = lambda x:x[0])
    return list

@app.route('/', methods=['GET', 'POST'])
def table():
    column_descriptions = get_column_description(df)
    if request.method == "POST":
        minCol = int(request.form["minCol"])
        maxCol = int(request.form["maxCol"])
        minRow = int(request.form["minRow"])
        maxRow = int(request.form["maxRow"])
        return render_template('index.html', tables=[
            df.iloc[minRow:maxRow + 1, minCol - 1:maxCol].to_html(classes="container table table-bordered mystyle")],column_descriptions=column_descriptions)
    else:
        return render_template('index.html', tables=[df.iloc[0:15, 0:9].to_html(classes="container table table-bordered mystyle")], column_descriptions=column_descriptions)

if __name__ == "__main__":
    app.run(debug=True, host="localhost", port=int("5000"))
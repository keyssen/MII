import os

import numpy

import pandas as pd

import matplotlib.pyplot as plt



# INCH = 25.4

# def create_plot_jpg(df: pd.DataFrame):
#     # набор атрибутов - независимых переменных - площадь
#     column_value = df["Value ($)"].array
#
#     # набор меток - зависимых переменных, значение которых требуется предсказать - выручка
#     column_transaction = df["Transaction"].array
#
#     # делим датафрейм на набор тренировочных данных и данных для тестов, test_size содержит определние соотношения этих наборов
#     X_train, X_test, y_train, y_test = train_test_split(column_value, column_transaction, test_size=0.01, random_state=0)
#
#     regressor = LinearRegression()
#
#     # X_train = X_train.reshape(-1, 1)
#     # X_test = X_test.reshape(-1, 1)
#
#     regressor.fit(X_train, y_train)
#
#     # массив numpy, который содержит все предсказанные значения для входных значений в серии X_test
#     y_pred = regressor.predict(X_test)
#
#     df.plot(x='Store_Sales', y='Store_Area', style='o')
#
#     plt.title('Зависимость продаж от площади магазина')
#     plt.xlabel('Продажи')
#     plt.ylabel('Площадь')
#
#
#     listMessages = ['Средняя абсолютная ошибка (MAE): ' + str(metrics.mean_absolute_error(y_test, y_pred)),
#                     'Среднеквадратичная ошибка (MSE): ' + str(metrics.mean_squared_error(y_test, y_pred)),
#                     'Среднеквадратичная ошибка (RMSE): ' + str(np.sqrt(metrics.mean_squared_error(y_test, y_pred)))]
#
#     return listMessages

# def linear_regression(_x, _y):
def linear_regression():
    this_df = pd.read_csv('TSLA.csv')
    X = numpy.array(this_df['Value ($)'])
    Y = numpy.array(this_df['Transaction'])
    n = X.size
    sumX = sum(X)
    sumY = sum(Y)
    sumXY = sum(X*Y)
    sumXX = sum(X*X)
    b1 = (sumXY - (sumY * sumX) / n) / (sumXX - sumX * sumX / n)
    b0 = (sumY - b1 * sumX) / n
    plt.scatter(X, Y, alpha=0.8)
    plt.axline(xy1=(0, b0), slope=b1, color='r', label=f'$y = {b1:.2f}x {b0:+.2f}$')
    plt.legend()
    plt.show()
    print(X, Y)

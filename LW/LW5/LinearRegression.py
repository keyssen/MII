import numpy
import pandas as pd
from matplotlib import pyplot as plt


def linear_regression():
    this_df = pd.read_csv('TSLA.csv')
    this_df['Value ($)'] = this_df['Value ($)'].str.replace(',', '').astype('int64')
    this_df['Transaction'] = this_df['Transaction'].replace({'Sale': 1, 'Option Exercise': 2})
    # unique_dict = {name: number for number, name in enumerate(this_df['Transaction'])}
    X = numpy.array(this_df['Value ($)'])
    Y = numpy.array(this_df['Transaction'])
    n = X.size
    # Разделение данных на обучающий и тестовый наборы
    n_test = int(n * 0.2)
    n_train = n - n_test
    X_train, Y_train = X[:n_train], Y[:n_train]
    X_test, Y_test = X[n_train:], Y[n_train:]
    sumY_train = sum(Y_train)
    sumX_train = sum(X_train)
    sumXY_train = sum(X_train * Y_train)
    sumXX_train = sum(X_train * X_train)

    b1 = (sumXY_train - (sumY_train * sumX_train) / n_train) / (sumXX_train - sumX_train * sumX_train / n_train)
    b0 = (sumY_train - b1 * sumX_train) / n_train

    # Построение модели на обучающем наборе
    plt.scatter(X_train, Y_train, alpha=0.8)
    plt.axline(xy1=(0, b0), slope=b1, color='r', label=f'$y = {b1:.2f}x {b0:+.2f}$ (Training)')

    # Оценка производительности модели на тестовом наборе
    Y_pred = b0 + b1 * X_test
    mse = sum((Y_test - Y_pred)**2) / n_test

    plt.scatter(X_test, Y_test, alpha=0.8, color='g')

    plt.legend()
    return mse, r_squared(Y_test, Y_pred)

def r_squared(y_true, y_pred):
    # Вычисляем среднее значение целевой переменной
    mean_y_true = numpy.mean(y_true)

    # Вычисляем сумму квадратов отклонений от среднего
    ss_total = numpy.sum((y_true - mean_y_true) ** 2)

    # Вычисляем сумму квадратов остатков
    ss_residual = numpy.sum((y_true - y_pred) ** 2)

    # Вычисляем коэффициент детерминации
    r_squared = 1 - (ss_residual / ss_total)

    return r_squared
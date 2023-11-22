import math

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np

from LW.LW6.DecisionTree import DecisionTree

this_df = pd.read_csv('../../TSLA.csv')
this_df['Value ($)'] = this_df['Value ($)'].str.replace(',', '').astype('int64')
this_df['Transaction'] = this_df['Transaction'].replace({'Sale': 1, 'Option Exercise': 2})
this_df['Date'] = pd.to_datetime(this_df['Date']).dt.month
combined_array = this_df[['Date', 'Value ($)']].values.tolist()
X = np.array(combined_array)
y = np.array(this_df['Transaction'].tolist())

# data = datasets.load_breast_cancer()
# X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)

clf = DecisionTree(max_depth=4)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

def accuracy(y_test, y_pred):
    return np.sum(y_test == y_pred) / len(y_test)

acc = accuracy(y_test, predictions)
print(acc)

clf.print_tree()

x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, math.log2(X_train[:, 1].max() + 1)
result_natural_log = np.log2(X_train[:, 1])
xx, yy = np.meshgrid(np.arange(x_min, x_max),
                     np.arange(y_min, y_max))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure(figsize=(10,6))
plt.pcolormesh(xx, yy, Z, cmap='coolwarm', alpha=0.2)
# Более тёплые цвета (красные/оранжевые) будут соответствовать значениям в y_train,
plt.scatter(X_train[:, 0], result_natural_log, c=y_train, cmap='coolwarm')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("DecisionTree Classification")
plt.show()
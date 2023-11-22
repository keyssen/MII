import math

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree


this_df = pd.read_csv('../../TSLA.csv')
this_df['Value ($)'] = this_df['Value ($)'].str.replace(',', '').astype('int64')
this_df['Transaction'] = this_df['Transaction'].replace({'Sale': 1, 'Option Exercise': 2})
this_df['Date'] = pd.to_datetime(this_df['Date']).dt.month
combined_array = this_df[['Date', 'Value ($)']].values.tolist()

feature_names = ['Date', 'Value ($)']
X = np.array(combined_array)
y = this_df['Transaction'].tolist()
print(combined_array)
clf = DecisionTreeClassifier(max_depth=3)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2
)

clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
tree_rules = export_text(clf, feature_names=feature_names)



def accuracy(y_test, y_pred):
    return np.sum(y_test == y_pred) / len(y_test)

acc = accuracy(y_test, predictions)
print(acc)
print(tree_rules)

tree.plot_tree(clf)

x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, math.log10(X_train[:, 1].max() + 1)
xx, yy = np.meshgrid(np.arange(x_min, x_max),
                     np.arange(y_min, y_max))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.figure(figsize=(10,6))
plt.pcolormesh(xx, yy, Z, cmap='coolwarm', alpha=0.2)
# Plot the data points
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='coolwarm')
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.title("DecisionTree Classification")
plt.show()


# #
# import numpy as np
# from matplotlib import pyplot as plt
# from sklearn.datasets import make_classification
# from sklearn.metrics import classification_report
#
# from LW.LW6.DecisionTree import DecisionTree
#
# X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, n_informative=2,
#                            random_state=1, n_clusters_per_class=1)
# clf = DecisionTree(max_depth=3)
# print(X)
# print(y)
# clf.fit(X, y)
# # Прогнозируем метки классов на тестовой выборке
# y_pred = clf.predict(X)
# print("y_pred")
# print(y_pred)
# print("y_pred")
# print(classification_report(y, y_pred))
# clf.visualize_tree().render(directory='image').replace('\\', '/')
#
# # x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
# # y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
# # xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
# #                      np.arange(y_min, y_max, 0.02))
# # Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
# # Z = Z.reshape(xx.shape)
# # plt.figure(figsize=(10,6))
# # plt.pcolormesh(xx, yy, Z, cmap='coolwarm', alpha=0.2)
# # # Plot the data points
# # plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm')
# # plt.xlim(xx.min(), xx.max())
# # plt.ylim(yy.min(), yy.max())
# # plt.title("DecisionTree Classification")
# # plt.show()
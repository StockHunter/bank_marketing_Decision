import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.tree import export_graphviz
import pydotplus
from IPython.display import Image

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
submit_sample = pd.read_csv("submit_sample.csv", header=None)

trainX = train.iloc[:, 0:17]
y = train["y"]

testX = test.copy()

trainX = pd.get_dummies(trainX)
testX = pd.get_dummies(testX)

clf1 = DT(max_depth=2, min_samples_leaf=500)

clf1.fit(trainX, y)

export_graphviz(clf1, out_file="tree.dot", feature_names=trainX.columns, class_names=["0", "1"], filled=True, rounded=True)
g = pydotplus.graph_from_dot_file(path="tree.dot")
print(Image(g.create_png()))

pred = clf1.predict_proba(testX)
pred = pred[:, 1]
submit_sample[1] = pred

submit_sample.to_csv("submit_bank.csv",index=None, header=None)

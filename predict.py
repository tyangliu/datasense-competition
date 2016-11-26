import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm

df = pd.get_dummies(pd.read_csv("training.csv", header=0))
df_X = df.drop(["morethan60kyr"], axis=1)
df_Y = df[["morethan60kyr"]]

X_train, X_test, Y_train, Y_test = train_test_split(df_X, df_Y, test_size=0.2)
print(Y_test.as_matrix()[:,0])
clf = svm.SVC(kernel='rbf', C=1).fit(X_train.as_matrix(), Y_train.as_matrix()[:,0])
sc = clf.score(X_test.as_matrix(), Y_test.as_matrix()[:,0])
print(sc)


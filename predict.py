import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

df = pd.read_csv("training.csv", header=0)
df.drop(["CASEID"], axis=1, inplace=True)
df = pd.get_dummies(df)
df_X = df.drop(["morethan60kyr"], axis=1)
df_Y = df[["morethan60kyr"]]

X_train, X_test, Y_train, Y_test = train_test_split(df_X, df_Y, test_size=0.1)
X_tr = X_train.as_matrix()
Y_tr = Y_train.as_matrix()[:,0]
X_te = X_test.as_matrix()
Y_te = Y_test.as_matrix()[:,0]

clf = GradientBoostingClassifier(
    n_estimators=900,
    learning_rate=0.05,
    min_samples_split=1200,
    min_samples_leaf=60,
    max_depth=9,
    max_features='sqrt',
    subsample=0.8,
    random_state=10
).fit(X_tr, Y_tr)

print(clf.score(X_te, Y_te))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.feature_extraction.text import  CountVectorizer

df = pd.read_csv("training.csv", header=0)
df = df.drop(["CASEID"], axis=1)
df = pd.get_dummies(df)
df_X = df.drop(["morethan60kyr"], axis=1)
df_Y = df[["morethan60kyr"]]

X_train, X_test, Y_train, Y_test = train_test_split(df_X, df_Y, test_size=0.4)
X_tr = X_train.as_matrix()
Y_tr = Y_train.as_matrix()[:,0]
X_te = X_test.as_matrix()
Y_te = Y_test.as_matrix()[:,0]

clf2 = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=1.0,
    max_depth=1,
    random_state=0
).fit(X_tr, Y_tr)

print(clf2.score(X_te, Y_te))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier

df = pd.read_csv("training.csv", header=0)
df.drop(["CASEID"], axis=1, inplace=True)
df = pd.get_dummies(df)
df_X = df.drop(["morethan60kyr"], axis=1)
df_Y = df[["morethan60kyr"]]

df_test = pd.read_csv("test.csv", header=0)
df_test.drop(["CASEID"], axis=1, inplace=True)
df_test = pd.get_dummies(df_test)

X_train, X_test, Y_train, Y_test = train_test_split(df_X, df_Y, test_size=0.2)
X_tr = X_train.as_matrix()
Y_tr = Y_train.as_matrix()[:,0]
X_te = X_test.as_matrix()
Y_te = Y_test.as_matrix()[:,0]

clf = GradientBoostingClassifier(
    n_estimators=2000,
    learning_rate=0.03,
    min_samples_split=1200,
    min_samples_leaf=60,
    max_depth=12,
    max_features='sqrt',
    subsample=0.8,
    random_state=10
)

# print(clf.fit(X_tr, Y_tr).score(X_te, Y_te))

# scores = cross_val_score(clf, df_X, df_Y.as_matrix()[:,0], cv=10)
# print("Accuracy: %0.5f (+/- %0.5f)" % (scores.mean(), scores.std() * 2))
final_clf = clf.fit(df_X, df_Y.as_matrix()[:,0])
predictions = clf.predict(df_test)
np.savetxt("predictions.csv", predictions, delimiter=",")

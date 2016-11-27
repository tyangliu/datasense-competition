import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("training.csv", header=0)
df.drop(["CASEID"], axis=1, inplace=True)

df_pos = df[df['morethan60kyr'] == True]
df_neg = df[df['morethan60kyr'] == False]

fig = plt.figure()

ax = fig.add_subplot(111)
ax2 = ax.twinx()

width = 0.4

df_pos.mtnno.value_counts().plot(color='pink', kind='bar', ax=ax, width=width, position=1)
df_neg.mtnno.value_counts().plot(color='orange', kind='bar', ax=ax2, width=width, position=0)

ax.set_ylabel('Over')
ax.set_ylabel('Under')

plt.show()

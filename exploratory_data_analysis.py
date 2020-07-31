#%% Importing modules
import pandas as pd
import numpy as np
#%% Reading Data
df = pd.read_csv("mushrooms_fixed.csv")
df.head() # Preview Data
# %% Splitting df
df_length = len(df) # 8124
Y = df["class"]
X_columns = [x for x in df.columns if x != "class"] 
X = df[X_columns]
# %%  Initializing K Fold
from sklearn.model_selection import KFold
kf = KFold()
print(list(kf.split(X,Y))[0]) # to understand how KFold works

for train_index, test_index in kf.split(X,Y):
    x_train, y_train = X[train_index], Y[train_index]
    x_test, y_test = X[test_index], Y[test_index]

# %%

#%% Importing modules
import pandas as pd
import numpy as np
#%% Reading Data - Start getting rid of this when submitting code
df = pd.read_csv("mushrooms_fixed.csv")
df.head() # Preview Data
#%% Data Treatment - From Ashley's Code
for i in list(df.columns):
    print(i,"is",str(type(df[i][0]))) # print out 
for i in list(df.columns):
    print(i,"contains the values: \n",str(df[i].unique()))
for i in list(df.columns):
    print(i,"contains the values: \n",str(df[i].value_counts()))

df = df[df['cap.shape'] != 'c']
df = df[df['cap.surface'] != 'g']
df = df[df['stalk.color.above.ring'] != 'y']
df = df[df['veil.color'] != 'y']

print("After cleaning. Rows =",str(len(df))) # 8111
# %% Splitting df
Y = df["class"]
X_columns = [x for x in df.columns if x != "class"] 
X = df[X_columns]

# %%  Initializing K Fold - test code to understand
from sklearn.model_selection import KFold
kf = KFold()
print(list(kf.split(X,Y))[0]) # to understand how KFold works
x_np = X.to_numpy()
y_np = Y.to_numpy()
for train_index, test_index in kf.split(X,Y):
    x_train, y_train = X.loc[train_index], Y.loc[train_index]
    x_test, y_test = X.loc[test_index], Y.loc[test_index]
# %% Remove all code before this and import this file separately in main ML doc
# %%
class CV:
    '''
    A cross-validation core class function built from scratch

    input parameters: 
    :parameter - X, the entire set of data for the predictor variables, df
    :parameter - Y, the entire set of data for the response variables, df
    '''
    def __init__(self,X,Y,shuffle=True): # only require x and y for supervised learning
        if len(X) != len(Y):
            raise Exception("X and Y must be same length")
        self.n = len(X)
        seed = np.random.randint(0, 1000000) # create random seed
        np.random.seed(seed) # set seed
        if shuffle:
            self._xseed = seed
            np.random.shuffle(X)
            self.x = X
        else:
            self._x = X
            self._xseed = False
        seed = np.random.randint(0, 1000000) # create random seed
        np.random.seed(seed) # set seed
        if shuffle:
            self._yseed = seed
            np.random.shuffle(Y) 
            self.y = Y
        else:
            self._y = Y
            self._yseed = False # False if no shuffling

    def get_xseed(self):
        return self._xseed
    
    def get_yseed(self):
        return self._yseed

    def split(self,n_splits=10):
        '''
        Splits the X and Y data into a number of splits
        returns dataframes/lists equal to the number of splits for training and testing
        '''
        test_indices = []
        train_indices = []
        split_size = int(self.n/n_splits)
        for n in range(0,n_splits):
            test_index = list(range(n*split_size,(n+1)*split_size)) # quik mafs
            train_index = [num for num in range(0,self.n) if num not in test_index]
            test_indices.append(test_index)
            train_indices.append(train_index)
        return train_indices, test_indices
        
            
        






'''
kfold(training) ----> [[x_t,y_t,x,y] [] [] [] [] [] [] [] []]

# FOR ONE FOLD
x_train 
y_train
x_test
y_test
rndfr = RandomForest(x,y)
rndfr.predict(x_test) --> y_predict
y_test # true value
r2_list = r2_score(y_predict,y_test)

# Calculate how well our model did on all the folds
r2_avg = sum(r2_list)/len(r2_list)

'''

# %%

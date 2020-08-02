
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
        seed = np.random.randint(0,100000)
        self._seed = seed
        self.n = len(X)
        if shuffle:
            x = X.sample(frac=1,random_state=seed)
            y = Y.sample(frac=1,random_state=seed)
        self.x,self.y = (x.reset_index(drop=True),y.reset_index(drop=True))


    def get_seed(self):
        return self._seed

    def split(self,n_splits=10):
        '''
        Splits the X and Y data into a number of splits
        returns dataframes/lists equal to the number of splits for training and testing
        '''
        test_indices = []
        train_indices = []
        self.split_size = int(self.n/n_splits)
        for n in range(0,n_splits):
            test_index = list(range(n*self.split_size,(n+1)*self.split_size)) # quik mafs
            train_index = [num for num in range(0,self.n) if num not in test_index]
            test_indices.append(test_index)
            train_indices.append(train_index)
        return train_indices, test_indices
#%%
if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    #Reading Data - Start getting rid of this when submitting code
    df = pd.read_csv("mushrooms_fixed.csv")
    df.head() # Preview Data
    #Data Treatment - From Ashley's Code
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
    df.reset_index(drop=True,inplace=True)
    print("After cleaning. Rows =",str(len(df))) # 8111
    # Splitting df
    Y = df["class"]
    X_columns = [x for x in df.columns if x != "class"] 
    X = df[X_columns]
    cv=CV(X,Y)
    print("Training sets")
    print(f"Length of each traning set (number of splits = 3):{len(cv.split(n_splits=3)[0][0])} ")
    print("Testing set")
    print(f"Length of each traning set (number of splits = 3):{len(cv.split(n_splits=3)[1][0])} ")



# %%
        # self.n = len(X)
        # seed = np.random.randint(0, 1000000) # create random seed
        # np.random.seed(seed) # set seed
        # if shuffle:
        #     self._xseed = seed
        #     np.random.shuffle(X)
        #     self.x = X
        # else:
        #     self._x = X
        #     self._xseed = False
        # seed = np.random.randint(0, 1000000) # create random seed
        # np.random.seed(seed) # set seed
        # if shuffle:
        #     self._yseed = seed
        #     np.random.shuffle(Y) 
        #     self.y = Y
        # else:
        #     self._y = Y
        #     self._yseed = False # False if no shuffling
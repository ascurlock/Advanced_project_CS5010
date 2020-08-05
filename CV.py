import pandas as pd
import numpy as np
class CV:
    '''
    A cross-validation core class built from scratch

    input parameters: 
    :parameter - X, the entire set of data for the predictor variables, df
    :parameter - Y, the entire set of data for the response variables, df
    '''
    def __init__(self,X,Y,shuffle=True,seed=None): # only require x and y for supervised learning
        if len(X) != len(Y):
            raise Exception("X and Y must be same length")  
        if seed == None:      
            seed = np.random.randint(0,100000)
        self._seed = seed
        self.n = len(X)
        if shuffle:
            x = X.sample(frac=1,random_state=seed)
            y = Y.sample(frac=1,random_state=seed)
        self.x,self.y = (x.reset_index(drop=True),y.reset_index(drop=True))


    def get_seed(self):
        '''
        Returns the seed used to randomly sample the data 
        '''
        return self._seed

    def split(self,n_splits=10):
        '''
        Splits the X and Y data into a number of splits
        returns a list of training and testing indices for each split
        '''
        test_indices = []
        train_indices = []
        self.split_size = int(self.n/n_splits)
        for n in range(0,n_splits):
            test_index = list(range(n*self.split_size,(n+1)*self.split_size)) # quik mafs
            train_index = [num for num in range(0,self.n) if num not in test_index]
            test_indices.append(test_index)
            train_indices.append(train_index)

            
        return (train_indices, test_indices)

    def shuffle(self):
        '''
        Shuffles the X and Y data and returns the shuffled X and Y variables
        '''
        return self.x, self.y 
    
    def stratified_kfold(self):
        pass
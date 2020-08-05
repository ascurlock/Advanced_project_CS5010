#%% Import Modules
from CV import CV
import unittest
import pandas as pd

class CV_TestCases(unittest.TestCase):
    '''
    Class to test the CV class for n-fold cross-validation
    '''
    def test_CV_init(self):
        '''
        Tests the constructor for the CV class to see whether the CV object, X, and Y dataframes are instanciated
        '''
        X=pd.DataFrame([1,2,3,4,5,6])
        Y=pd.DataFrame([1,2,3,4,4,5])
        kfold = CV(X,Y)
        self.assertIsInstance(kfold, CV)
        self.assertIsInstance(kfold.y, pd.DataFrame)
        self.assertIsInstance(kfold.x, pd.DataFrame)

    def test_CV_split(self):
        '''
        Tests the split method to see whether the splits are of the right shape and length
        '''
        X=pd.DataFrame([1,2,3,4,5,6])
        Y=pd.DataFrame([1,2,3,4,4,5])
        n_splits = 3 
        # Initialize CV 
        kfold=CV(X,Y)
        # Do splits of 3
        self.assertEqual(len(kfold.split(n_splits=n_splits)[0]), 3)
        self.assertEqual(len(kfold.split(n_splits=n_splits)[1]), 3)

    def test_init_shuffle_x(self):  
        '''
        Tests to see whether the x and y dataframes are shuffled properly in the constructor
        '''
        X=pd.DataFrame([1,2,3,4,5,6])
        Y=pd.DataFrame([1,2,3,4,4,5])
        kfold=CV(X,Y)
        self.assertFalse(kfold.x.equals(X))

    def test_init_shuffle_y(self):
        '''
        Tests to see whether the class returns the x and y dataframes properly
        '''
        X=pd.DataFrame([1,2,3,4,5,6])
        Y=pd.DataFrame([1,2,3,4,4,5])
        kfold=CV(X,Y)
        self.assertFalse(kfold.y.equals(Y))

    def test_shuffle_method_x(self):
        '''
        Tests to see whether the shuffle method works right for the feature dataframe
        '''
        X=pd.DataFrame([1,2,3,4,5,6])
        Y=pd.DataFrame([1,2,3,4,4,5])
        kfold=CV(X,Y)
        x_shuff , _ = kfold.shuffle()
        self.assertFalse(x_shuff.equals(X))

    def test_shuffle_method_y(self):
        '''
        Tests to see whether the shuffle method works right for the label dataframe
        '''
        X=pd.DataFrame([1,2,3,4,5,6])
        Y=pd.DataFrame([1,2,3,4,4,5])
        kfold=CV(X,Y)
        _ , y_shuff = kfold.shuffle()
        self.assertFalse(y_shuff.equals(Y))

    def test_get_seed_exists(self):
        '''
        Tests to see whether the get_seed() method properly returns a seed given no input seed
        '''
        X=pd.DataFrame([1,2,3,4,5,6])
        Y=pd.DataFrame([1,2,3,4,4,5])
        kfold=CV(X,Y,seed=None)
        self.assertTrue(kfold.get_seed())

    def test_get_seed_returns_correct(self):
        '''
        Tests to see whether the get_seed() method properly returns a seed given an input seed
        '''
        X=pd.DataFrame([1,2,3,4,5,6])
        Y=pd.DataFrame([1,2,3,4,4,5])
        seed = 42
        kfold=CV(X,Y,seed=42)
        self.assertEqual(seed,kfold.get_seed())

    def test_seed_shuffle(self):
        '''
        Tests to see whether inputting a seed creates reproducible X and Y dataframes.
        '''
        X=pd.DataFrame([1,2,3,4,5,6])
        Y=pd.DataFrame([1,2,3,4,4,5])
        seed = 42
        kfold=CV(X,Y,seed=42)
        x = X.sample(frac=1,random_state=seed)
        x.reset_index(drop=True,inplace=True)
        y = Y.sample(frac=1,random_state=seed)
        y.reset_index(drop=True,inplace=True)
        self.assertTrue(x.equals(kfold.shuffle()[0]))

if __name__ == "__main__":
    unittest.main()

# %%

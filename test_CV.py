#%% Import Modules
from CV import CV
import unittest
import pandas as pd

class CV_TestCases(unittest.TestCase):
    #testing the class CV

    def test_CV_init(self):
        # Create test dataframes and create object
        # check that it is not None
        X=pd.DataFrame([1,2,3,4,5,6])
        Y=pd.DataFrame([1,2,3,4,4,5])
        kfold = CV(X,Y)
        self.assertIsInstance(kfold, CV)
        self.assertIsInstance(kfold.y, pd.DataFrame)
        self.assertIsInstance(kfold.x, pd.DataFrame)

    def test_CV_split(self):
        # Create test dataframes
        X=pd.DataFrame([1,2,3,4,5,6])
        Y=pd.DataFrame([1,2,3,4,4,5])
        n_splits = 3 
        # Initialize CV 
        kfold=CV(X,Y)
        # Do splits of 3
        self.assertEqual(len(kfold.split(n_splits=n_splits)[0]), 3)
        self.assertEqual(len(kfold.split(n_splits=n_splits)[1]), 3)

    def test_init_shuffle_x(self):
        X=pd.DataFrame([1,2,3,4,5,6])
        Y=pd.DataFrame([1,2,3,4,4,5])
        kfold=CV(X,Y)
        self.assertFalse(kfold.x.equals(X))

    def test_init_shuffle_y(self):
        X=pd.DataFrame([1,2,3,4,5,6])
        Y=pd.DataFrame([1,2,3,4,4,5])
        kfold=CV(X,Y)
        self.assertFalse(kfold.y.equals(Y))

    def test_shuffle_method_x(self):
        X=pd.DataFrame([1,2,3,4,5,6])
        Y=pd.DataFrame([1,2,3,4,4,5])
        kfold=CV(X,Y)
        x_shuff , _ = kfold.shuffle()
        self.assertFalse(x_shuff.equals(X))

    def test_shuffle_method_y(self):
        X=pd.DataFrame([1,2,3,4,5,6])
        Y=pd.DataFrame([1,2,3,4,4,5])
        kfold=CV(X,Y)
        _ , y_shuff = kfold.shuffle()
        self.assertFalse(y_shuff.equals(Y))

    def test_get_seed_exists(self):
        X=pd.DataFrame([1,2,3,4,5,6])
        Y=pd.DataFrame([1,2,3,4,4,5])
        kfold=CV(X,Y,seed=None)
        self.assertTrue(kfold.get_seed())

    def test_get_seed_returns_correct(self):
        X=pd.DataFrame([1,2,3,4,5,6])
        Y=pd.DataFrame([1,2,3,4,4,5])
        seed = 42
        kfold=CV(X,Y,seed=42)
        self.assertEqual(seed,kfold.get_seed())

    def test_seed_shuffle(self):
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

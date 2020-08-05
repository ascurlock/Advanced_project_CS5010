#%% Import Modules
from CV import *
import unittest

class CV_TestCases(unittest.TestCase):
    def test_CV_init_x(self):
        X=pd.DataFrame([1,2,3,4,5,6])
        Y=pd.DataFrame([1,2,3,4,4,5])
        kfold = CV(X,Y)
        self.assertTrue(kfold.x is not None)
    def test_CV_init_y(self):
        X=pd.DataFrame([1,2,3,4,5,6])
        Y=pd.DataFrame([1,2,3,4,4,5])
        kfold = CV(X,Y)
        self.assertTrue(kfold.y is not None)
    def test_CV_split_length(self):
        X=pd.DataFrame([1,2,3,4,5,6])
        Y=pd.DataFrame([1,2,3,4,4,5])
        n_splits = 3 
        kfold=CV(X,Y)
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


if __name__ == "__main__":
    unittest.main()

# %%

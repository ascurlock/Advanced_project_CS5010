#%% Import Modules
from CV import *
import unittest

#%%
class CV_TestCases(unittest.TestCase):
    def test_CV_init(self):
        X = [1,2,3,4,5]
        Y = [1,2,3,4,5]
        kfold = CV(X,Y)
        self.assertEqual(kfold._x,X)
        self.assertEqual(kfold._y,Y)
    def test_CV_split_length(self):
        X=[1,2,3,4,5,6]
        Y=[1,2,3,4,4,5]
        n_splits = 3 
        kfold=CV(X,Y)
        self.assertEqual(len(kfold.split(n_splits=n_splits)[0]), 3)
        self.assertEqual(len(kfold.split(n_splits=n_splits)[1]), 3)

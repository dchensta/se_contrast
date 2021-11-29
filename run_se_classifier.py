import pandas as pd
import pickle as pkl
from SE_Classifier import SE_Classifier
import random
import numpy as np

random.seed(0)
np.random.seed(0)

#Initialize parameters for SE_Classifier object by choosing to analyze contrast or test set.
ui = input("Predict on contrast or test test? (contrast/test)")
folder = ""; flag = ""
if ui == "contrast" :
    folder = "sd_test_CONTRASTS"
    flag = "contrast"
elif ui == "test" :
    folder = "sd_test_GOLD"
    flag = "test"

lr_clf = SE_Classifier(folder, flag)
predictions = lr_clf.predict() #results are in /Data/scored_output.csv
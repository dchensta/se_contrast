import pandas as pd
import pickle as pkl
from SE_Classifier import SE_Classifier
import random
import numpy as np

random.seed(0)
np.random.seed(0)

#Initialize parameters for SE_Classifier object by choosing to analyze contrast or test set.
flag = input("contrast or test? ")
folder = ""
if flag == "contrast" :
    folder = "sd_test_CONTRASTS"
elif flag == "test" :
    folder = "sd_test_GOLD"

lr_clf = SE_Classifier(folder, flag)
predictions = lr_clf.predict() #results are in "sd_contrast_output" or "sd_test_output"
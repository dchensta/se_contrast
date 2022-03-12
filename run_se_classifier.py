import pandas as pd
import pickle as pkl
from SE_Classifier import SE_Classifier
import random
import numpy as np

random.seed(0)
np.random.seed(0)

#Initialize parameters for SE_Classifier object by choosing to analyze contrast or test set.
folder = "sd_test_CONTRASTS"
flag = "contrast"

lr_clf = SE_Classifier(folder, flag)
predictions = lr_clf.predict() #results are in /Data/scored_output.csv
import pandas as pd
import pathlib
import regex as re
import pickle as pkl
from SE_Classifier import SE_Classifier
import random
import numpy as np

random.seed(0)
np.random.seed(0)

def analyze(dir):
    dir = pathlib.Path(dir).rglob("*csv")

    for file in dir :
        df = pd.read_csv(file)
        size = len(df["Contrast"])

        #contrast_acc:
        contrast_acc = 0; gold_acc = 0
        contrast_size = 0

        for pred, contrast_label, strategy in zip(df["predictions"], df["Contrast"], df["Strategy"]) : #contrast flag
            if type(pred) != str :
                continue
            p = re.sub('[\[\]\']', '', pred) #Remove brackets and quotation marks from predictions cell
            #gold_p = re.sub('[\[\]\']', '', gold_pred)
            if type(strategy) == str : #contrast flag
                contrast_size += 1

                if p == contrast_label :
                    contrast_acc += 1
                    strategy_dict[strategy] = strategy_dict.get(strategy, 0) + 1
                #print(f"gold_p: {gold_p}, gold_label: {gold_label}")
                #if gold_p == gold_label :
                #    gold_acc += 1

        contrast_acc = round((contrast_acc/contrast_size)*100, 2)
        print("gold size: ", gold_acc)
        gold_acc = round((gold_acc/contrast_size)*100, 2)
        print(f"Contrast Set Accuracy for {file.name}: {contrast_acc}%")
        print(f"Gold Test Set Accuracy for {file.name}: {gold_acc}%")
        print(f"file size: {size}")
    
    
if __name__ == "__main__":
    dir = "sd_contrast_output"
    analyze(dir)
    #separate analyze function
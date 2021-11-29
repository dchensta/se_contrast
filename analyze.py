import pandas as pd
import pathlib
import regex as re

def analyze(dir, flag):
    dir = pathlib.Path(dir).rglob("*csv")

    total_same = 0
    total_diff = 0
    total_size = 0
    for file in dir :
        df = pd.read_csv(file)
        size = len(df["Gold"])

        #Count number of correct predictions, collect incorrect clauses.
        same = 0; diff = 0; same_clauses = []; diff_clauses = []
        for gold, pred, clause in zip(df["Gold"], df["predictions"], df["Clause"]) :
            p = re.sub('[\[\]\']', '', pred)
            if gold == p :
                same += 1
                same_clauses.append(clause)
            else :
                diff += 1
                diff_clauses.append(clause)
        
        #Calculate raw accuracy for this file.
        if flag == "test" :
            acc = round((same/size)*100, 2)
            print(f"Accuracy for {file.name}: {acc}%")
            print(f"Problem clauses for {file.name}: ")
            for cl in diff_clauses :
                print(cl)
            print('\n')
        elif flag == "contrast" :
            acc = round((diff/size)*100, 2)
            print(f"Accuracy for {file.name}: {acc}%")
            print(f"Problem clauses for {file.name}: ")
            for cl in same_clauses :
                print(cl)
            print('\n')

        #Append to total_correct and total_denom for test set accuracy.
        total_same += same
        total_diff += diff
        total_size += size
    
    #Print accuracy for entire test set.
    if flag == "test" :
        total_acc = round((total_same/total_size)*100, 2)
        print(f"Accuracy for entire test set: {total_acc}%")
        
if __name__ == "__main__":
    dir_name = input("test or contrast? ")
    dir = ""
    if dir_name == "test" :
        dir = "sd_test_output"
    elif dir_name == "contrast" :
        dir = "sd_contrast_output"

    print(f"Analyzing {dir_name} set...")
    analyze(dir, dir_name)
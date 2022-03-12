import pandas as pd
import pathlib
import regex as re

def analyze(dir, flag):
    dir = pathlib.Path(dir).rglob("*csv")

    total_same = 0
    total_diff = 0
    total_size = 0
    strategy_dict = {}
    for file in dir :
        df = pd.read_csv(file)
        size = len(df["Contrast"])

        #Count number of correct predictions, collect incorrect clauses.
        same = 0; diff = 0; same_clauses = []; diff_clauses = []
        cd_gold = 0; cd_pred = 0

        #contrast_acc:
        contrast_acc = 0; test_acc = 0
        contrast_size = 0; strategy_size = 0

        #for gold, pred, clause, contrast_label, strategy in zip(df["Gold"], df["predictions"], df["Clause"], df["Contrast"], df["Strategy"]) : #test flag
        for pred, clause, contrast_label, strategy in zip(df["predictions"], df["Clause"], df["Contrast"], df["Strategy"]) : #contrast flag
            if type(pred) != str :
                continue
            p = re.sub('[\[\]\']', '', pred) #Remove brackets and quotation marks from predictions cell

            if flag == "test" :
                if contrast_label == "y" :
                    contrast_size += 1
                    if gold == p :
                        same += 1
                        #same_clauses.append(clause)
                else :
                    diff += 1
                    #diff_clauses.append(clause)
                if type(strategy) == str :
                    strategy_size += 1

            elif flag == "contrast" and type(strategy) == str : #contrast flag
                contrast_size += 1

                strategy_dict[strategy] = strategy_dict.get(strategy, 0) + 1
                if p == contrast_label :
                    contrast_acc += 1
                if contrast_label == "CANNOT_DECIDE" :
                    cd_gold += 1
                if p == "CANNOT_DECIDE" :
                    cd_pred += 1

        #print(f"CANNOT_DECIDE Gold for {file.name}: {cd_gold} ")
        #print(f"CANNOT_DECIDE Predictions for {file.name}: {cd_pred} ")
        #Calculate raw accuracy for this file.
        if flag == "test" :
            print(f"contrast_size: {contrast_size}, stragegy_size: {strategy_size}")
            print(f"test_acc: {same}, test_size: {contrast_size}")
            acc = round((same/contrast_size)*100, 2)
            print(f"Accuracy for {file.name}: {acc}%")
            '''
            print(f"Problem clauses for {file.name}: ")
            for cl in diff_clauses :
                print(cl)
            print('\n')
            '''
        elif flag == "contrast" :
            '''
            print(f"Problem clauses for {file.name}: ")
            for cl in same_clauses :
                print(cl)
            print('\n')
            '''
            print(f"contrast_acc: {contrast_acc}, contrast_size: {contrast_size}")
            contrast_acc = round((contrast_acc/contrast_size)*100, 2)
            print(f"Contrast Set Accuracy for {file.name}: {contrast_acc}%")
            print(f"file size: {size}")

        #Append to total_correct and total_denom for test set accuracy.
        total_same += same
        total_diff += diff
        total_size += size
    
    #Print accuracy for entire test set.
    if flag == "test" :
        total_acc = round((total_same/total_size)*100, 2)
        print(f"Accuracy for entire test set: {total_acc}%")

    print("\nStrategy_Dict: ")
    for key, value in strategy_dict.items() :
        print(f"{key}: {value}" )
    print("\n")
        
if __name__ == "__main__":
    dir_name = input("test or contrast? ")
    dir = ""
    if dir_name == "test" :
        dir = "sd_test_output"
    elif dir_name == "contrast" :
        dir = "sd_contrast_output"

    print(f"Analyzing {dir_name} set...")
    analyze(dir, dir_name)
    #separate analyze function
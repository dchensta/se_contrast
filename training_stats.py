from posixpath import dirname
import pandas as pd
import pathlib
import regex as re

def analyze(dir):
    dir = pathlib.Path(dir).rglob("*csv")
    clause_count = 0
    dynamic_count = 0; stative_count = 0; cannot_decide_count= 0

    for file in dir :
        df = pd.read_csv(file)
        gold = df["Gold"]
        clause_count += len(gold)
        for g in gold :
            if g == "DYNAMIC" :
                dynamic_count += 1
            elif g == "STATIVE" :
                stative_count += 1
            else :
                cannot_decide_count += 1

    print("Final number of clauses: ", clause_count)
    print("Dynamic count: ", dynamic_count)
    print("Stative count: ", stative_count)
    print("Cannot decide count: ", cannot_decide_count)

if __name__ == "__main__":
    #dir = "sd_train_official"
    dir = "sd_test_gold"
    print(f"Analyzing {dir}...")
    analyze(dir)
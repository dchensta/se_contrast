from Featurizer import FeaturizerFactory
import pickle as pkl
import pathlib
import pandas as pd
import torch
import numpy as np
import os
import random

os.environ['KMP_DUPLICATE_LIB_OK']='True'

class SE_Classifier:
    def __init__(self, test_dir, test):
        self.testFlag = test
        print(test_dir)
        self.data, self.filenames = self.__read_data(test_dir, test)
        self.featX, self.featY = self.__create_testset()
        self.model = self.__read_model()

        self.output_path = ""
        if test == "test" :
            self.output_path = "sd_test_output"
        elif test == "contrast" :
            self.output_path = "sd_contrast_output_subsets"

    def __read_model(self):
        clf_model = pkl.load(open("sd_model/lr_sd_clf_bert.pkl", "rb"))
        #clf_model = pkl.load(open("sd_model/lr_sd_clf_roberta.pkl", "rb"))
        return clf_model

    def __read_data(self, test_dir, test):
        '''
        This function, called at class object initialization time, creates a DataFrame for each test file.
        Sets to self.data variable for __create_testset() to use

        @param {string} test_dir
        @caller __init__
        @return {array} master_data
        @return {array} filenames
        '''
        #Load Pathlib object for iterating through each file in the directory.
        dir = pathlib.Path(test_dir).rglob("*csv")

        master_data = []
        filenames = []
        
        for file in dir :
            filenames.append(file.name)
            #Open each file as a Pandas DataFrame.
            df = pd.read_csv(file)

            #Choose key name for "gold" column
            gold_key = ""
            if test == "test" :
                gold_key = "Gold"
            else :
                gold_key = "Contrast" #used to be Gold/Contrast

            clauses = list(df["Clause"])
            mainVerbs = list(df["mainVerb"])
            mainReferents = list(df["mainReferent"])
            gold = list(df[gold_key]) 
            #alternate gold name in contrast directory: Gold/Contrast

            #Return a Pandas DataFrame containing all data from all CSV files.
            data = pd.DataFrame({"Clause":clauses, 
                                "mainVerb":mainVerbs,
                                "mainReferent":mainReferents,
                                "Gold": gold})

            data = data.dropna(axis=0, subset=["Clause"])
            data = data.dropna(axis=0, subset=["Gold"])
            master_data.append(data)

        return master_data, filenames

    def __create_testset(self) :
        print("Creating test set")
        featX = []; featY = []
        '''Original SE Classifier
        for file in self.data :
            clauses = file["Clause"] #Access Clause column from master data DataFrame
            featurizer = FeaturizerFactory()

            X = featurizer.featurize(clauses, "BERT")
            filtered_X = X[~torch.any(X.isnan(),dim=1)]
            y = file["Gold"] #Gold stative/dynamic labels

            featX.append(filtered_X)
            featY.append(y)
        '''

        '''Subset SE Classifier'''
        X = []; y = []; stative_count = 0; dynamic_count = 0
        for file in self.data :
            if stative_count == 120 and dynamic_count == 120 :
                break
            #if stative_count == 84 and dynamic_count == 120 :
            #    break
            featurizer = FeaturizerFactory()
            clauses = file["Clause"]; contrasts = file["Gold"]
            paired_list = list(zip(clauses, contrasts))
            random.shuffle(paired_list)
            for clause, contrast in paired_list :
                if contrast == "STATIVE" :
                    stative_count += 1
                elif contrast == "DYNAMIC" :
                    dynamic_count += 1

                #happens for both types of labels
                X += clause
                y += contrast
        X = featurizer.featurize(X, "BERT")
        filtered_X = X[~torch.any(X.isnan(),dim=1)]
        featX = [filtered_X]
        featY = [y]

        return featX, featY


    def get_data(self):
        '''
        @caller __init__
        '''
        return self.featX, self.featY
    
    def predict(self) :
        print("Classifying test data")

        master_predictions = []
        file_idx = 0 #Update original dataframe file
        for fileX, fileY in zip(self.featX, self.featY) :
            predictions = []
            for x in fileX: #for each clause
                x = x.reshape(1, -1)
                predictions.append(self.model.predict(x))

            data = self.data[file_idx] #Pull out relevant data file to append predictions column to.
            #filename = self.filenames[file_idx]
            filename = "random_clauses_equal.csv"
            #filename = "random_clauses_resemble_test_distr.csv"

            pred_column_name = "predictions"
            data[pred_column_name] = predictions
            data.to_csv(self.output_path + "/" + filename)
            file_idx += 1
            master_predictions.append(predictions)
        print("Classification finished.")
        return master_predictions
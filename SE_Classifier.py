from Featurizer import FeaturizerFactory
import pickle as pkl
import pathlib
import pandas as pd
import torch
import numpy as np
import os
import random
import re

os.environ['KMP_DUPLICATE_LIB_OK']='True'

class SE_Classifier:
    def __init__(self, test_dir, test_flag):
        self.test_flag = test_flag
        print(test_dir)
        self.data, self.filenames = self.__read_data(test_dir)
        self.featX, self.featY = self.__create_testset()
        self.model = self.__read_model()

        self.output_path = ""
        if self.test_flag == "contrast" :
            self.output_path = "sd_contrast_output"
        elif self.test_flag == "test" :
            self.output_path = "sd_test_output"

    def __read_model(self):
        clf_model = pkl.load(open("sd_model/lr_sd_clf_bert.pkl", "rb"))
        return clf_model

    def __read_data(self, test_dir):
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
            if self.test_flag == "test" :
                gold_key = "Gold"
            else :
                gold_key = "Contrast" 

            clauses = list(df["Clause"])
            orig_clauses = self.get_orig_clauses(file.name, len(clauses))
            main_verbs = list(df["mainVerb"])
            main_referents = list(df["mainReferent"])
            gold = list(df[gold_key]) 

            contrast_verb = [None] * len(clauses) #All clauses must be the same length to create a new DataFrame object
            strats = [None] * len(clauses)
            #alternate gold name in contrast directory: Gold/Contrast

            #"data" corresponds to one filename. master_data appends each file's data.
            data = None
            if self.test_flag == "contrast" :
                data = pd.DataFrame({"orig_clause":orig_clauses,
                                "Clause":clauses, 
                                "Strategy":strats,
                                "orig_clause_main_verb":main_verbs,
                                "contrast_verb":contrast_verb,
                                "main_referent":main_referents,
                                "Contrast":gold}) #variable is called gold, but final CSV will label this column as "Contrast"
            else : #test_flag == test
                data = pd.DataFrame({"Clause":clauses, 
                                "main_verb":main_verbs,
                                "main_referent":main_referents,
                                "Gold":gold})

            data = data.dropna(axis=0, subset=["Clause"])
            if self.test_flag == "contrast" :
                data = data.dropna(axis=0, subset=["orig_clause"])
                data = data.dropna(axis=0, subset=["Contrast"])
            else :
                data = data.dropna(axis=0, subset=["Gold"])
            master_data.append(data)

        #Return a Pandas DataFrame containing all data from all CSV files.
        return master_data, filenames

    def __create_testset(self) :
        '''
        Creates self.featX and self.featY, to be used in the predict function that can be publically called by the user.
        '''
        print("Creating test set")
        featX = []; featY = []
        '''Original SE Classifier'''
        for file in self.data : #self.data created by __read_data() function
            clauses = file["Clause"] #Access Clause column from master data DataFrame
            featurizer = FeaturizerFactory()

            X = featurizer.featurize(clauses, "BERT")
            filtered_X = X[~torch.any(X.isnan(),dim=1)]

            y = None
            if self.test_flag == "test" :
                y = file["Gold"]
            else :
                y = file["Contrast"]

            featX.append(filtered_X)
            featY.append(y)
        return featX, featY

    def get_orig_clauses(self, filename, len_contrast_clauses) :
        '''
        If multiple contrast alterations exist for one clause, user must manually realign
        original clauses column. See Rows 46-48, "In its letter, which was signed by all six members"
        in the document news_20020731-nyt_contrasts_output".
        '''
        if self.test_flag == "test" :
            return []
        else :
            gold_filename = "sd_test_GOLD/" + filename[:-14] + ".csv"
            df = pd.read_csv(gold_filename)
            gold_list = list(df["Clause"])
            if len(gold_list) < len_contrast_clauses:
                diff = len_contrast_clauses - len(gold_list)

                extra_list = []
                i = 0
                while i < diff :
                    extra_list.append("NEED_TO_MANUALLY_REALIGN")
                    i += 1

                padded_list = gold_list + extra_list #pad length to match length of clause 
                return padded_list
            else :
                return gold_list

    def get_data(self):
        '''
        @caller __init__
        '''
        return self.featX, self.featY
    
    def predict(self) :
        print("Classifying test data")

        master_predictions = []
        file_idx = 0 #Update original dataframe file
        for fileX, _ in zip(self.featX, self.featY) :
            predictions = []
            for x in fileX: #for each clause
                x = x.reshape(1, -1)
                y_hat = self.model.predict(x)
                y_hat = re.sub('[\[\]\']', '', str(y_hat)) #Remove brackets and quotation marks from predictions cell
                predictions.append(y_hat)

            data = self.data[file_idx] #Pull out relevant data file to append predictions column to.
            filename = self.filenames[file_idx]
            #filename = "random_clauses_equal.csv"
            #filename = "random_clauses_resemble_test_distr.csv"

            pred_column_name = "predictions"
            data[pred_column_name] = predictions
            data.to_csv(self.output_path + "/" + filename[:-4] + "_output.csv") #see init function for output_path names
            file_idx += 1
            master_predictions.append(predictions)
        print("Classification finished.")
        return master_predictions
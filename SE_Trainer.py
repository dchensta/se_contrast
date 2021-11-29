from Featurizer import FeaturizerFactory
from numpy.core.fromnumeric import argmax
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from operator import itemgetter
import pickle as pkl
import warnings
import pathlib
import torch

class SE_Trainer:
    def __init__(self, train_dir, model_dir):
        self.data = self.__read_data(train_dir)
        self.X, self.y = self.__create_trainset()
        print("X:", self.X)
        print("y:", self.y)
        self.model_path = model_dir

    def train(self):
        '''
        This is the main function of the Trainer class. It outputs trained filter and classifier models, 
        as well as corresponding stats reports.

        @param {.csv} data => 
        '''
        
        #warnings.filterwarnings(action='ignore')

        print('Initiating training process')

        #print("Running hyperparameter searches: Classifier")
        #clf_results = self.__run_hyperparam_search()
        #print("Models trained.")

        # clf_model = clf_results["model"]
        # clf_featurization = clf_results["featurization"]
        # clf_avg_score = clf_results["avg_score"]
        # clf_std = clf_results["std_score"]

        #report = f"Best performing classifier: {clf_model}, Featurization: {clf_featurization}, Classifier Performance: {clf_avg_score}, Classifier Standard Deviation: {clf_std}"

        #11/1/21 attempt
        X, y = self.get_data()
        clf_model = LogisticRegression(solver="liblinear", random_state=0, max_iter=500).fit(X, y)
        print('Training Completed')
        #print(report)

        pkl.dump(clf_model, open(self.model_path + "/lr_se_clf.pkl", "wb"))
        #return clf_results
        return "Finished training classifier."
    
    def get_data(self):
        '''
        @caller __init__
        '''
        return self.X, self.y

    def __read_data(self, train_dir):
        '''
        This function, called at class object initialization time, consolidates all CSV files in a training directory 
        into a single master DataFrame, so that the machine learning process has access to all training data all at once.

        @param {string} train_dir 
        @caller __init__
        '''
        #Load Pathlib object for iterating through each file in the directory.
        dir = pathlib.Path(train_dir).rglob("*csv")

        data = []
        clauses = []; mainVerbs = []; mainReferents = []; gold = []
        
        for file in dir :
            #Open each file as a Pandas DataFrame.
            df = pd.read_csv(file)
            #Append all 4 (full) columns contained in each file.
            clauses += list(df["Clause"])
            mainVerbs += list(df["mainVerb"])
            mainReferents += list(df["mainReferent"])
            gold += list(df["Gold"])

        #Return a Pandas DataFrame containing all data from all CSV files.
        data = pd.DataFrame({"Clause":clauses, 
                            "mainVerb":mainVerbs,
                            "mainReferent":mainReferents,
                            "Gold": gold})

        data = data.dropna(axis=0, subset=["Clause"])
        data = data.dropna(axis=0, subset=["Gold"])
        return data

    def __create_trainset(self) :
        print("Creating training set")
        clauses = self.data["Clause"] #Access Clause column from master data DataFrame
        featurizer = FeaturizerFactory()
        print("Featurizing")
        # X = featurizer.featurize(clauses, "BERT")
        # filtered_X = X[~torch.any(X.isnan(),dim=1)]
        # torch.save(filtered_X, 'x_tensor.pt')
        filtered_X = torch.load("x_tensor.pt")
        y = self.data["Gold"] #Gold stative/dynamic labels
        return filtered_X, y

    def __run_hyperparam_search(self):
        '''
        This function runs a two-fold grid search for optimal parameters and hyperparameters for sklearn machine learning models.

        @param {dict} prop_dict
        @param {Pandas DataFrame} train_data
        '''
        best_systems = []

        #To implement: for feat in featurizations loop
        print("Making training data")
        #X, y = self.trainset.get_data()
        X, y = self.get_data()

        print("Running searches")
        lr_hyperparams_dict = self.__get_lr_hyperparams(X, y)
        #lr_hyperparams_dict["featurization"] = feat
        lr_hyperparams_dict["featurization"] = "BERT"

        svm_hyperparams_dict = self.__get_svm_hyperparams(X, y)
        #svm_hyperparams_dict["featurization"] = feat
        svm_hyperparams_dict["featurization"] = "BERT"

        best_systems.append(lr_hyperparams_dict)
        best_systems.append(svm_hyperparams_dict)

        print("Found best system")
        best_system = best_systems[argmax(
            [d["avg_score"] for d in best_systems])]
        return best_system

    def __get_lr_hyperparams(self, X, y):
        '''
        This function runs a grid search on Logistic Regression to find the ideal hyperparameters for the passed-in dataset.

        @param {list} X = featurization of text data
        @param {list} y = annotated scores assigned to the text being represented by X

        @return {dict} optimal_dict = { "avg_score":{float} = optimal average 5-fold cross-validation score
                                        "std_score":{float} = standard deviation accompanying optimal cross-validation score
                                        "model":{sklearn LogisticRegression} = Logistic Regression model that produced optimal average 5-fold cross-validation score
                                      }
        '''
        solvers = ['liblinear', 'newton-cg', 'lbfgs'] #algorithms that do some approximation 
        penalties = ['none', 'l2']
        c_values = [100, 10, 1.0, 0.1, 0.01]

        lr_hyperparams = []
        for c in c_values:
            for solver in solvers:
                for penalty in penalties:
                    lr_model = LogisticRegression(
                        C=c, solver=solver, penalty=penalty).fit(X, y) 
                    cv = cross_val_score(lr_model, X, y, cv=5)
                    lr_cv_score = np.mean(cv)
                    lr_std_score = np.std(cv)
                    lr_hyperparams.append(
                        (lr_cv_score, lr_std_score, lr_model))

        optimal_tuple = max(lr_hyperparams, key=itemgetter(0))
        return {"avg_score": optimal_tuple[0], "std_score": optimal_tuple[1], "model": optimal_tuple[2]}

    def __get_svm_hyperparams(self, X, y):
        '''
        This function runs a grid search on Support Vector Machine to find the ideal hyperparameters for the passed-in dataset.

        @param {list} X = featurization of text data
        @param {list} y = annotated scores assigned to the text being represented by X

        @return {dict} optimal_dict = { "avg_score":{float} = optimal average 5-fold cross-validation score
                                        "std_score":{float} = standard deviation accompanying optimal cross-validation score
                                        "model":{sklearn SVC} = Support Vector Machine model that produced optimal average 5-fold cross-validation score
                                      }
        '''
        kernels = ['linear', 'rbf']
        c_values = [50, 10, 1.0, 0.1, 0.01]
        gamma = 'scale'

        svm_hyperparams = []
        for c in c_values:
            for kernel in kernels:
                svm_model = SVC(C=c, kernel=kernel, gamma=gamma).fit(X, y)
                cv = cross_val_score(svm_model, X, y, cv=5)
                svm_cv_score = np.mean(cv)
                svm_std_score = np.std(cv)
                svm_hyperparams.append(
                    (svm_cv_score, svm_std_score, svm_model))

        optimal_tuple = max(svm_hyperparams, key=itemgetter(0))
        return {"avg_score": optimal_tuple[0], "std_score": optimal_tuple[1], "model": optimal_tuple[2]}
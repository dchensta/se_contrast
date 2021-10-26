from Featurizer import FeaturizerFactory

class SE_Classifier:
    def __create_test(self):
        featurizer = FeaturizerFactory()
        X = featurizer.featurize(clauses, "BERT")
        #y = STATIVE/DYNAMIC
        return X, y
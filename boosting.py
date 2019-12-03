import pandas as pd 
import dtree as dt

class Boosting:
    def __init__(self, df, no_of_models, test_df):
        self.df = df
        self.no_of_models = no_of_models
        self.test_df = test_df
        self.alpha_vals = None
        self.models = None
        self.accuracies = []
        self.predictions = None


    def fit(self):
        features = self.df.drop(['language'], axis = 1)
        #map to 
        labels = self.df['language'].where(self.df['language']==1, -1)

        eval_df = pd.DataFrame(labels.copy())
        eval_df['weights'] = 1/len(self.df)

        alpha_vals = []
        models = []

        for i in range(self.no_of_models):
            tree_model = dt.Tree(df.columns[:-1], df.columns[-1])
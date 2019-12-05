import pandas as pd 
import dtree as dt
import numpy as np 
import lab2 as lab2

class Boosting:
    def __init__(self, df, no_of_models):
        self.df = df
        self.no_of_models = no_of_models
        self.alpha_vals = None
        self.models = None
        self.accuracies = []
        self.predictions = None

    def get_predictions(self, features, tree_model):
        pred_list = []
        f_dict = features.to_dict(orient='records')
        for i in range(len(features)):
            q = f_dict[i]
            pred_list.append(tree_model.predict(q, tree_model.tree))
        return np.asarray(pred_list)

    def predict(self, query):
        predictions = []
        for alpha, model in zip(self.alpha_vals, self.models):
            pred = model.predict(query)
            predictions.append(alpha * (1 if pred=='en' else -1))

        prediction = np.sign(np.sum(np.array(predictions), axis = 0))

        return 'en' if prediction > 0 else 'nl'


    def fit(self):
        features = self.df.iloc[:, :-1]
         
        labels = self.df['language']

        self.df['language'].map(dict(en=1, nl=-1))

        eval_df = pd.DataFrame(labels.copy())
        eval_df['weights'] = 1/len(self.df)

        alpha_vals = []
        models = []

        for i in range(self.no_of_models):
            tree_model = dt.Tree(self.df.columns[:-1], self.df.columns[-1])
            tree_model.build_stump(self.df, eval_df['weights'])
            models.append(tree_model)
            predictions = self.get_predictions(features, tree_model)
            
            eval_df['predictions'] = predictions
            eval_df['evaluation'] = np.where(eval_df['predictions'] == self.df['language'], 1, 0)
            eval_df['misclassified'] = np.where(eval_df['predictions'] != self.df['language'], 1, 0)
            acc = sum(eval_df['evaluation'])/len(eval_df['evaluation'])
            
            err = np.sum(eval_df.weights*eval_df.misclassified) / np.sum(eval_df['weights'])

            a = np.log((1-err)/err)
            alpha_vals.append(a)

            new_weights = eval_df['weights'] * np.exp( a * eval_df['misclassified'])
            
            eval_df['weights'] = new_weights
            
        self.alpha_vals = alpha_vals
        self.models = models


def show_analysis(boost, df, no_of_models, test_df):
    for i in range(no_of_models):
        model = Boosting(df, no_of_models)
        model.fit()
import build_tree as b
import pydotplus
import numpy as np

class Tree():
    features: []
    labels: str
    tree = {}

    def __init__(self, features, labels):
        self.features = features
        self.labels = labels
        self.tree = {}

    def build(self, df):
        self.tree = b.build_tree(df, df, df.columns.values[:-1], 'language')
        #print(self.tree)

    def build_stump(self, df, sample_weight):
        self.tree = b.build_decision_stump(df, df, df.columns.values[:-1], 'language', sample_weight)
        #print(self.tree)

    def show_tree(self):
        print(tree)

    def predict(self, query, tr = None, default = 1):
        if tr == None:
            tr = self.tree
        for key in list(query.keys()):
            if key in list(tr.keys()):
                #2.
                try:
                    result = tr[key][query[key]] 
                except:
                    return default
    
                #3.
                result = tr[key][query[key]]
                #4.
                if isinstance(result,dict):
                    return self.predict(query,result)
                else:
                    return result
import pandas as pd
import numpy as np
import math 
from pprint import pprint
import matplotlib.pyplot as plt 

def read_data(file_name):
    attr = np.arange(1, 10)
    df = pd.read_csv(file_name, sep = ' ', names = attr)
    return df, attr



def get_entropy(df, target_attr, sample_weight):
    #In this case, the unique elements are A and B 
    elements, counts = np.unique(df[target_attr], return_counts=True)
    #print(elements, counts)
    entropy = 0

    if np.unique(sample_weight)[0] != 1:
        print(len(np.unique(sample_weight)))
        for i in range(len(df)):
            idx = elements.index(df[target_attr][i])
            counts[idx] = counts[idx] + sample_weight[i] - 1

    for i in range(len(elements)):
        pA = counts[i]/np.sum(sample_weight)
        entropy = entropy + (pA * math.log2(1/pA)) 
    return entropy



def get_info_gain(df, attr, sample_weight, target = 'language'):
    #split feature df into True and False value, calculated weighted entropy for each 
    original_entropy = get_entropy(df, target, sample_weight)
    elements, counts = np.unique(df[attr], return_counts = True)
    if np.unique(sample_weight)[0] != 1:
        for i in range(len(df)):
            idx = elements.index(df[attr][i])
            counts[idx] = counts[idx] + sample_weight[i] - 1
    weighted_entropy = 0

    for i in range(len(elements)):
        #pA = counts[i]/len(df)
        pA = counts[i] / np.sum(sample_weight)
        df_stripped = df.where(df[attr]==elements[i]).dropna()
        weighted_entropy = weighted_entropy + pA * get_entropy(df_stripped, target, sample_weight)
    
    info_gain = original_entropy - weighted_entropy
    return info_gain


def get_mode_value(vals):
    elements, counts = np.unique(vals, return_counts=True)
    return elements[np.argmax(counts)]


def build_tree(df, original_df, features, target_attr, prev_feature = None):
    if len(np.unique(df[target_attr])) <= 1:
        #print('aha')
        return np.unique(df[target_attr])[0]
    elif len(df) == 0:
        #print('aha2')
        return get_mode_value(original_df[target_attr])
    elif len(features) == 0:
        #print('aha3')
        return prev_feature
    else:
        prev_feature = np.unique(df[target_attr])[np.argmax(np.unique(df[target_attr], return_counts=True)[1])]
        item_values = []
        for feature in features:
            item_values.append(get_info_gain(df, feature, np.full(len(df), 1), target_attr))
        best_feature = features[np.argmax(item_values)]


        tree = {best_feature:{}}

        features = [feature for feature in features if feature != best_feature]

        for value in np.unique(df[best_feature]):
            #print(best_feature, value)
            value = value
            sub_df = df.where(df[best_feature] == value).dropna()
            sub_tree = build_tree(sub_df, original_df, features, target_attr, prev_feature)
            tree[best_feature][value] = sub_tree
        return(tree)

def build_decision_stump(df, original_df, features, target_attr, sample_weight, max_depth = 1):
    item_values = []
    for feature in features:
        item_values.append(get_info_gain(df, feature, sample_weight, target_attr))
    best_feature = features[np.argmax(item_values)]

    tree = {best_feature: {}}

    for value in np.unique(df[best_feature]):
        value = value 
        tree[best_feature][value] = value
    return(tree)

def main():
    df, attr = read_data('dtree-data.txt')
    #print(df.columns)
    features = attr[:-1]
    labels = attr[8]
    
    #print(get_info_gain(df, 4))

    tree = build_tree(df, df, features, 9)
    pprint(tree)
    

if __name__ == '__main__':
    main()

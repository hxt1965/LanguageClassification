import pandas as pd
import numpy as np
import math 
from pprint import pprint
import matplotlib.pyplot as plt 
import sys

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
        #print(len(np.unique(sample_weight)))
        #print(np.sum(sample_weight))
        counts = np.full(len(elements), 0).astype(float)
        for i in range(len(df)):
            idx = elements.tolist().index(df[target_attr][df.index[i]])
            counts[idx] = counts[idx] + sample_weight[i]
    #print(counts)
    for i in range(len(elements)):
        #print(elements[i], counts[i])
        #print(counts[i], np.sum(sample_weight))
        pA = counts[i]/np.sum(sample_weight)
        """if pA == 0:
            print(target_attr, '\n', counts[i], elements[i])
            sys.exit()"""
        entropy = entropy + (pA * math.log2(1/pA) if pA!=0 else 0) 
    return entropy



def get_info_gain(df, attr, sample_weight, target = 'language'):
    #split feature df into True and False value, calculated weighted entropy for each 
    original_entropy = get_entropy(df, target, sample_weight)
    elements, counts = np.unique(df[attr], return_counts = True)

    if np.unique(sample_weight)[0] != 1:
        counts = np.full(len(elements), 0).astype(float)
        for i in range(len(df)):
            idx = elements.tolist().index(df[attr][i])
            counts[idx] = counts[idx] + sample_weight[i] 
    weighted_entropy = 0

    for i in range(len(elements)):
        #pA = counts[i]/len(df)
        pA = counts[i] / np.sum(sample_weight)
        #print(attr)
        df_stripped = df.where(df[attr]==elements[i]).dropna()
        weighted_entropy = weighted_entropy + pA * get_entropy(df_stripped, target, sample_weight)
    
    info_gain = original_entropy - weighted_entropy
    return info_gain


def get_mode_value(vals):
    elements, counts = np.unique(vals, return_counts=True)
    return elements[np.argmax(counts)]

def get_stump_decision(df, target_attr, sample_weight):
    elements = np.unique(df[target_attr])
    #print('eles!', elements)
    weights = np.full(len(elements), 0).astype(float)

    for i in range(len(elements)):
        for k in df.index.values:
            if elements[i] == df[target_attr][k]:
                weights[i] = weights[i] + sample_weight[k]
    
    return elements[np.argmax(weights)]

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

def build_decision_stump(df, original_df, features, target_attr, sample_weight, max_depth = 1, prev_feature = None):
    
    item_values = []
    #print(df.columns.values)
    for feature in features:
        item_values.append(get_info_gain(df, feature, sample_weight, target_attr))
    best_feature = features[np.argmax(item_values)]
    item_values.sort()
    #print(best_feature, '\n', item_values)
    tree = {best_feature: {}}

    #print(item_values)

    for value in np.unique(df[best_feature]):
        value = value 
        sub_df = df.where(df[best_feature] == value).dropna()
        dec = get_stump_decision(sub_df, target_attr, sample_weight)
        #print(best_feature, value, dec)
        tree[best_feature][value] = dec
    return(tree)

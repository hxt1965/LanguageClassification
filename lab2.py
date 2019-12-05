import dtree as dt
import sys
import subprocess
from collections import Counter
import pandas as pd 
import boosting as boost 
from sklearn.utils import shuffle
import pickle

features = ['nl_bigrams', 'en_bigrams', 'hyphenated', 'double_letters', 'double_consonants', \
            'no_of_vowels', 'has_long_words', 'has_short_words', 'has_f', 'has_y', 'language']

MESSAGE_ERR_ARGS = 'Usage: python3 lab2.py <examples> <hypothesisOut> <learning-type>'

vowels = ['a', 'e', 'i', 'o', 'u']
mode_en = ['e', 't', 'a', 'i', 'o']
mode_nl = ['e', 'n', 'a', 't', 'i']

nl_bigrams = ['en', 'er', 'de']
en_bigrams = ['th', 'he', 'in']

#english = 4.7
#dutch  = 8.41
# number of words with length greater than 7
def has_nl_bigrams(line):
    words = line.split(' ')
    cnt = 0
    for word in words:
        for bigram in nl_bigrams:
            if bigram in word:
                cnt = cnt+1
    return True if cnt >= 2 else False

def has_en_bigrams(line):
    words = line.split(' ')
    cnt = 0
    for word in words:
        for bigram in en_bigrams:
            if bigram in word:
                cnt = cnt+1
    return True if cnt >= 2 else False

def has_f(line):
    words = line.split(' ')
    for word in words:
        for c in word:
            if c == 'f':
                return True
    return False

def has_y(line):
    words = line.split(' ')
    for word in words:
        for c in word:
            if c == 'y':
                return True
    return False

def has_short_words(line):
    words = line.split(' ')
    cnt = 0
    for word in words:
        if len(word) <= 4:
            cnt = cnt+1
    return True if cnt>3 else False

def has_long_words(line):
    words = line.split(' ')
    cnt = 0
    for word in words:
        if len(word) >= 7:
            cnt = cnt+1
    return True if cnt>3 else False

def get_is_hyphenated(line):
    return line.count('-') > 0

def get_has_double_letters(line):
    for c in range(len(line) - 1):
        if line[c] == line[c+1]:
            return True 
    if(line[len(line)-1] == line[len(line)-2]):
        return True
    return False

def get_has_double_consonants(line):
    l = len(line)
    for c in range(l-1):
        if line[c] == line[c+1] and \
            line[c] not in vowels and line[c+1] not in vowels \
            and line[c].isalpha() and line[c+1].isalpha():
            return True

    if(line[len(line)-1] == line[len(line)-2]) and \
        line[len(line)-1] not in vowels and line[len(line)-2] not in vowels \
            and line[len(line)-1].isalpha() and line[len(line)-2].isalpha():
        return True
    return False

def get_number_of_vowels(line):
    cnt = 0
    for c in line:
        if c in vowels:
            cnt = cnt+1
    return True if cnt <= 28 else False

# Build data frame with features and values 
def build_features(line, label):
    # went sending line into respective functions, send line.lower()
    entry = {features[0]: has_nl_bigrams(line.lower()), \
                features[1]: has_en_bigrams(line.lower()), \
                features[2]: get_is_hyphenated(line.lower()), \
                features[3]: get_has_double_letters(line.lower()), \
                features[4]: get_has_double_consonants(line.lower()), \
                features[5]: get_number_of_vowels(line.lower()), \
                features[6]: has_long_words(line.lower()), \
                features[7]: has_short_words(line.lower()), \
                features[8]: has_f(line.lower()), \
                features[9]: has_y(line.lower()), \
                features[10]: label}
    return entry 

def read_data(data_filename, df):
    with open(data_filename, 'r') as fp:
        for line in fp:
            s = line.split('|')
            entry = build_features(s[1], s[0])
            #print(entry)
            df = df.append(entry, ignore_index = True)
    return df


def main():
    argv = sys.argv
    try:
        if argv[1] == 'train':
            print('\nReading in the data...')
            df = pd.DataFrame(columns=features)
            df = read_data(argv[2], df)
            df = shuffle(df)

            if argv[4] == 'dt':   
                print('Training Decision Tree Model...') 
                tree_obj = dt.Tree(features[:-1], 'language')
                tree_obj.build(df)
                print('Writing model to ', argv[3], '...')
                with open(argv[3], 'wb') as handle:
                    pickle.dump(tree_obj, handle, protocol=pickle.HIGHEST_PROTOCOL)
            elif argv[4] == 'ada':
                df = df.drop('has_y', 1)
                print('Training Adaboost Model...') 
                #for i in range(20):
                model = boost.Boosting(df, 30)
                model.fit()
                acc = 0
                print('Writing model to ', argv[3], '...')
                with open(argv[3], 'wb') as handle:
                    pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)  
        elif argv[1] == 'predict':
            pickle_in = open(argv[2], 'rb')
            model = pickle.load(pickle_in)
            with open(argv[3], 'r', encoding='utf8') as fp:
                for line in fp:
                    s = line.split('|')
                    q = build_features(s[1], s[0])
                    #print(s[1], s[0], '\n')
                    if q.get('language') : del q['language']
                    #print(q)
                    print('Language: ', model.predict(q))
    except:
        print(MESSAGE_ERR_ARGS)
    
if __name__ == '__main__':
    subprocess.call('pip install -r requirements.txt')
    main()
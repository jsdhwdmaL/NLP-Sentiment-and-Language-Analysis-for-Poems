import pandas as pd
import pickle
import re
import string
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk import pos_tag
from nltk.tag import StanfordNERTagger
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from jiayan import load_lm
from jiayan import CharHMMTokenizer
from jiayan import CRFPOSTagger
from jiayan import CRFSentencizer


"""Feature engineering functions"""

lm = load_lm('/Users/jojoli/jiayan_models/jiayan.klm')
postagger = CRFPOSTagger()
postagger.load('/Users/jojoli/jiayan_models/pos_model')

def count_words(text):
    cnt = len(str(text))
    for ch in str(text):
        if ch == " ":
            cnt-=1
    return cnt

def count_stopwords(text):
    cnt = 0
    ls = postagger.postag(text)
    for chr in ls:
        if chr == 'p' or chr == 'r' or chr == 'u' or chr == 'c':
            cnt+=1
    return cnt

def count_lines(text):
    return len(str(text).split())

def count_tag_noun(text):
    pos_counts = 0
    ls = postagger.postag(text)
    for each in ls:
        if each == 'b' or each == 'n' or each == 'nd' or each == 'nh' or each == 'ni' or each == 'nl' or each == 'ns' or each == 'nt' or each == 'nz' or each == 'j':
            pos_counts+=1
    return pos_counts

def count_tag_pron(text):
    pos_counts = 0
    ls = postagger.postag(text)
    for each in ls:
        if each == 'r':
            pos_counts+=1
    return pos_counts

def count_tag_verb(text):
    pos_counts = 0
    ls = postagger.postag(text)
    for each in ls:
        if each == 'v':
            pos_counts+=1
    return pos_counts

def count_tag_adj(text):
    pos_counts = 0
    ls = postagger.postag(text)
    for each in ls:
        if each == 'a' or each == 'z':
            pos_counts+=1
    return pos_counts

def count_tag_adv(text):
    pos_counts = 0
    ls = postagger.postag(text)
    for each in ls:
        if each == 'd':
            pos_counts+=1
    return pos_counts

# Function gain_features to get all features needed before CoreNLP


"""CoreNLP Functions"""

def read_parsing_file():
    """
    Read every parsing file
    Product: add to the preprocessed csv file with: 1. nouns that each adjective describes; 2. adjectives
    """
    basename = "/Users/jojoli/Documents/夏校申请:项目制作/英才计划/正式培养/chinese/datasets/parsing/parsing"
    df = pd.read_csv('/Users/jojoli/Documents/夏校申请:项目制作/英才计划/正式培养/chinese/datasets/Tang_Dynasty_preprocessed.csv')
    df = df.head(30)
    all_nouns = []
    all_adjs = []
    for i in range(0, 30):
        result = read_parsing(basename+str(i)+".txt")
        nouns = []
        adjs = []
        for key in result:
            nouns.append(key)
            adjs.append(result[key])
        all_nouns.append(nouns)
        all_adjs.append(adjs)
    df['dep_nouns'] = all_nouns
    df['dep_adjs'] = all_adjs

    # Save results into a csv file
    df.to_csv("/Users/jojoli/Documents/夏校申请:项目制作/英才计划/正式培养/chinese/datasets/Tang_Dynasty_parsed.csv", index=False)
    print("Parsing analysis completed.\n")


def read_parsing(filename):
    """
    Extract adjectives and nouns (imageries) from the parsing result of each poem
    Return them in a dictionary:
    {'noun':'adj',...}
    """
    # Analyze dependency parses
    result = {}
    nounkeylist = ["obj", "nmod"]
    adjkeylist = ["amod", "advmod"]
    with open(filename, 'r') as fileDP:
        lines = fileDP.readlines()
        for i in range(0, len(lines)):
            templine = lines[i]
            # Extract useful relationships (find adjectives and modified nouns)
            for k in nounkeylist:
                if k in templine:
                    # Extract nouns
                    noun = templine[4]
                    # Now find the nearest adjectives for the nouns
                    step = 1
                    flag = False
                    while (i-step >= 0 and i+step < len(lines)):
                        l1 = lines[i+step]
                        for j in adjkeylist:
                            if j in l1:
                                flag = True
                                adj = l1[4]
                                result[noun] = adj

                        if (flag == True):
                            break

                        l2 = lines[i-step]
                        for j in adjkeylist:
                            if j in l2:
                                flag = True
                                adj = l2[4]
                                result[noun] = adj

                        if (flag == True):
                            break

                        step+=1

    return result



# Imagery classification (by the basic sentiment model):
# it prints the results and return the counts of positive/negative imageries (for feature engineering)

def imagery_process(filename):
    df = pd.read_csv(filename)
    basic_df = pd.read_csv("/Users/jojoli/Documents/夏校申请:项目制作/英才计划/正式培养/chinese/datasets/sentiment_XS_30k_cleaned.csv")
    vectorizer = TfidfVectorizer()
    vectorizer.fit(basic_df["tokenized"])
    all_pred = []
    for each in df["dep_adjs"]:
        adj_string = []
        for word in each:
            adj_string.append(word)

        temp_df = pd.DataFrame({'adj':adj_string})
        test = vectorizer.transform(temp_df)
        df_test = pd.DataFrame(test)

        # sentiment classification
        filename = "/Users/jojoli/Documents/夏校申请:项目制作/英才计划/正式培养/chinese/code/basic_model_svc.sav"
        load_model = pickle.load(open(filename, 'rb'))
        pred = load_model.predict(df_test)
        all_pred.append(pred)
        print("Done")
    df['adj_tag'] = all_pred

    # Save
    df.to_csv("/Users/jojoli/Documents/夏校申请:项目制作/英才计划/正式培养/chinese/datasets/Tang_Dynasty_adj-tagged.csv", index=False)
    print("Imagery classification completed.")
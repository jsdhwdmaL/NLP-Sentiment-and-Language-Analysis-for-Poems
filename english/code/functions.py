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

eliza_stw = ['adieu', 'anon', 'attend', 'aye', 'but soft', 'doth', "e'en", "e'er", 'haply', 'hence', 'hither', 'sirrah', 'thee', 'thither', 'thou', 'thou art', 'thy', 'whence', 'wilt', 'withal', 'hath']
eliza_stw = stopwords.words('english') + eliza_stw
stw = stopwords.words('english')

pos_dic = {
    "NN" : "noun", "NNS" : "noun", "NNP": "noun", "NNPS" : "noun",
    "PRP" : "pron", "PRP$" : "pron", "WP" : "pron", "WP$" : "pron",
    "VB" : "verb", "VBD" : "verb", "VBG" : "verb", "VBN" : "verb", "VBP" : "verb", "VBZ": "verb",
    "JJ" : "adj", "JJR" : "adj", "JJS" : "adj",
    "RB"  : "adv", "RBR" : "adv", "RBS" : "adv", "WRB" : "adj"
}

def count_words(text):
    return len(str(text).split())

def count_uniquewords(text):
    return len(set(str(text).split()))

def count_chars(text):
    return len(str(text))

def word_density(text):
    return count_chars(text) / (count_words(text) + 1)

def count_stopwords(text):
    stopwords = [word for word in str(text).split() if word in stw]
    return len(stopwords)

def count_puncts(text):
    puncts = re.findall('[' + string.punctuation + ']' + '\n' + '\r', str(text))
    return len(puncts)

def count_upperwords(text):
    upperwords = re.findall(r"\b[A-Z0-9]+\b", str(text))
    return len(upperwords)

def count_lines(text):
    lines = 0
    for i in text:
        if i == '\n':
            lines = lines + 1
    return lines + 1

def count_tag_noun(text):
    pos_counts = 0
    for each in pos_tag(str(text).split()):
        if each[1] == 'NN' or each[1] == 'NNP' or each[1] == 'NNPS' or each[1] == 'NNS':
            pos_counts = pos_counts + 1
    return pos_counts

def count_tag_pron(text):
    pos_counts = 0
    for each in pos_tag(str(text).split()):
        if each[1] == 'PRP' or each[1] == 'PRP$' or each[1] == 'WP' or each[1] == 'WP$':
            pos_counts = pos_counts + 1
    return pos_counts

def count_tag_verb(text):
    pos_counts = 0
    for each in pos_tag(str(text).split()):
        if each[1] == 'VB' or each[1] == 'VBD' or each[1] == 'VBG' or each[1] == 'VBN' or each[1] == 'VBP' or each[1] == 'VBZ':
            pos_counts = pos_counts + 1
    return pos_counts

def count_tag_adj(text):
    pos_counts = 0
    for each in pos_tag(str(text).split()):
        if each[1] == 'JJ' or each[1] == 'JJR' or each[1] == 'JJS':
            pos_counts = pos_counts + 1
    return pos_counts

def count_tag_adv(text):
    pos_counts = 0
    for each in pos_tag(str(text).split()):
        if each[1] == 'RB' or each[1] == 'RBR' or each[1] == 'RBS' or each[1] == 'WRB':
            pos_counts = pos_counts + 1
    return pos_counts

# Function gain_features to get all features needed before CoreNLP


"""CoreNLP Functions"""

def read_parsing_file():
    """
    Read every parsing file
    Product: add to the preprocessed csv file with: 1. nouns that each adjective describes; 2. adjectives
    """
    basename = "/Users/jojoli/Documents/夏校申请:项目制作/英才计划/正式培养/english/datasets/coref/r"
    df = pd.read_csv('/Users/jojoli/Documents/夏校申请:项目制作/英才计划/正式培养/english/datasets/renaissance_preprocessed.csv')
    df = df.head(51)
    all_nouns = []
    all_adjs = []
    for i in range(0, 51):
        result = read_parsing(basename+str(i+1)+".txt.out")
        print("No."+str(i+1)+" done.")
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
    df.to_csv("/Users/jojoli/Documents/夏校申请:项目制作/英才计划/正式培养/english/datasets/renaissance_parsed.csv", index=False)
    print("Parsing analysis completed.\n")


def read_parsing(filename):
    """
    Extract adjectives and nouns (imageries) from the parsing result of each poem
    Return them in a dictionary:
    {'noun':'adj',...}
    """

    with open(filename, 'a') as file_1:
        # add a line that signifies end of file
        file_1.write("\n*********+++++++++")


    # Analyze dependency parses
    dep_cnt = 0
    result = {}
    with open(filename, 'r') as fileDP:

        key = "Dependency Parse"
        dkeylist = ["amod", "nmod:of"]
        templine = fileDP.readline()
        while (templine != "*********+++++++++"):

            # Extract useful relationships (find adjectives and modified nouns)
            for k in dkeylist:
                if k in templine:
                    head = 0
                    end = 0
                    for i in range(0, len(templine)):
                        if templine[i]=='(':
                            head = i + 1
                        if templine[i]=='-':
                            end = i
                            break
                    noun = templine[head:end]
                    for i in range(0, len(templine)):
                        if templine[i]=='-':
                            end = i
                        if templine[i]==',':
                            head = i + 2
                    adj = templine[head:end]
                    result[noun] = adj

            dep_cnt += 1
            templine = fileDP.readline()

    # print(result, '\n')


    # Find NERs
    ner  = {}
    with open(filename, 'r') as fileNER:

        key = "Extracted the following NER entity mentions"
        temp = fileNER.readline()
        while (temp != "*********+++++++++"):
            if key in temp:
                temp = fileNER.readline()
                while not (("Sentence" in temp) or ("Coreference" in temp)):
                    if ("-" in temp) or (":" in temp):
                        wordend = 0
                        for i in range(1, len(temp)):
                            if not (temp[i] >= 'a' and temp[i] <= 'z'):
                                wordend = i
                                break
                        word = temp[0:wordend]
                        wordbegin = 0
                        flag = False
                        for i in range(1, len(temp)):
                            if ((flag == False) and (temp[i] >= 'A') and (temp[i] <= 'Z')):
                                flag = True
                                wordbegin = i
                            elif ((flag == True) and (not ((temp[i] >= 'A') and (temp[i] <= 'Z')))):
                                wordend = i
                                break
                        ner[word] = temp[wordbegin:wordend]
                    temp = fileNER.readline()

            temp = fileNER.readline()

    # print(ner, '\n')


    #  Remove those NERs that are contained in extracted nouns
    for name in ner:
        if name in result:
            result.pop(name)
    # print(result, '\n')

    # print("Done.")
    return result


"""Imagery classification"""
# by the basic sentiment model
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


# featured_df = pd.read_csv("/Users/jojoli/Documents/夏校申请:项目制作/英才计划/正式培养/english/datasets/renaissance_featured.csv")

# Write a function to: FOR EVERY POEM, merge the above four functions and put the returns of last two in featured_df
# it saves the updated featured_df as "..._updated.csv" and returns nothing




# Processing json lines of OAGL source files. Reades all files of a source 
# directory and removes records with short fields or abstracts not in English 
# and applies my core_tokenizer to each of them. 

from __future__ import print_function
import pandas as pd
import numpy as np
from ast import literal_eval
from tqdm import *
from langid.langid import LanguageIdentifier, model
import os, sys, argparse, json, re
from multiprocessing import Pool
from nltk import word_tokenize, sent_tokenize
from nltk.tokenize.treebank import TreebankWordTokenizer

from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.metrics import accuracy_score
from sklearn import linear_model
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
# from gensim.models import doc2vec
from collections import namedtuple
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion

# just lowercase and ascii encode
def lower_key_string(text):
    # # lowercase
    # text = text.lower()
    # remove special characters by performing encode-decode in ascii
    text = text.encode('ascii', 'ignore').decode('ascii')
    return text

# takes list of unique keywords and returns keyword summary string
def summary_from_keywords(key_list):
    # remove any empty strings
    keys = [x for x in key_list if len(x) >= 1]
    # remove trailing spaces
    keys = [x.strip() for x in keys] 
    # remove duplicate keywords
    keys = list(set(keys))
    # generate and return comma-separated keyword string
    key_string = ' , '.join(keys)
    return key_string  

# function that tokenizes text same as Stanford CoreNLP
def core_tokenize(text, alb=False):
    ''' Takes a text string and returns tokenized string using NLTK word_tokenize 
    same as in Stanford CoreNLP. space, \n \t are lost. "" are replace by ``''
    '''
    # tokenize | _ ^ / ~ + = * that are not tokenized by word_tokenize
    text = text.replace("|", " | ") ; text = text.replace("_", " _ ")
    text = text.replace("^", " ^ ") ; text = text.replace("/", " / ")
    text = text.replace("+", " + ") ; text = text.replace("=", " = ")
    text = text.replace("~", " ~ ") ; text = text.replace("*", " * ") 
   
    # tokenize with word_tokenize preserving lines similar to Stanford CoreNLP
    tokens = word_tokenize(text, preserve_line=True)

    # fix the unsplit . problem and Albanian language short forms
    for i, tok in enumerate(tokens):
        if tok == '...':
            continue
        # double match
        if re.match(r'[^.\s]{2,}\.[^.\s]{2,}', tok):
            tokens[i] = tok.replace('.', ' . ')
    	# left match
        if re.match(r'[^.\s]{2,}\.', tok):
            tokens[i] = tok.replace('.', ' . ')
    	# right match
        if re.match(r'\.[^.\s]{2,}', tok):
            tokens[i] = tok.replace('.', ' . ')

        # corrections for albanian texts -- may add n' | t'
        if alb:
            p = re.match(r"(s' | c' | ç')([\w]+)", tok, re.VERBOSE) 
            if p:
                tokens[i] = ' '.join([p.group(1), p.group(2)])

    # put all tokens together
    text = ' '.join(tokens)
    # remove double+ spaces
    text = re.sub(r'\s{2,}', " ", text)

    # lowercase - used only in this script...! (not in general core_tokenize function)
    text = text.lower()
    # remove special characters by performing encode-decode in ascii
    text = text.encode('ascii', 'ignore').decode('ascii')

    return text

# tokenizes text in ["abstract", "title"] fields of a dictionary or dataframe record
def record_tokenize(rec):
	''' Tokenizes ALL fields of a dictionary or dataframe record wich core_tokenize'''
	for k,v in rec.items():
		if k in ["abstract", "title"]:
			rec[k] = core_tokenize(v)
		elif k == "keywords":
			key_str = summary_from_keywords(v)
			rec[k] = lower_key_string(key_str)
	return rec

# function for reading a csv in dataframe trying different encodings
def read_df(fin):
    # trying to read csv file with the main encodings
    try:
        df = pd.read_csv(fin, encoding="utf-8")
    except:
        try:
            df = pd.read_csv(fin, encoding="latin1")  
        except:
            try:
                df = pd.read_csv(fin, encoding="ISO-8859-1")
            except:
                return None 
    return df

# read file json lines from given file path and return them in a list
def read_dicts_from_list(file_path):
    '''read json lines and store them in a list that is returned'''
    with open(file_path, "r", encoding = 'utf-8') as inf:   
        # strip \n at the end of each line
        line_list = [json.loads(line) for line in inf]
    return line_list

# write list records as lines in a given file path
def write_dicts_to_file(file_path, line_list):
	'''write list lines in a file path that is opened'''
	outf = open(file_path, "a", encoding = 'utf-8')  # in this case i need to append to file
	for itm in line_list:
		json.dump(itm, outf)
		outf.write('\n')
	outf.close()

# funciton for dropping short text records or those not in english
def drop_empty_cells(line_list):
    # iterate over each record
    for row in line_list:
        # checking if language is not english
        abstract = row["abstract"]
        t, _ = identifier.classify(abstract)
        # linking all conditions in a short-circuit OR - optimise order...!
        if (t != 'en') or (int(row["plength"]) < MIN_P_LEN) or \
            (len(abstract) < MIN_AB_LEN) or \
            (len(str(row["title"])) < MIN_T_LEN) or \
            (len(row["keywords"]) < MIN_K_LEN) or \
            (len(str(row["title"])) > MAX_T_LEN) or \
            (len(abstract) > MAX_AB_LEN) or \
            (len(row["keywords"]) > MAX_K_LEN) or \
            (int(row["plength"]) > MAX_P_LEN):
            row["abstract"] = ''
            continue 
    # return the reduced list of dictionaries
    dict_list = [d for d in line_list if (d["abstract"] != '')]
    return dict_list

# creating language identifier
identifier = LanguageIdentifier.from_modelstring(model, norm_probs=False)


if __name__ == '__main__': 

	train_list = read_dicts_from_list(os.path.join("data", "train.txt"))
	val_list = read_dicts_from_list(os.path.join("data", "val.txt"))
	test_list = read_dicts_from_list(os.path.join("data", "test.txt"))

	# # limit the number of samples for shorter run time - remove when done
	# LIMIT = 2000
	# if len(line_list) > LIMIT:
	# 	line_list = line_list[:LIMIT]

	# tokenize, lowercase and convert list of keywords to string
	train_list = [record_tokenize(rec) for rec in train_list]
	val_list = [record_tokenize(rec) for rec in val_list]
	test_list = [record_tokenize(rec) for rec in test_list]

	y_train = [int(s["plength"]) for s in train_list]
	y_val = [int(s["plength"]) for s in val_list]
	y_test = [int(s["plength"]) for s in test_list]

	# putting title, abstract and keywords together
	X_train = [x["title"].lower() + " " + x["abstract"].lower() + " " + x["keywords"]  for x in train_list]
	X_val = [x["title"].lower() + " " + x["abstract"].lower() + " " + x["keywords"]  for x in val_list]
	X_test = [x["title"].lower() + " " + x["abstract"].lower() + " " + x["keywords"]  for x in test_list]

	# count_vectorizer = CountVectorizer()
	# count_vectorizer.fit_transform(X)
	# freq_term_matrix = count_vectorizer.transform(X)
	# tfidf = TfidfTransformer(norm="l2")
	# tfidf.fit(freq_term_matrix)
	# tf_idf_matrix = tfidf.transform(freq_term_matrix)
	# X = tf_idf_matrix

	tfidf_vect = TfidfVectorizer(use_idf=True, smooth_idf=True, sublinear_tf=False, ngram_range=(2,2))
	bow = CountVectorizer(ngram_range=(2,2))
	hash_vect = HashingVectorizer(ngram_range=(2,2))
	combined_features = FeatureUnion([("tfidf_vect", tfidf_vect)]) #, ("bow", bow), ("hash",hash_vect)])
	X_train = combined_features.fit_transform(X_train)
	X_val = combined_features.fit_transform(X_val)
	X_test = combined_features.fit_transform(X_test)
	# print(X.toarray())

	# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)
	regr = MLPRegressor(random_state=1, max_iter=500).fit(X_train, y_train)
	y_pred = regr.predict(X_test)

	# Print MSE MAE R2 scores
	print(f"Samples: {len(y)} \t Train: {len(y_train)} \t Test: {len(y_test)}")
	print(f"MSE: {mean_squared_error(y_test, y_pred):.4f} \t MAE: {mean_absolute_error(y_test, y_pred):.4f} \t R2: {r2_score(y_test, y_pred):.4f}")



'''
Experiments with title + abstract + keywords metadata concatenated and
vectorized with Count and TfIdf vectorizer with their default parameters
'''

from __future__ import print_function
import pandas as pd
import numpy as np
import os, sys, argparse, json, re
from nltk import word_tokenize, sent_tokenize
from nltk.tokenize.treebank import TreebankWordTokenizer

from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from collections import namedtuple
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from gensim.models import KeyedVectors

from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, PredefinedSplit, GridSearchCV
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.ensemble import RandomForestRegressor

# just lowercase and ascii encode
def lower_key_string(text):
	# lowercase
	text = text.lower()
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
def core_tokenize(text):
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

	# put all tokens together
	text = ' '.join(tokens)
	# remove double+ spaces
	text = re.sub(r'\s{2,}', " ", text)

	# lowercase
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

def text_to_wordlist(text, remove_stopwords=False, stem_words=False):
	# Clean the text, with the option to remove stopwords and to stem words.
	
	# Convert words to lower case and split them
	text = text.lower().split()
  
	text = " ".join(text)

	# # Clean the text
	# text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
	# text = re.sub(r"what's", "what is ", text)
	# text = re.sub(r"\'s", " ", text)
	# text = re.sub(r"\'ve", " have ", text)
	# text = re.sub(r"can't", "cannot ", text)
	# text = re.sub(r"n't", " not ", text)
	# text = re.sub(r"i'm", "i am ", text)
	# text = re.sub(r"\'re", " are ", text)
	# text = re.sub(r"\'d", " would ", text)
	# text = re.sub(r"\'ll", " will ", text)
	# text = re.sub(r",", " ", text)
	# text = re.sub(r"\.", " ", text)
	# text = re.sub(r"!", " ! ", text)
	# text = re.sub(r"\/", " ", text)
	# text = re.sub(r"\^", " ^ ", text)
	# text = re.sub(r"\+", " + ", text)
	# text = re.sub(r"\-", " - ", text)
	# text = re.sub(r"\=", " = ", text)
	# text = re.sub(r"'", " ", text)
	# text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
	# text = re.sub(r":", " : ", text)
	# text = re.sub(r" e g ", " eg ", text)
	# text = re.sub(r" b g ", " bg ", text)
	# text = re.sub(r" u s ", " american ", text)
	# text = re.sub(r"\0s", "0", text)
	# text = re.sub(r" 9 11 ", "911", text)
	# text = re.sub(r"e - mail", "email", text)
	# text = re.sub(r"j k", "jk", text)
	# text = re.sub(r"\s{2,}", " ", text)
	
	# Return a list of words
	return(text)

if __name__ == '__main__': 

	# read the data
	train_list = read_dicts_from_list(os.path.join("data", "train.txt"))
	val_list = read_dicts_from_list(os.path.join("data", "val.txt"))
	test_list = read_dicts_from_list(os.path.join("data", "test.txt"))

	# tokenize, lowercase and convert list of keywords to string
	train_list = [record_tokenize(rec) for rec in train_list]
	val_list = [record_tokenize(rec) for rec in val_list]
	test_list = [record_tokenize(rec) for rec in test_list]

	y_train = [int(s["plength"]) for s in train_list]
	y_val = [int(s["plength"]) for s in val_list]
	y_test = [int(s["plength"]) for s in test_list]

	# putting title, abstract and keywords together
	X_train = [x["keywords"] + " " + x["title"] + " " + x["abstract"] for x in train_list]
	X_val = [x["keywords"] + " " + x["title"] + " " + x["abstract"] for x in val_list]
	X_test = [x["keywords"] + " " + x["title"] + " " + x["abstract"] for x in test_list]

	# trying different vectorizers
	tfidf = TfidfVectorizer()
	bow = CountVectorizer()
	hash = HashingVectorizer()
	comb = FeatureUnion([("tfidf", tfidf), ("bow", bow), ("hash", hash)])
	
	# trying different regressors
	mlp_model = MLPRegressor(random_state=7)
	lr_model = LinearRegression()
	svr_model = SVR()

	# select the vectorizer for each run
	vect = tfidf
	# select the model for each run
	model = svr_model

	# create and fit the pipeline
	pipe_model = Pipeline([("vect", vect), ("model", model)])
	pipe_model.fit(X_train, y_train)
	y_pred = pipe_model.predict(X_test)

	# Print MSE MAE R2 scores
	print(f"Train: {len(y_train)} \t Val: {len(y_val)} \t Test: {len(y_test)}")
	print(f"MSE: {mean_squared_error(y_test, y_pred):.4f} \t MAE: {mean_absolute_error(y_test, y_pred):.4f} \t R2: {r2_score(y_test, y_pred):.4f}")




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
from nltk.corpus import stopwords
stopWords = set(stopwords.words('english'))

from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import SVR
from collections import namedtuple
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from gensim.models import KeyedVectors
from sklearn.impute import SimpleImputer

from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, PredefinedSplit, GridSearchCV
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
import xgboost as xgb
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor, make_column_transformer
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, PowerTransformer, StandardScaler, RobustScaler

# just lowercase and ascii encode
def lower_key_string(text):
	# lowercase
	text = text.lower()
	# remove special characters by performing encode-decode in ascii
	text = text.encode('ascii', 'ignore').decode('ascii')
	return text

# takes list of unique keywords and returns keyword summary string
def summary_from_keywords(key_lst):
	# remove any empty strings
	keys = [x for x in key_lst if len(x) >= 1]
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
		if k in ["abstract", "title", "venue"]:
			rec[k] = core_tokenize(v)
		elif k == "keywords":
			key_str = summary_from_keywords(v)
			rec[k] = lower_key_string(key_str)
	return rec

# read file json lines from given file path and return them in a list
def read_dicts_from_lst(file_path):
	'''read json lines and store them in a list that is returned'''
	with open(file_path, "r", encoding = 'utf-8') as inf:   
		# strip \n at the end of each line
		line_lst = [json.loads(line) for line in inf]
	return line_lst

# write list records as lines in a given file path
def write_dicts_to_file(file_path, line_lst):
	'''write list lines in a file path that is opened'''
	outf = open(file_path, "a", encoding = 'utf-8')  # in this case i need to append to file
	for itm in line_lst:
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

# return the average characters in each word of text
def avg_word(text):
	words = text.split()
	return (sum(len(word) for word in words) / len(words))

# read the data
train_lst = read_dicts_from_lst(os.path.join("data", "train.txt"))
val_lst = read_dicts_from_lst(os.path.join("data", "val.txt"))
test_lst = read_dicts_from_lst(os.path.join("data", "test.txt"))
full_sample_lst = train_lst + val_lst + test_lst

# tokenize, lowercase and convert list of keywords to string
full_sample_lst = [record_tokenize(rec) for rec in full_sample_lst]

# getting the targets
y = pd.DataFrame({"target": [int(s["plength"]) for s in full_sample_lst]})
y = np.ravel(y)

# getting the rest of fields
key_lst = [str(x["keywords"]) for x in full_sample_lst]
title_lst = [str(x["title"]) for x in full_sample_lst]
abst_lst = [str(x["abstract"]) for x in full_sample_lst]
venue_lst = [str(x["venue"]) for x in full_sample_lst]
year_lst = [int(x["year"]) for x in full_sample_lst]
cit_lst = [int(x["n_citation"]) for x in full_sample_lst]

# # putting all features together
X = pd.DataFrame({"keywords": key_lst, "title": title_lst, "abstract": abst_lst, 
	"venue": venue_lst, "year": year_lst, "citations": cit_lst}) #, "num_keys": num_keys_lst, "title_words": title_words_lst, "abst_words": abst_words_lst})

if __name__ == '__main__': 

	# TFIDF vectorizer
	vect = TfidfVectorizer(lowercase=False)
	# create train - test splits
	y_train = y[:4000] ; y_test = y[4000:5000]
	X_train = X[:4000] ; X_test = X[4000:5000]

	# parameters of the vectorizer - 72
	vect_grid = {
	'union__abstract__ngram_range': [(1, 1), (1,2), (1,3)], 
	'union__abstract__stop_words': [stopWords, None], 
	'union__abstract__norm': ['l1', 'l2', None], 'union__abstract__smooth_idf': [True, False],
	'union__abstract__sublinear_tf': [True, False]
	}

	# parameters of the vectorizer - second round - 24
	vect_grid2 = {
	'union__abstract__ngram_range': [(1,2), (1,3)], 
	# 'union__abstract__stop_words': [stopWords, None], 
	'union__abstract__norm': ['l1', 'l2', None], 
	'union__abstract__smooth_idf': [True, False],
	'union__abstract__sublinear_tf': [True, False]
	}

	# parameters for linear regression - 8
	lr_grid = {'reg__fit_intercept': [True, False], 'reg__normalize': [True, False],
	'reg__copy_X': [True, False]}

	# parameters for support vector regression - 80
	svr_grid = {'reg__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
	'reg__gamma': ['scale', 'auto'], 'reg__C': [0.01, 0.1, 1, 10, 100],
	'reg__shrinking': [True, False]}

	# parameters for multilayer perceptron regressor - 25
	mlp_grid = {'reg__hidden_layer_sizes': [(50, ), (75, ), (100, ), (125, ), (150, )],
	'reg__alpha': [0.00005, 0.0001, 0.0005, 0.001, 0.005],
	'solver': ['lbfgs', 'sgd', 'adam']}

	# parameters for multilayer perceptron regressor - 9
	mlp_grid2 = {'reg__hidden_layer_sizes': [(50, ), (75, ), (100, )],
	'reg__alpha': [0.00005, 0.0001, 0.0005],
	# 'solver': ['lbfgs', 'sgd', 'adam']
	}

	# parameters for random forest regressor - 72
	rfr_grid = {'reg__n_estimators': [50, 75, 100, 125, 150, 200],
	'reg__max_features': ["auto", "sqrt", "log2"], 'reg__bootstrap': [True, False],
	'reg__oob_score': [True, False]}

	# parameters for random forest regressor - second round - 48
	rfr_grid2 = {'reg__n_estimators': [30, 40, 50, 60],
	'reg__max_features': ["auto", "sqrt", "log2"], 'reg__bootstrap': [True, False],
	'reg__oob_score': [True, False]}

	# parameters for gradient boosting regressor - 162
	gb_grid = {'reg__n_estimators': [50, 75, 100, 125, 150, 200],
	'reg__criterion': ['friedman_mse', 'mse', 'mae'], 'reg__max_features': ['auto', 'sqrt', 'log2'],
	'reg__max_depth': [2, 3, 4]}

	# parameters for gradient boosting regressor - second round - 36
	gb_grid2 = {'reg__n_estimators': [30, 40, 50, 60],
	'reg__max_features': ['auto', 'sqrt', 'log2'],
	'reg__max_depth': [2, 3, 4]}

	# parameters for xg boosting regressor - 360
	xg_grid = {'reg__n_estimators': [50, 75, 100, 125, 150, 200], 'reg__gamma': [0, 0.01, 0.1], 
	'reg__eta': [0.1, 0.2, 0.3, 0.4], 'reg__max_depth': [4, 5, 6, 7, 8]}

	# parameters for xg boosting regressor - new - 108
	xg_grid2 = {'reg__n_estimators': [40, 50, 60], 'reg__gamma': [0.005, 0.01, 0.05], 
	'reg__eta': [0.05, 0.1, 0.15], 'reg__max_depth': [2, 3, 4, 5]}

	# parameters for xg boosting regressor - trial 3 - 192
	xg_grid3 = {'reg__n_estimators': [45, 50, 55, 60], 'reg__gamma': [0.002, 0.005, 0.007, 0.01], 
	'reg__eta': [0.03, 0.05, 0.07, 0.1], 'reg__max_depth': [2, 3, 4]}

	# trying different regressors
	lr_model = LinearRegression()
	svr_model = SVR()
	mlp_model = MLPRegressor(random_state=7)
	rfr_model = RandomForestRegressor(random_state=7, n_jobs=-1)
	gb_model = GradientBoostingRegressor(random_state=7)
	xg_model = xgb.XGBRegressor(random_state=7)

	# select one model to try 
	model = mlp_model
	# select the parameter grid of the model to try 
	model_grid = mlp_grid2 

	# creating pgrid joining vect_grid with model_grid
	pgrid = {**vect_grid2, **model_grid}

	# the vectorizer and model pipeline
	pipe_model = Pipeline([
		('union', ColumnTransformer(
			[('keywords', vect, 0),
			('title', vect, 1),
			('abstract', vect, 2),
			('venue', vect, 3),
			], remainder='passthrough')),
		('reg', model)])

	# fit the model and prepare the gridsearch
	pipe_model.fit(X_train, y_train)
	# get the predictions from the default model
	y_pred_model = pipe_model.predict(X_test)

	gs = GridSearchCV(estimator=pipe_model, param_grid=pgrid, cv=4, n_jobs=-1)
	grid_result = gs.fit(X_train, y_train)
	best_model = grid_result.best_estimator_
	# get the predictions from the best model
	y_pred_grid = best_model.predict(X_test)
	
	# Print MSE MAE R2 scores for default and best model
	print(f"Default model scores:")
	print(f"MSE: {mean_squared_error(y_test, y_pred_model):.4f} \t MAE: {mean_absolute_error(y_test, y_pred_model):.4f} \t R2: {r2_score(y_test, y_pred_model):.4f}")
	print(f"Best model scores:")
	print(f"MSE: {mean_squared_error(y_test, y_pred_grid):.4f} \t MAE: {mean_absolute_error(y_test, y_pred_grid):.4f} \t R2: {r2_score(y_test, y_pred_grid):.4f}")
	print(f"Best model params:")
	print(f"{grid_result.best_params_}")

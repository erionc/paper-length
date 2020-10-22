
'''
Author:	Erion Çano
Descri:	Experiments with title, abstract, keywords, venue text fields 
        independently vectorized using TfIdf and publication year and 
        number of citations joined as numeric features. An MLP, a Linear 
        Regressor, a Support Vector Regressor, a Random Forest, a Gradient 
        Boosting and and Extreme Gradient Boosting regressor are used as 
        paper length predictors.
Langu: 	Python 3.6.9
Usage:	python set3_grid.py --regressor REG
'''

import pandas as pd
import numpy as np
import os, sys, argparse, json, re
from nltk import word_tokenize, sent_tokenize
from nltk.tokenize.treebank import TreebankWordTokenizer
from nltk.corpus import stopwords
stopWords = set(stopwords.words('english'))

from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV 
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler

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
	''' 
	Takes a text string and returns tokenized string using NLTK word_tokenize 
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

# tokenizes text in ["abstract", "title", "keywords"] fields
def record_tokenize(rec):
	''' Tokenizes ALL fields of a dictionary or dataframe record with core_tokenize'''
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
	with open(file_path, "r", encoding='utf-8') as inf:   
		# strip \n at the end of each line
		line_list = [json.loads(line) for line in inf]
	return line_list

# write list records as lines in a given file path
def write_dicts_to_file(file_path, line_list):
	'''write list lines in a file path that is opened'''
	outf = open(file_path, "a", encoding='utf-8')  
	for itm in line_list:
		json.dump(itm, outf)
		outf.write('\n')
	outf.close()

# read the data
train_lst = read_dicts_from_list(os.path.join("data", "train.txt"))
val_lst = read_dicts_from_list(os.path.join("data", "val.txt"))
test_lst = read_dicts_from_list(os.path.join("data", "test.txt"))
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
	"venue": venue_lst, "year": year_lst, "citations": cit_lst}) 

parser = argparse.ArgumentParser()
parser.add_argument('-r', '--regressor', choices=['mlp', 'lr', 'svr', 'rf', 'gb', 'xgb'], help='Regression Model', required=True)
args = parser.parse_args()

if __name__ == '__main__': 

	# TFIDF vectorizer
	vect = TfidfVectorizer(lowercase=False)
	# create train - test splits
	y_train = y[:4000] ; y_test = y[4000:5000]
	X_train = X[:4000] ; X_test = X[4000:5000]

	# parameters of the vectorizer
	vect_grid = {
	'union__abstract__ngram_range': [(1, 1), (1,2), (1,3)],
	'union__abstract__norm': ['l1', 'l2', None], 
	'union__abstract__smooth_idf': [True, False], 
	'union__abstract__stop_words': [stopWords, None], 
	'union__abstract__sublinear_tf': [True, False]}

	# parameters for linear regression
	lr_grid = {
	'reg__copy_X': [True, False],
	'reg__fit_intercept': [True, False], 
	'reg__normalize': [True, False]}

	# parameters for support vector regression 
	svr_grid = {
	'reg__C': [0.01, 0.1, 1, 10, 100],
	'reg__gamma': ['scale', 'auto'], 
	'reg__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
	'reg__shrinking': [True, False]}

	# parameters for multilayer perceptron regressor 
	mlp_grid = {
	'reg__hidden_layer_sizes': [(50, ), (75, ), (100, ), (125, ), (150, )],
	'reg__alpha': [0.00005, 0.0001, 0.0005, 0.001, 0.005],
	'solver': ['lbfgs', 'sgd', 'adam']}

	# parameters for random forest regressor
	rf_grid = {
	'reg__n_estimators': [50, 60, 70, 85, 100, 150, 200],
	'reg__max_features': ["auto", "sqrt", "log2"], 
	'reg__bootstrap': [True, False], 'reg__oob_score': [True, False]}

	# parameters for gradient boosting regressor
	gb_grid = {
	'reg__n_estimators': [50, 60, 70, 85, 100, 150, 200],
	'reg__max_features': ['auto', 'sqrt', 'log2'],
	'reg__max_depth': [2, 3, 4, 5, 6, 7, 8]}

	# parameters for xg boosting regressor
	xgb_grid = {
	'reg__n_estimators': [50, 60, 70, 85, 100, 150, 200], 
	'reg__eta': [0.008, 0.015, 0.03, 0.05, 0.1, 0.15, 0.3],
	'reg__gamma': [0.002, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2], 
	'reg__max_depth': [2, 3, 4, 5, 6, 7, 8]}

	# creating different regressors
	svr_model = SVR()
	lr_model = LinearRegression()
	mlp_model = MLPRegressor(random_state=7)
	rf_model = RandomForestRegressor(random_state=7, n_jobs=-1)
	gb_model = GradientBoostingRegressor(random_state=7)
	xgb_model = xgb.XGBRegressor(random_state=7)

	# selecting the regression model and the respective param grid
	if args.regressor.lower() == "mlp":
		model = mlp_model
		model_grid = mlp_grid 
	elif args.regressor.lower() == "lr":
		model = lr_model
		model_grid = lr_grid 
	elif args.regressor.lower() == "svr":
		model = svr_model
		model_grid = svr_grid 
	elif args.regressor.lower() == "rf":
		model = rf_model
		model_grid = rf_grid 
	elif args.regressor.lower() == "gb":
		model = gb_model
		model_grid = gb_grid 
	elif args.regressor.lower() == "xgb":
		model = xgb_model
		model_grid = xgb_grid 
	else:
		print("Wrong Regressor...")
		sys.exit()

	# creating pgrid joining vect_grid with model_grid
	pgrid = {**vect_grid, **model_grid}

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

	gs = GridSearchCV(estimator=pipe_model, param_grid=pgrid, cv=4, n_jobs=-1, pre_dispatch=16, verbose=1)
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

'''
Auth:	Erion Çano
Desc:	Experiments with title + abstract + keywords metadata concatenated 
        and vectorized with TfIdf, Count, Hash, and Union vectorizers with 
        their default parameters. An MLP regressor (mlp), a Linear Regressor
        (lr) and a Support Vector Regressor (svr) are used as length predictors.
Lang: 	Python 3.6.9
Use:	python set1_default.py --vectorizer VECT --regressor REG 
'''

import numpy as np
import os, sys, argparse, json, re
from nltk import word_tokenize, sent_tokenize
from nltk.tokenize.treebank import TreebankWordTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, HashingVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import FeatureUnion, Pipeline

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

parser = argparse.ArgumentParser()
parser.add_argument('-v', '--vectorizer', choices=['tfidf', 'count', 'hash', 'union'], help='Text Vectorizer', required=True)
parser.add_argument('-r', '--regressor', choices=['mlp', 'lr', 'svr'], help='Regression Model', required=True)
args = parser.parse_args()

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
	count = CountVectorizer()
	hash = HashingVectorizer()
	union = FeatureUnion([("tfidf", tfidf), ("count", count), ("hash", hash)])
	
	# trying different regressors
	mlp_model = MLPRegressor(random_state=7)
	lr_model = LinearRegression()
	svr_model = SVR()

	# selecting the vectorizer 
	if args.vectorizer.lower() == "tfidf":
		vect = tfidf
	elif args.vectorizer.lower() == "count":
		vect = count
	elif args.vectorizer.lower() == "hash":
		vect = hash
	elif args.vectorizer.lower() == "union":
		vect = union
	else:
		print("Wrong vectorizer...")
		sys.exit()

	# selecting the regressor
	if args.regressor.lower() == "mlp":
		model = mlp_model
	elif args.regressor.lower() == "lr":
		model = lr_model
	elif args.regressor.lower() == "svr":
		model = svr_model
	else:
		print("Wrong Regressor...")
		sys.exit()

	# create and fit the pipeline
	pipe_model = Pipeline([("vect", vect), ("model", model)])
	pipe_model.fit(X_train, y_train)
	y_pred = pipe_model.predict(X_test)

	# Print MSE MAE R2 scores
	print(f"Train: {len(y_train)} \t Val: {len(y_val)} \t Test: {len(y_test)}")
	print(f"MSE: {mean_squared_error(y_test, y_pred):.4f} \t MAE: {mean_absolute_error(y_test, y_pred):.4f} \t R2: {r2_score(y_test, y_pred):.4f}")
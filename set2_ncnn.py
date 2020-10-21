
'''
Author:	Erion Çano
Descri:	Experiments with keywords + title + abstract metadata concatenated.
        The regressor is a neural network with a static layer of 300d word 
        embeddings from Glove and Word2Vec (you chose for each run) and one 
        layer from NgramCNN architecture with 3 convolution branches of 
        kernel sizes 1, 2, 3.
Langu: 	Python 3.6.9
Usage:	python set2_ncnn.py --embeddings EMB
'''

import numpy as np
from numpy import array, asarray, zeros
from ast import literal_eval
from tqdm import *
import os, sys, argparse, json, re, random
from nltk import word_tokenize, sent_tokenize
from nltk.tokenize.treebank import TreebankWordTokenizer

from gensim.models import doc2vec
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score
from gensim.models import KeyedVectors
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

import tensorflow
from tensorflow.keras.preprocessing.text import one_hot, Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Embedding, Bidirectional, LSTM, Input
from tensorflow.keras.layers import GlobalMaxPooling1D, Dropout, Conv1D, MaxPooling1D, concatenate

# for reproducibility 
sd = 7
np.random.seed(sd)
random.seed(sd)
tensorflow.random.set_seed(sd)
os.environ['PYTHONHASHSEED'] = str(sd)

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
parser.add_argument('-e', '--embeddings', choices=['gs', 'gb', 'w2v'], help='Embedding type', required=True)
args = parser.parse_args()

EMB_DIR = './embed/'
EMB_SIZE = 300
max_length = 400

# file of word embeddings to open
if args.embeddings.lower() == "gs":
	EMB_FILE = 'glove.6B.300d.txt'
	GLOVE_EMB = True
elif args.embeddings.lower() == "gb":
	EMB_FILE = 'glove.840B.300d.txt'
	GLOVE_EMB = True
elif args.embeddings.lower() == "w2v":
	EMB_FILE = 'GoogleNews-vectors-negative300.txt'
	word2vec = KeyedVectors.load_word2vec_format(os.path.join(EMB_DIR, EMB_FILE), binary=False)
	GLOVE_EMB = False
else:
	print("Wrong embeddings...")
	sys.exit()

if __name__ == '__main__': 

	# reading the data samples from the files
	train_list = read_dicts_from_list(os.path.join("data", "train.txt"))
	val_list = read_dicts_from_list(os.path.join("data", "val.txt"))
	test_list = read_dicts_from_list(os.path.join("data", "test.txt"))

	# tokenize, lowercase and convert list of keywords to string
	train_list = [record_tokenize(rec) for rec in train_list]
	val_list = [record_tokenize(rec) for rec in val_list]
	test_list = [record_tokenize(rec) for rec in test_list]

	# getting the document lengths
	y_train = array([int(s["plength"]) for s in train_list])
	y_val = array([int(s["plength"]) for s in val_list])
	y_test = array([int(s["plength"]) for s in test_list])

	# putting title, abstract and keywords together
	X_train = [x["keywords"] + " " + x["title"] + " " + x["abstract"] for x in train_list]
	X_val = [x["keywords"] + " " + x["title"] + " " + x["abstract"] for x in val_list]
	X_test = [x["keywords"] + " " + x["title"] + " " + x["abstract"] for x in test_list]

	# prepare tokenizer
	t = Tokenizer()
	docs = X_train + X_val + X_test
	t.fit_on_texts(docs)
	vocab_size = len(t.word_index) + 1
	# create a weight matrix for words in training docs
	embedding_matrix = zeros((vocab_size, EMB_SIZE))

	# integer encode the documents
	X_train = t.texts_to_sequences(X_train)
	X_val = t.texts_to_sequences(X_val)
	X_test = t.texts_to_sequences(X_test)

	# pad documents to a max length of 4 words
	X_train = pad_sequences(X_train, maxlen=max_length, padding='post', value=0)
	X_val = pad_sequences(X_val, maxlen=max_length, padding='post', value=0)
	X_test = pad_sequences(X_test, maxlen=max_length, padding='post', value=0)

	# functions to load the whole Glove or w2v embedding into memory
	def load_glove():
		embeddings_index = dict()
		f = open(os.path.join(EMB_DIR, EMB_FILE), encoding='utf-8')
		for line in tqdm(f, desc='reading embeddings'):
			values = line.split()
			word = values[0]
			try:
				coefs = asarray(values[1:], dtype='float32')
			except:
				continue
			embeddings_index[word] = coefs
		f.close()
		print('Loaded %s word vectors.' % len(embeddings_index))
		# create a weight matrix for words in training docs
		for word, i in t.word_index.items():
			embedding_vector = embeddings_index.get(word)
			if embedding_vector is not None:
				embedding_matrix[i] = embedding_vector

	def load_w2v():
		embeddings_index = dict()
		f = open(os.path.join(EMB_DIR, EMB_FILE), encoding='utf-8')
		for line in tqdm(f, desc='reading embeddings'):
			# skip the header line of GoogleNews embeddings file
			if len(line) < 20:
				continue
			values = line.split()
			word = values[0]
			try:
				coefs = asarray(values[1:], dtype='float32')
			except:
				continue
			embeddings_index[word] = coefs
		f.close()
		# create a weight matrix for words in training docs 
		for word, i in t.word_index.items():
			if word in word2vec.vocab:
				embedding_matrix[i] = word2vec.word_vec(word)

	if GLOVE_EMB == True:
		load_glove()
	else:
		load_w2v()

	# train a 1D convnet with global maxpooling
	sequence_input = Input(shape=(max_length,), dtype='int32')
	e = Embedding(vocab_size, EMB_SIZE, weights=[embedding_matrix], 
		input_length=max_length, trainable=False)
	embedded_sequences = e(sequence_input)

	# three convolution branches of 1, 2, 3 kernel sizes
	x = Conv1D(filters=20, kernel_size=1, activation='relu', 
		strides=1)(embedded_sequences)
	x = GlobalMaxPooling1D()(x)
	# x = MaxPooling1D(4)(x)
	# x = Flatten()(x) # flatten if using MaxPooling1D instead of GlobalMaxPooling1D

	y = Conv1D(filters=20, kernel_size=2, activation='relu', 
		strides=1)(embedded_sequences)
	y = GlobalMaxPooling1D()(y)
	z = Conv1D(filters=20, kernel_size=3, activation='relu', 
		strides=1)(embedded_sequences)
	z = GlobalMaxPooling1D()(z)

	# joining the branches together 
	w = concatenate([x, y, z])
	# dense and output layers	
	w = Dense(units=100, activation='relu')(w)
	preds = Dense(1)(w)

	# compile the model
	model = Model(sequence_input, preds)
	model.compile(optimizer='adam', loss='mse', metrics=['mse'])
	print(model.summary())

	# fit the model and get predictions
	model.fit(X_train, y_train, batch_size=32, epochs=5,
	          validation_data=(X_val, y_val), shuffle=False)
	y_pred = model.predict(X_test)

	# Print MSE MAE R2 scores
	print(f"Train: {len(y_train)} \t Val: {len(y_val)} Test: {len(y_test)}")
	print(f"MSE: {mean_squared_error(y_test, y_pred):.4f} \t MAE: {mean_absolute_error(y_test, y_pred):.4f} \t R2: {r2_score(y_test, y_pred):.4f}")
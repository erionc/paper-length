# Paper Length Prediction
Paper length prediction task conceived as a regression problem. This is the code for the paper: 

[How Many Pages? Paper Length Prediction from the Metadata](https://dl.acm.org/doi/10.1145/3443279.3443305) (NLPIR 2020) \
Erion Çano, Ondřej Bojar

## Overview

There are several latent correlations between the length of a document and its publication metadata. Understanding those correlations could be useful for for meta-research and important for optimizing several information retrieval tasks. We conceived the paper length prediction task as a regression learning problem and use this code and OAGL datset we created and released to observe how well this task can be solved with existing machine learning algorithms. 

## Dependencies

The code is written and tested with the following libraries:
- python >= 3.6.9
- numpy >= 1.18.2
- scikit-learn >= 0.23.1
- xgboost >= 1.1.1

## Data

Please download [OAGL dataset](http://hdl.handle.net/11234/1-3257) of article metadata to reproduce the experiments. You first copy its *train.txt*, *test.txt*, and *val.txt* files inside the *data* folder of this repository. You then need to download the Glove word embeddings from [here](https://nlp.stanford.edu/projects/glove/) and copy *glove.6B.300d.txt* and *glove.840B.300d.txt* inside the *embed* folder of this repository. The Google News word embeddings can be downloaded from [here](https://code.google.com/archive/p/word2vec) and should be converted from binary to text format. Please refer to [this post](https://stackoverflow.com/questions/27324292/convert-word2vec-bin-file-to-text) to do that. 

## Experiments

**To observe scores of methods with their default parameters:**

```
python set1_default.py --vectorizer VECT --regressor REG 
```
The VECT can be one of: *tfidf*, *count*, *hash*, *union*. The REG can be one of: *mlp*, *lr*, *svr*.

**To observe scores of a neural network with Glove small, Glove big and Word2vec:**

```
python set2_dense.py --embeddings EMB
```
EMB can be one of: *gs*, *gb*, *w2v*.

**To observe scores of NgramCNN with Glove small, Glove big and Word2vec:**

```
python set2_ncnn.py --embeddings EMB
```
EMB takes the same values as above.

**To observe default and optimized scores of various regression models:**

```
python set3_grid.py --regressor REG
```
REG can be one of: *mlp*, *lr*, *svr*, *rf*, *gb*, *xgb*. 

## Citation

**If using this code or OAGL dataset please cite the following paper:**

Çano Erion, Bojar Ondřej: How Many Pages? Paper Length Prediction from the Metadata.
NLPIR 2020, Proceedings of the the 4th International Conference on Natural Language
Processing and Information Retrieval, Seoul, Korea, December 2020.











## Reporting the length mean and std and plotting the length distribution
## for each sample field

import json, gzip, re, string
import glob, os, sys, gc
from json import JSONDecoder
from functools import partial
import functools
import itertools
import pandas as pd
# import matplotlib.pyplot as plt
import numpy as np
import textdistance
from tqdm import tqdm

# read file json lines from given file path and return them in a list
def file_lines_to_list(file_path):
    '''read json lines and store them in a list that is returned'''
    with open(file_path, "r") as inf:
        # strip \n at the end of each line
        line_list = [json.loads(line) for line in inf]
    return line_list

# count and return number of lines in a given file path
def count_file_lines(file_path):
	'''counts number of lines in the file path given as argument'''
	fp = open(file_path, "r")
	num_lines = sum(1 for line in fp)
	return num_lines

# remove , from keyword string
def keywords_comma_fix(keys):
    key_lst = ' , '.join(keys)
    key_lst = key_lst.split(' , ')
    fixed_keys = ' '.join(key_lst)
    return fixed_keys

# compute jaccard index between two strings
def jindex(source, target):
	# split and rejoin the target 
	targ = keywords_comma_fix(target)
	# remove punctuation from both strings
	s = source.translate(translator)
	t = targ.translate(translator)
	# compute and return jaccard index
	return textdistance.jaccard(s.split(), t.split())

# compute jaccard indexes between two string lists and return them as a list
def jindex_of_lists(src_lst, tgt_lst):
	# first check inequalith of lengths
	if len(src_lst) != len(tgt_lst):
		print("Different list lengths! Exiting...")
		return
	else:
		return list(map(jindex, src_lst, tgt_lst))

# overlap - unique target tokens y that overlap with one source token x
def tok_overlap(source, target):
	'''computes overlapping ratio between target and source tokens'''
	# remove punctuation in both lists 
	s = source.translate(translator) ; t = target.translate(translator)
	# get unique tokens in both lists 
	t = list(set(t.split())) ; s = list(set(s.split()))
	hits = 0 # counter for overlaps
	if len(t) == 0:
		return 0
	else:
		for tok in t: 
			if tok in s: 
				hits += 1
	return hits / len(t)

# overlap keywords - unique keywords y that overlap with one source token x
def key_overlap(source, target):
	'''computes overlapping ratio between target and source tokens'''
	# remove punctuation and split keyphrases 
	s = source.translate(translator) ; t_lst = target.split(' , ')
	hits = 0 # counter for overlaps
	if len(t_lst) == 0:
		return 0
	else:
		for tok in t_lst: 
			if tok in s: 
				hits += 1
	return hits / len(t_lst)

# count absent keyphrases - keyphrases in target that are not in source
def absent_keyterms(source, target):
	'''count keyphrases in target that do not appear in source'''
	# remove punctuation in the source
	s = source.translate(translator) ; t_lst = target.split(' , ')
	hits = 0 # counter for overlaps
	if len(t_lst) == 0:
		return 0
	else:
		for kw in t_lst:
			if kw not in s:
				hits += 1
	return hits / len(t_lst)

# for removing punctuation - used in jindex function
translator = str.maketrans('', '', string.punctuation)

# path of files to read
read_path = "./data/oagl/filt_samp"

# the big lists to keep fild lengths from all files
tit_len_lst, abs_len_lst, keytok_len_lst, key_len_lst = [], [], [], []
abs_tit_ji, abs_keytok_ji = [], [] 
abs_tit_ol, abs_keytok_ol, abs_keyword_ol = [], [], []
abs_absent_kw = []
page_len_lst, year_len_lst = [], []

if __name__ == "__main__":

    for filename in tqdm(os.listdir(read_path)):
        
        # reset list of current file
        file_rec_lst = []

        # full path of file being read
        fn = os.path.join(read_path, filename)
        print("\nAdding records from %s ..." % fn)
        file_rec_lst = file_lines_to_list(fn)
        file_tit_lst = [d["title"] for d in file_rec_lst if "title" in d]
        file_abs_lst = [d["abstract"] for d in file_rec_lst if "abstract" in d]
        file_key_lst = [d["keywords"] for d in file_rec_lst if "keywords" in d]
        file_page_lst = [int(d["plength"]) for d in file_rec_lst if "plength" in d]
        file_year_lst = [int(d["year"]) for d in file_rec_lst if "year" in d]

        # aggregating in the directory lists of lengths
        tit_len_lst.extend([len(s.split()) for s in file_tit_lst])
        abs_len_lst.extend([len(s.split()) for s in file_abs_lst])
        page_len_lst.extend([s for s in file_page_lst])
        year_len_lst.extend([s for s in file_year_lst])

        # # apply keywords_comma_fix to get tokens instead of keyphrases
        # keytok_len_lst.extend([len(s.split()) for s in list(map(keywords_comma_fix, file_key_lst))])
        # split with split(' , ') to get keyphrases instead of tokens
        key_len_lst.extend([len(s) for s in file_key_lst])
        
        # compute jacard index of abstract-title and abstract-keytok
        # abs_tit_ji.extend(jindex_of_lists(file_abs_lst, file_tit_lst))
        # abs_keytok_ji.extend(jindex_of_lists(file_abs_lst, list(map(keywords_comma_fix, file_key_lst))))

        # compute overlaps of abstract-title, abstract-keytok and abstract-keywords
        # abs_tit_ol.extend(list(map(tok_overlap, file_abs_lst, file_tit_lst)))
        # abs_keytok_ol.extend(list(map(tok_overlap, file_abs_lst, file_key_lst)))
        # abs_keyword_ol.extend(list(map(key_overlap, file_abs_lst, file_key_lst)))

        # compute absent keywords between abstract and keywords string
        # abs_absent_kw.extend(list(map(absent_keyterms, file_abs_lst, file_key_lst)))

        # clean memory
        del file_rec_lst ; gc.collect()

    # get mean and std of field strings
    tit_arr = np.array(tit_len_lst)
    abs_arr = np.array(abs_len_lst)
    page_arr = np.array(page_len_lst)
    year_arr = np.array(year_len_lst)
    # keytok_arr = np.array(keytok_len_lst)
    key_arr = np.array(key_len_lst)
    # abs_tit_arr = np.array(abs_tit_ji)
    # abs_keytok_arr = np.array(abs_keytok_ji)
    # abs_tit_ol_arr = np.array(abs_tit_ol)
    # abs_keytok_ol_arr = np.array(abs_keytok_ol)
    # abs_keyword_ol_arr =  np.array(abs_keyword_ol)
    # abs_absent_kw_arr = np.array(abs_absent_kw)

    print("title \t\t mean: {:.2f} \t std: {:.2f}".format(np.mean(tit_arr), np.std(tit_arr)))
    print("abstract \t mean: {:.2f} \t std: {:.2f}".format(np.mean(abs_arr), np.std(abs_arr)))
    # print("keyword tokens - mean: {:.2f} std: {:.2f}".format(np.mean(keytok_arr), np.std(keytok_arr)))
    print("keywords \t mean: {:.2f} \t std: {:.2f}".format(np.mean(key_arr), np.std(key_arr)))
    print("page \t\t mean: {:.2f} \t std: {:.2f}".format(np.mean(page_arr), np.std(page_arr)))
    print("year \t\t mean: {:.2f} \t std: {:.2f}".format(np.mean(year_arr), np.std(year_arr)))
    # print("abstract-title jaccard index - mean: {:.4f} std: {:.4f}".format(np.mean(abs_tit_arr), np.std(abs_tit_arr)))
    # print("abstract-keytok jaccard index - mean: {:.4f} std: {:.4f}".format(np.mean(abs_keytok_arr), np.std(abs_keytok_arr)))
    # print("abstract-title overlap index - mean: {:.4f} std: {:.4f}".format(np.mean(abs_tit_ol_arr), np.std(abs_tit_ol_arr)))
    # print("abstract-keytok overlap index - mean: {:.4f} std: {:.4f}".format(np.mean(abs_keytok_ol_arr), np.std(abs_keytok_ol_arr)))
    # print("abstract-keyword overlap index - mean: {:.4f} std: {:.4f}".format(np.mean(abs_keyword_ol_arr), np.std(abs_keyword_ol_arr)))
    # print("abstract-keyword absent keywords - mean: {:.4f} std: {:.4f}".format(np.mean(abs_absent_kw_arr), np.std(abs_absent_kw_arr)))


    # # plot the histogram of title length distribution
    # plt.hist(tit_len_lst, bins=np.arange(max(tit_len_lst)), histtype='step', linewidth=1)
    # plt.title("Length Distribution of {}".format("title"), fontsize=14, fontweight='bold')
    # plt.xlabel('length in # tokens', fontsize=12, fontweight='bold')
    # plt.ylabel('# samples', fontsize=12, fontweight='bold')
    # plt.show()

    # # plot the histogram of abstract length distribution
    # plt.hist(abs_len_lst, bins=np.arange(max(abs_len_lst)), histtype='step', linewidth=1)
    # plt.title("Length Distribution of {}".format("abstract"), fontsize=14, fontweight='bold')
    # plt.xlabel('length in # tokens', fontsize=12, fontweight='bold')
    # plt.ylabel('# samples', fontsize=12, fontweight='bold')
    # plt.show()

    # # plot the histogram of keytoks length distribution
    # plt.hist(keytok_len_lst, bins=np.arange(max(keytok_len_lst)), histtype='step', linewidth=1)
    # plt.title("Length Distribution of {}".format("keyword tokens"), fontsize=14, fontweight='bold')
    # plt.xlabel('length in # tokens', fontsize=12, fontweight='bold')
    # plt.ylabel('# samples', fontsize=12, fontweight='bold')
    # plt.show()

    # # plot the histogram of keywords length distribution
    # plt.hist(key_len_lst, bins=np.arange(max(key_len_lst)), histtype='step', linewidth=1)
    # plt.title("Length Distribution of {}".format("keywords"), fontsize=14, fontweight='bold')
    # plt.xlabel('length in # tokens', fontsize=12, fontweight='bold')
    # plt.ylabel('# samples', fontsize=12, fontweight='bold')
    # plt.show()

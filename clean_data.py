# -*- coding: utf-8 -*-
"""
Created on Sun Apr 10 09:13:57 2022

@author: FourR
"""
import re
import string
import numpy as np
from pickle import dump
import io


# load doc into memory
def load_doc(filename):
    # open the file as read only
    file = io.open(filename, 'rt', encoding='utf8')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


# Each line contains a single pair of phrases,
# first English and then German, separated by a tab character.

# split a loaded document into sentences
def to_pairs(doc):
    lines = doc.strip().split('\n')
    pairs = [line.split('\t') for line in lines]
    new_pairs = [pair[:2] for pair in pairs]
    return new_pairs


# clean a list of lines
def clean_pairs(lines):
    cleaned = list()
    # prepare translation table for removing punctuation
    table = str.maketrans('', '', string.punctuation)
    for pair in lines:
        clean_pair = list()
        for line in pair:
            # tokenize on white space
            line = line.split()
            # remove punctuation from each token
            line = [word.translate(table) for word in line]
            line = [re.sub(r'\([^)]*\)', '', word) for word in line]
            line = [re.sub(r'\<[^>]*\)', '', word) for word in line]
            line = [re.sub(r'\[([A-Za-z0-9_:/*$%!@?#&]+)\]', '', word) for word in line]
            # store as string
            clean_pair.append(' '.join(line))
        cleaned.append(clean_pair)
    return np.array(cleaned)


# save a list of clean sentences to file
def save_clean_data(sentences, filename):
    dump(sentences, open(filename, 'wb'))
    print('Saved: %s' % filename)


# load dataset
filename = 'pes.txt'
doc = load_doc(filename)
# split into english-persian pairs
pairs = to_pairs(doc)
print(pairs[:5])
# clean sentences
clean_pairs = clean_pairs(pairs)
print(clean_pairs[1:2])
# save clean pairs to file
save_clean_data(pairs, 'english-persian.pkl')
# spot check
for i in range(100):
    print('[%s] => [%s]' % (clean_pairs[i, 0], clean_pairs[i, 1]))
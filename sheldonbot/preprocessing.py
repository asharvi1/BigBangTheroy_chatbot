from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unicodedata
from io import open
import re
from sheldonbot.vocabulary import Voc

# Preprocessing the text
def unicodeToAscii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s

def readVocs(datafile):
    print('Reading lines...')
    lines = open(datafile, encoding = 'utf-8').read().strip().split('\n')
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    return pairs

def filterPair(p, max_length):
    return len(p[0].split(' ')) < max_length and len(p[1].split(' ')) < max_length

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def loadPrepareData(corpus_name, movies_datafile, bigbang_datafile):
    print('Start preparing training data ...')
    movies_pairs = readVocs(movies_datafile)
    bigbang_pairs = readVocs(bigbang_datafile)
    voc = Voc(corpus_name)

    print('Filtering Movies Pairs ...')
    print('Read {!s} sentence pairs'.format(len(movies_pairs)))
    movies_pairs = filterPairs(movies_pairs)
    print('Trimmed to {!s} sentence pairs'.format(len(movies_pairs)))
    print('counting words ...')
    for pair in movies_pairs:
        voc.addSentence(pair[0])
        voc.addSentence(pair[1])

    print('Filtering Big Bang Pairs ...')
    print('Read {!s} sentence pairs'.format(len(bigbang_pairs)))
    bigbang_pairs = filterPairs(bigbang_pairs)
    print('Trimmed to {!s} sentence pairs'.format(len(bigbang_pairs)))
    print('counting words ...')
    for pair in bigbang_pairs:
        voc.addSentence(pair[0])
        voc.addSentence(pair[1])

    print('Counter words:', voc.num_words)

    return voc, movies_pairs, bigbang_pairs

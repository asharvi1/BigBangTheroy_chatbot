from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import itertools
from sheldonbot.vocabulary import Tokens


tokens = Tokens()

# Creating batches for training
def sentence_indexes(voc, sentence, tokens = tokens):
    return [voc.word2index[word] for word in sentence.split(' ')] + [tokens.EOS_token]

def zero_padding(l, fillvalue=tokens.PAD_token):
    return list(itertools.zip_longest(*l, fillvalue=fillvalue))

def binary_matrix(l, value=tokens.PAD_token):
    m = []
    for i, seq in enumerate(l):
        m.append([])
        for token in seq:
            if token == tokens.PAD_token:
                m[i].append(0)
            else:
                m[i].append(1)
    return m

def input_seq(l, voc):
    indexes_batch = [sentence_indexes(voc, sentence) for sentence in l]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zero_padding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths

def output_seq(l, voc):
    indexes_batch = [sentence_indexes(voc, sentence) for sentence in l]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zero_padding(indexes_batch)
    mask = binary_matrix(padList)
    mask = torch.ByteTensor(mask)
    padVar = torch.LongTensor(padList)
    return padVar, mask, max_target_len

def batch2train(voc, pair_batch):
    pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
    input_batch, output_batch = [], []
    for pair in pair_batch:
        input_batch.append(pair[0])
        output_batch.append(pair[1])
    inp, lengths = input_seq(input_batch, voc)
    output, mask, max_target_len = output_seq(output_batch, voc)
    return inp, lengths, output, mask, max_target_len

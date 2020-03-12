import numpy as np
import torch
from torch.utils.data import Dataset
import json


SOS_TOKEN, EOS_TOKEN, UNK = 0, 1, 2


def json_to_dict(filename):
    with open(filename) as infile:
        return json.load(infile)


def read_dataset(filename, source_dictionary, target_dictionary):
    with open(filename, "r") as infile:
        for line in infile:
            src = [SOS_TOKEN] + [source_dictionary.get(x, UNK) for x in line.strip().split(" ")] + [EOS_TOKEN]
            tgt = [SOS_TOKEN] + [target_dictionary.get(x, UNK) for x in line.strip().split(" ")] + [EOS_TOKEN]
            yield src, tgt


class NMTData(Dataset):
    def __init__(self, TRAIN, SRC_DICT, TGT_DICT):
        self.source = list()
        self.target = list()
        self.source_mask = list()
        self.target_mask = list()
        self.maxlen_source = 0
        self.maxlen_target = 0
        self.length = 0

        source_dict = json_to_dict(SRC_DICT)
        target_dict = json_to_dict(TGT_DICT)

        for item in read_dataset(TRAIN, source_dict, target_dict):
            self.maxlen_source = max(self.maxlen_source, len(item[0]))
            self.maxlen_target = max(self.maxlen_target, len(item[1]))
            self.source.append(item[0])
            self.target.append(item[1])

        for i in range(self.source):
            diff = self.maxlen_source - len(self.source[i])
            self.source_mask[i] += [False] * len(self.source[i])
            self.source[i] += [EOS_TOKEN] * diff
            self.source_mask[i] += [True] * diff

        for i in range(self.target):
            diff = self.maxlen_target - len(self.target[i])
            self.target_mask[i] += [False] * len(self.target[i])
            self.target[i] += [EOS_TOKEN] * diff
            self.target_mask[i] += [True] * diff

        self.source = torch.from_numpy(np.asarray(self.source))
        self.target = torch.from_numpy(np.asarray(self.target))
        self.source_mask = torch.ByteTensor(self.source_mask)
        self.target_mask = torch.ByteTensor(self.target_mask)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.train_source[index], self.train_target[index], self.source_mask[index], self.target_mask[index]


def idxs_to_sentences(tgt_list, tgt_vocab):
    """
    Given a tensor of a batch of token indices, return the list of sentences
    :param tgt_tokens: (N, T)
    :param tgt_vocab: {idx: BPE subword}
    :return: [str]
    """
    sentences = list()

    for sentence in tgt_list:
        sent = [tgt_vocab.get(x) for x in sentence]
        sentences.append(' '.join(sent))

    return sentences
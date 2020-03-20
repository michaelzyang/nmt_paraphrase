import numpy as np
import torch
from torch.utils.data import Dataset
import json


PAD, SOS_TOKEN, EOS_TOKEN, UNK = 0, 1, 2, 3

def json_to_dict(filename):
    with open(filename) as infile:
        return json.load(infile)


def read_dataset(source_filename, target_filename, source_dictionary, target_dictionary):
    with open(source_filename, "r") as sfile, open(target_filename) as tfile:
        for src_line, tgt_line in zip(sfile, tfile):
            src = [SOS_TOKEN] + [source_dictionary.get(x, UNK) for x in src_line.strip().split(" ")] + [EOS_TOKEN]
            tgt = [SOS_TOKEN] + [target_dictionary.get(x, UNK) for x in tgt_line.strip().split(" ")] + [EOS_TOKEN]
            yield src, tgt


class NMTData(Dataset):
    def __init__(self, SOURCE, TARGET, SRC_DICT, TGT_DICT):
        self.source = list()
        self.target = list()
        self.source_mask = list()
        self.target_mask = list()
        self.maxlen_source = 0
        self.maxlen_target = 0
        self.length = 0
        self.src_len = list()
        self.tgt_len = list()

        source_dict = json_to_dict(SRC_DICT)
        target_dict = json_to_dict(TGT_DICT)

        # now_maxlen_source, now_maxlen_target = 0, 0
        self.maxlen_source = 256
        self.maxlen_target = 256
        for item in read_dataset(SOURCE, TARGET, source_dict, target_dict):
            # self.maxlen_source = max(self.maxlen_source, len(item[0]))
            # self.maxlen_target = max(self.maxlen_target, len(item[1]))
            # now_maxlen_source = max(now_maxlen_source, len(item[0]))
            # now_maxlen_target = max(now_maxlen_target, len(item[1]))
            self.source.append(item[0][:self.maxlen_source])
            self.target.append(item[1][:self.maxlen_target])
            self.length += 1

        for i in range(len(self.source)):
            diff = self.maxlen_source - len(self.source[i])
            # diff = now_maxlen_source - len(self.source[i])
            assert diff>=0
            mask = list()
            mask += [False] * len(self.source[i])
            self.src_len.append(len(self.source[i]))
            self.source[i] += [PAD] * diff
            mask += [True] * diff
            self.source_mask.append(mask)

        for i in range(len(self.target)):
            diff = self.maxlen_target - len(self.target[i])
            # diff = now_maxlen_target - len(self.target[i])
            assert diff>=0
            mask = list()
            mask += [False] * len(self.target[i])
            self.tgt_len.append(len(self.target[i]))
            self.target[i] += [PAD] * diff
            mask += [True] * diff
            self.target_mask.append(mask)

        # self.source = torch.from_numpy(np.asarray(self.source))
        # self.target = torch.from_numpy(np.asarray(self.target))
        # self.source_mask = torch.BoolTensor(self.source_mask)
        # self.target_mask = torch.BoolTensor(self.target_mask)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return torch.LongTensor(self.source[index]), torch.LongTensor([self.src_len[index]]), torch.LongTensor(self.target[index]), torch.LongTensor([self.tgt_len[index]])


def idxs_to_sentences(tgt_list, tgt_vocab):
    """
    Given a tensor of a batch of token indices, return the list of sentences
    :param tgt_tokens: (N, T)
    :param tgt_vocab: {idx: BPE subword}
    :return: [str]
    """
    sentences = list()

    for sentence in tgt_list:
        sent = []
        for x in sentence:
            if x == SOS_TOKEN:
                continue
            if x == EOS_TOKEN:
                break
            sent.append(tgt_vocab.get(x))
        sentences.append(' '.join(sent).replace("@@ ", ""))

    return sentences

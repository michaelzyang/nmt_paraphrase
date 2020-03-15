import argparse
import utils
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_processing import NMTData, json_to_dict, SOS_TOKEN, EOS_TOKEN, PAD
from models import TransformerModel, init_weights
from train_test import train, eval_loss, eval_bleu


# ======================= Set Up Environment ======================= #
# System parameters
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument('-i', '--inference', action='store_true')
parser.add_argument('--train-src', type=str, default="data/train.BPE.en")
parser.add_argument('--train-tgt', type=str, default="data/train.BPE.de")
# parser.add_argument('--train-src', type=str, default="data/train_small.BPE.en")
# parser.add_argument('--train-tgt', type=str, default="data/train_small.BPE.de")
parser.add_argument('--train-src-dict', type=str, default="data/train.BPE.en.json")
parser.add_argument('--train-tgt-dict', type=str, default="data/train.BPE.de.json")
parser.add_argument('--dev-src-dict', type=str, default="data/train.BPE.en.json")  # TODO: Accept train dictionary
parser.add_argument('--dev-tgt-dict', type=str, default="data/train.BPE.de.json")  # TODO: Accept train dictionary
parser.add_argument('--dev-src', type=str, default="data/dev.BPE.en")
parser.add_argument('--dev-tgt', type=str, default="data/dev.BPE.de")
# parser.add_argument('--dev-src', type=str, default="data/train_small.BPE.en")
# parser.add_argument('--dev-tgt', type=str, default="data/train_small.BPE.de")
# parser.add_argument('--test-src', type=str, default="data/test.BPE.en")
# parser.add_argument('--test-tgt', type=str, default="data/test.BPE.de")
parser.add_argument('--test-src', type=str, default="data/train_small.BPE.en")
parser.add_argument('--test-tgt', type=str, default="data/train_small.BPE.de")
parser.add_argument('--test-src-dict', type=str, default="data/train.BPE.en.json")  # TODO: Accept train dictionary
parser.add_argument('--test-tgt-dict', type=str, default="data/train.BPE.de.json")  # TODO: Accept train dictionary
parser.add_argument('--save-dir', type=str, default="save/")
# parser.add_argument('--checkpt-path', type=str, default=None)
# Hyperparameters: data
parser.add_argument('-n', '--batch', type=int, default=32)
# Hyperparameters: architecture
parser.add_argument('-d', '--hidden-dim', type=int, default=512)
parser.add_argument('-l', '--max-len', type=int, default=512)
parser.add_argument('-h', '--num_heads', type=int, default=8)
parser.add_argument('--enc-layers', type=int, default=6)
parser.add_argument('--dec-layers', type=int, default=6)
parser.add_argument('-f', '--dim-feedforward', type=int, default=2048)
parser.add_argument('-p', '--dropout', type=float, default=0.1)
parser.add_argument('-a', '--activation', type=str, default='relu', choices=['relu'])
# Hyperparameters: training
parser.add_argument('-e', '--epochs', type=int, default=100)
# Hyperparameters: optimization
parser.add_argument('-o', '--optimizer', type=str, default='adam', choices=['adam', 'sgd'])
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--scheduler', type=str, default='plateau', choices=['none', 'plateau'])
# Hyperparameters: regularization
parser.add_argument('--weight-decay', type=float, default=0.0)
# Hyperparameters: inference
parser.add_argument('-b', '--beam_size', type=int, default=5)
parser.add_argument('--checkpt_path', type=str, default=None)

args = parser.parse_args()
MODE = 'inference' if args.inference else 'train'
if args.save_dir[-1] != '/':
    args.save_dir = args.save_dir + '/'

# Ensure save path exists
args.save_dir = utils.check_save_dir(args.save_dir)

# System config
device = utils.get_device()
print(f"Device = {device}")


# ======================= Load Data ======================= #
# TODO Determine whether label smoothing occurs here or in the train loop

if MODE == 'train':
    train_data = NMTData(args.train_src, args.train_tgt, args.train_src_dict, args.train_tgt_dict)
    train_loader = DataLoader(train_data, batch_size=args.batch, shuffle=True, num_workers=0)
    dev_data = NMTData(args.dev_src, args.dev_tgt, args.dev_src_dict, args.dev_tgt_dict)
    dev_loader = DataLoader(dev_data, batch_size=args.batch, shuffle=False, num_workers=0)
    print(f"Loaded {len(train_data)} training sentences and {len(dev_data)} development sentences.")
else: # MODE == 'inference'
    test_data = NMTData(args.test_src, args.test_tgt, args.test_src_dict, args.test_tgt_dict)
    test_loader = DataLoader(test_data, batch_size=args.batch, shuffle=False, num_workers=8)
    print(f"Loaded {len(test_data)} test sentences.")

subword_to_idx = json_to_dict(args.train_tgt_dict)
idx_to_subword = {v: k for k, v in subword_to_idx.items()}
src_vocab_size = len(json_to_dict(args.train_src_dict))
tgt_vocab_size = len(subword_to_idx)


# ======================= Prepare Torch Objects ======================= #
# Instantiate model and training objects
model = TransformerModel(src_vocab_size, tgt_vocab_size, args.hidden_dim, args.max_len, args.num_heads,
                         args.enc_layers, args.dec_layers, args.dim_feedforward, args.dropout, args.activation,
                         weight_tie=False)
model.to(device)

if MODE == 'train':
    criterion = nn.CrossEntropyLoss(ignore_index=PAD)

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
    else:
        raise NotImplementedError(f"Optimizer of type {args.optimizer} not implemented.")

    if args.scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=0, verbose=True)
    elif args.scheduler == "none":
        scheduler = None
    else:
        raise NotImplementedError(f"Learning rate scheduler of type {args.scheduler} not implemented.")


# Initialize from checkpoint if provided
if MODE == 'train':
    start_epoch = utils.init_load_train(model, args.checkpt_path, optimizer=optimizer, init_fn=None)
    if start_epoch == 1:
        with open(args.save_dir + "params.json", mode='w') as f:
            json.dump(vars(args), f)
else: # MODE == 'inference'
    utils.load_inference(model, args.checkpt_path)


# ======================= Run Script ======================= #
# Run model
if MODE == 'train':
    epochs_left = args.epochs - start_epoch + 1
    train(train_loader, dev_loader, idx_to_subword, SOS_TOKEN, EOS_TOKEN, args.max_len, args.beam_size, model,
          epochs_left, criterion, optimizer, scheduler, save_dir=args.save_dir, start_epoch=start_epoch,
          report_freq=100, device=device)
else: # MODE == 'inference'
    # TODO
    # raise NotImplementedError("Yixin to update beam search method in model and eval_bleu function in train_test.py")
    bleu_score = eval_bleu(model, test_loader, idx_to_subword, SOS_TOKEN, EOS_TOKEN, int(args.max_len / 2), args.beam_size, device=device)
    print(f"The model achieves a BLEU score of {bleu_score} on the provided test dataset.")

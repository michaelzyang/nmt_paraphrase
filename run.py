import argparse
import utils
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from models import TransformerModel, init_weights
from train_test import train


# ======================= Set Up Environment ======================= #
# System parameters
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--inference', action='store_true')
parser.add_argument('--train-src', type=str, default="data/train.BPE.en")
parser.add_argument('--train-tgt', type=str, default="data/train.BPE.de")
parser.add_argument('--train-src-dict', type=str, default="data/train.BPE.en.json")
parser.add_argument('--train-tgt-dict', type=str, default="data/train.BPE.de.json")
parser.add_argument('--dev-src', type=str, default="data/dev.BPE.en")
parser.add_argument('--dev-tgt', type=str, default="data/dev.BPE.de")
parser.add_argument('--dev-src-dict', type=str, default="data/dev.BPE.en.json")
parser.add_argument('--dev-tgt-dict', type=str, default="data/dev.BPE.de.json")
parser.add_argument('--test-src', type=str, default="data/test.BPE.en")
parser.add_argument('--test-tgt', type=str, default="data/test.BPE.de")
parser.add_argument('--test-src-dict', type=str, default="data/test.BPE.en.json")
parser.add_argument('--test-tgt-dict', type=str, default="data/test.BPE.de.json")
parser.add_argument('--save-dir', type=str, default="save/")
parser.add_argument('--checkpt-path', type=str, default="")
# Hyperparameters: data
parser.add_argument('-n', '--batch', type=int, default=256)
# Hyperparameters: architecture
parser.add_argument('s', '--src-vocab-size', type=int)
parser.add_argument('t', '--tgt-vocab-size', type=int)
parser.add_argument('-d', '--hidden-dim', type=int, default=512)
parser.add_argument('-l', '--max-len', type=int, default=512)
parser.add_argument('-h', '--num-heads', type=int, default=8)
parser.add_argument('--enc-layers', type=int, default=6)
parser.add_argument('--dec-layers', type=int, default=6)
parser.add_argument('f', '--dim-feedforward', type=int, default=2048)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('a', '--activation', type=str, default='relu', choices=['relu'])
# Hyperparameters: training
parser.add_argument('-e', '--epochs', type=int, default=50)
# Hyperparameters: optimization
parser.add_argument('o', '--optimizer', type=str, default='adam', choices=['adam', 'sgd'])
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--scheduler', type=str, default='plateau', choices=['none', 'plateau'])
# Hyperparameters: regularization
parser.add_argument('--weight-decay', type=float, default=0.0)
# Hyperparameters: inference
parser.add_argument('-b', '--beam_size', type=int, default=5)

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
    # train_data = datasets.ImageFolder(root=TRAIN_PATH, transform=data_transform)
    # train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    #
    # val_data = datasets.ImageFolder(root=VAL_PATH, transform=data_transform)
    # val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)
    #
    # print(f"Loaded {len(train_data)} training images, {len(val_data)} validation images.")
    # idx_to_class = {idx: data_class for data_class, idx in val_data.class_to_idx.items()}
    # del train_data
    # del val_data
else: # MODE == 'inference'
    # test_data = datasets.ImageFolder(root=TEST_PATH, transform=data_transform)
    # test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)
    # print(f"Loaded {len(test_data)} test images.")
    #
    # # get folder idx to class mapping from val_path, as test_path folder structure does not encode the mapping
    # val_data = datasets.ImageFolder(root=VAL_PATH, transform=data_transform)
    # class_to_idx = val_data.class_to_idx
    # idx_to_class = {idx: data_class for data_class, idx in class_to_idx.items()}
    # del val_data


# ======================= Prepare Torch Objects ======================= #
# Instantiate model and training objects
model = TransformerModel(args.src_vocab_size, args.tgt_vocab_size, args.hidden_dim, args.max_len, args.num_heads,
                         args.enc_layers, args.dec_layers, args.dim_feedforward, args.dropout, args.activation,
                         weight_tie=True)
model.to(device)

if MODE == 'train':
    criterion = nn.CrossEntropyLoss()

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
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
if MODE == 'inference':
    utils.load_inference(model, args.checkpt_path)
else: # MODE == "train"
    start_epoch = utils.init_load_train(model, args.checkpt_path, optimizer=optimizer, init_fn=init_weights)
    if start_epoch == 1:
        with open(args.save_dir + "params.json", mode='w') as f:
            json.dump(args, f)


# ======================= Run Script ======================= #
# Run model
if MODE == 'train':
    epochs_left = args.epochs - start_epoch + 1
    train(train_loader, val_loader, tgt_vocab, sos_token, eos_token, args.max_len, args.beam_size, model, epochs_left,
          criterion, optimizer, scheduler=None, save_dir=args.save_dir, start_epoch=start_epoch, report_freq=0,
          device='gpu')
else: # MODE == 'inference'
    raise NotImplementedError

import argparse
import utils

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from models import Vaswani, init_weights
from train_test import train, eval

# ======================= TODO MY ======================= #
# Parse parameters
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', type=str, default="train", choices=["train", "classify", "verify"])
parser.add_argument('--train-path', type=str, default="11-785hw2p2-s20/train_data/medium/")
parser.add_argument('--val-path', type=str, default="11-785hw2p2-s20/validation_classification/medium/")
parser.add_argument('--test-path', type=str, default="11-785hw2p2-s20/test_classification/")
parser.add_argument('--verification-pairs', type=str, default="11-785hw2p2-s20/test_trials_verification_student.txt")
parser.add_argument('--save-path', type=str, default="save/")
parser.add_argument('--load-path', type=str, default='')

# Hyperparameters: data
parser.add_argument('-b', '--batch', type=int, default=256)
# Hyperparameters: architecture
parser.add_argument('-c', '--block-channels', nargs='+', default=['64', '128', '256', '512'])
parser.add_argument('-l', '--layer-blocks', nargs='+', default=['2', '2', '2', '2'])
parser.add_argument('-k', '--kernel-sizes', nargs='+', default=['3', '3', '3', '3'])
parser.add_argument('-s', '--strides', nargs='+', default=['1', '1', '2', '2']) # 32 => 32 -> 32 -> 16 -> 8 => 1
parser.add_argument('-p', '--pool-size', type=int, default=4)
# Hyperparameters: training
parser.add_argument('-e', '--epochs', type=int, default=50)
# Hyperparameters: optimization
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--schedule', type=str, default='plateau')
# Hyperparameters: regularization
parser.add_argument('--weight-decay', type=float, default=5e-5)

args = parser.parse_args()
MODE = args.mode
TRAIN_PATH = args.train_path
VAL_PATH = args.val_path
TEST_PATH = args.test_path
VERIFICATION_PAIRS = args.verification_pairs
SAVE_PATH = args.save_path
if SAVE_PATH[-1] != '/':
    SAVE_PATH = SAVE_PATH + '/'
LOAD_PATH = args.load_path

BATCH_SIZE = args.batch
IN_CHANNELS = 3
BLOCK_CHANNELS = [int(x) for x in args.block_channels]
LAYER_BLOCKS = [int(x) for x in args.layer_blocks]
KERNEL_SIZES = [int(x) for x in args.kernel_sizes]
STRIDES = [int(x) for x in args.strides]
POOL_SIZE = args.pool_size
N_EPOCHS = args.epochs
LR = args.lr
MOMENTUM = args.momentum
SCHEDULE = args.schedule
W_DECAY = args.weight_decay

# Ensure save path exists
SAVE_PATH = utils.check_save_path(SAVE_PATH)

# System config
device = utils.get_device()
print(f"Device = {device}")


# ======================= TODO RM ======================= #

# TODO Define Dataset class
# TODO BPE
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



# ======================= TODO ALL ======================= #
# TODO Determine how to define the Vaswani et al. 'big' model using nn.Transformer

# Instantiate model and training objects
model = Resnet(IN_CHANNELS, BLOCK_CHANNELS, LAYER_BLOCKS, KERNEL_SIZES, STRIDES, POOL_SIZE, N_CLASSES)
model.to(device)


# ======================= TODO MY ======================= #
# TODO Write utils

if MODE == "train":
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, weight_decay=W_DECAY, momentum=MOMENTUM)
    if SCHEDULE == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=0, verbose=True)
    elif SCHEDULE.lower() == "none":
        scheduler = None
    else:
        raise NotImplementedError(f"Learning rate scheduler of type {SCHEDULE} not implemented.")


# Initialize from checkpoint if provided
if MODE == 'inference':
    utils.load_inference(model, LOAD_PATH)
else: # MODE == "train"
    start_epoch = utils.init_load_train(model, LOAD_PATH, optimizer=optimizer, init_fn=init_weights)
    if start_epoch == 1:
        with open(SAVE_PATH + "params.txt", mode='w') as f:
            f.writelines([
                f"Batch size: {BATCH_SIZE}\n",
                f"Block channels: {BLOCK_CHANNELS}\n",
                f"Layer blocks: {LAYER_BLOCKS}\n",
                f"Kernel sizes: {KERNEL_SIZES}\n",
                f"Strides: {STRIDES}\n",
                f"Pooling kernel size: {POOL_SIZE}\n",
                f"Learning rate: {LR}\n",
                f"Momentum: {MOMENTUM}\n",
                f"Scheduler: {SCHEDULE}\n",
                f"Weight decay: {W_DECAY}\n",
                str(model) + '\n',
                str(optimizer)
            ])

# ======================= TODO YL ======================= #
# TODO Write train loop using teacher forcing and MLE
# TODO Write eval loop using beam search and BLEU
# Run model
if MODE == 'train':
    # train(train_loader, val_loader, N_EPOCHS, model, criterion, optimizer, scheduler, SAVE_PATH, start_epoch, device, task='Classification')
    raise NotImplementedError
else: # MODE == 'inference'
    # # construct list of filenames
    # idx_to_file = [x.split('/')[-1] for x, _ in test_data.imgs]
    #
    # # predict labels
    # predictions = eval(test_loader, model, device, out_type="classes") # shape (N, )
    # predictions = predictions.cpu().detach().numpy()
    # predictions = np.array([idx_to_class[x] for x in predictions]) # map ImageFolder indices back to classes
    #
    # # save predictions in Kaggle submission format
    # submit_table = np.vstack([idx_to_file, predictions]).T
    # np.savetxt(SAVE_PATH + "predictions.csv", submit_table, fmt='%s', delimiter=',', header="Id,Category", comments='')
    raise NotImplementedError

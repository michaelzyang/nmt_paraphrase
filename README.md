# nmt_paraphrase

## Reimplementation of ["Attention Is All You Need"](https://papers.nips.cc/paper/7181-attention-is-all-you-need.pdf)

build_dictionary.py: building subword dictionary from dataset

data_processing.py: data processing and loading

label_smoothing_loss.py: reimplementation of the label smoothing loss

models.py: reimplementation of the transformer model based on pytorch modules

run.py: main procedure

train_test.py: training, evaluation and test procedure

transformer.py: transformer modules, adapted from [pytorch](https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/transformer.py)

utils.py: helper functions

## Usage

To train a new model, run 
``` shell
python run.py
```
To evaluate a checkpoint with beam search, run
``` shell
python run.py -i -b beam_size --checkpt-path path_to_the_checkpoint
```

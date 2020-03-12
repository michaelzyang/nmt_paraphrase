import numpy as np
import torch
import torch.nn as nn
import sacrebleu  # https://github.com/mjpost/sacreBLEU
from datetime import datetime

from models import TransformerModel
from data_preprocessing import idxs_to_sentences


def train(train_loader, dev_loader, idx_to_subword, sos_token, eos_token, max_len, beam_size, model, n_epochs,
          criterion, optimizer, scheduler=None, save_dir="./", start_epoch=1, report_freq=0, device='gpu'):
    """
    Training procedure, saving the model checkpoint after every epoch
    :param train_loader: training set dataloader
    :param dev_loader: training set dataloader
    :param n_epochs: the number of epochs to run
    :param model: the torch Module
    :param criterion: the loss criterion
    :param optimizer: the optimizer for making updates
    :param scheduler: the scheduler for the learning rate
    :param save_dir: the save directory
    :param start_epoch: the starting epoch number (greater than 1 if continuing from a checkpoint)
    :param device: the torch device used for processing the training
    :param report_freq: report training set loss every report_freq batches
    :return: None
    """
    # Setup
    if save_dir[-1] != '/':
        save_dir = save_dir + '/'

    model.train()
    tgt_mask = nn.Transformer.generate_square_subsequent_mask(max_len)

    print(f"Beginning training at {datetime.now()}")
    if start_epoch == 1:
        with open(save_dir + "results.txt", mode='a') as f:
            f.write("epoch,train_bleu,dev_bleu\n")

    # Train epochs
    for epoch in range(start_epoch, n_epochs + 1):
        avg_loss = 0.0
        for batch_num, batch in enumerate(train_loader):
            # Unpack batch objects
            src_tokens, src_key_padding_mask, tgt_tokens, tgt_key_padding_mask = batch
            src_tokens, src_key_padding_mask = src_tokens.to(device), src_key_padding_mask.to(device)
            tgt_tokens, tgt_key_padding_mask = tgt_tokens.to(device), tgt_key_padding_mask.to(device)

            # Update weights
            optimizer.zero_grad()
            outputs = model(src_tokens, tgt_tokens, src_mask=None, tgt_mask=tgt_mask, memory_mask=None,
                            src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask,
                            memory_key_padding_mask=src_key_padding_mask)
            loss = criterion(outputs, tgt_tokens.long())
            loss.backward()
            optimizer.step()

            # Accumulate loss for reporting
            avg_loss += loss.item()
            if report_freq and (batch_num + 1) % report_freq == 0:
                print(f'Epoch: {epoch}\t\
                      Batch: {batch_num + 1}\t\
                      Avg-Loss: {avg_loss / report_freq:.4f}\t\
                      {datetime.now()}')
                avg_loss = 0.0

            # Cleanup
            torch.cuda.empty_cache()
            del batch, src_tokens, src_key_padding_mask, tgt_tokens, tgt_key_padding_mask, loss

        # Evaluate epoch
        train_bleu = eval_epoch_bleu(model, train_loader, idx_to_subword, sos_token, eos_token, max_len, beam_size, device)
        dev_bleu = eval_epoch_bleu(model, dev_loader, idx_to_subword, sos_token, eos_token, max_len, beam_size, device)
        print(f'Train BLEU: {train_bleu:.2f}\t\
              Development BLEU: {dev_bleu:.2f}\t\
              {datetime.now()}')

        if scheduler:
            scheduler.step(dev_bleu)

        # save epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_BLEU': train_bleu,
            'dev_BLEU': dev_bleu
        }, save_dir + f"checkpoint_{epoch}_{dev_bleu:.2f}.pth")

        with open(save_dir + "results.txt", mode='a') as f:
            f.write(f"{epoch},{train_bleu},{dev_bleu}\n")


def eval_epoch_bleu(model, test_loader, idx_to_subword, sos_token, eos_token, max_len, beam_size, device='gpu'):
    model.eval()
    hyps = []
    refs = []

    with torch.no_grad():
        for batch in test_loader:
            src_tokens, src_key_padding_mask, tgt_tokens, tgt_key_padding_mask = batch
            src_tokens, src_key_padding_mask = src_tokens.to(device), src_key_padding_mask.to(device)
            tgt_tokens, tgt_key_padding_mask = tgt_tokens.to(device), tgt_key_padding_mask.to(device)

            hyp_batch, _ = model.beam_search(src_tokens, src_key_padding_mask, sos_token, eos_token, max_len, beam_size) # (S, N, V)
            hyps.extend(hyp_batch)  # [N]

            ref_batch = idxs_to_sentences(tgt_tokens, idx_to_subword)  # list of single-element lists
            refs.extend(ref_batch)

            torch.cuda.empty_cache()
            del batch, src_tokens, src_key_padding_mask, tgt_tokens, tgt_key_padding_mask

    bleu = sacrebleu.corpus_bleu(hyps, refs)

    model.train()
    return bleu.score

import torch
import torch.nn as nn
import sacrebleu  # https://github.com/mjpost/sacreBLEU
from datetime import datetime

from data_processing import idxs_to_sentences


def train(train_loader, dev_loader, idx_to_subword, sos_token, eos_token, max_len, beam_size, model, n_epochs,
          criterion, optimizer, scheduler=None, save_dir="./", start_epoch=1, report_freq=0, device='gpu'):
    """
    Training procedure, saving the model checkpoint after every epoch
    :param train_loader: training set dataloader
    :param dev_loader: training set dataloader
    :param idx_to_subword: The dictionary for the vocabulary of subword indices to subwords
    :param sos_token: The index of the start of sentence token
    :param eos_token: The index of the end of sentence token
    :param max_len: The maximum length of an output sequence
    :param beam_size: The beam size used for the beam search algorithm when decoding
    :param model: the torch Module
    :param n_epochs: the number of epochs to run
    :param criterion: the loss criterion
    :param optimizer: the optimizer for making updates
    :param scheduler: the scheduler for the learning rate
    :param save_dir: the save directory
    :param start_epoch: the starting epoch number (greater than 1 if continuing from a checkpoint)
    :param report_freq: report training set loss every report_freq batches
    :param device: the torch device used for processing the training
    :return: None
    """
    # Setup
    if save_dir[-1] != '/':
        save_dir = save_dir + '/'

    model.train()
    d_model = 512
    # tgt_mask = model.transformer.generate_square_subsequent_mask(sz=max_len - 1)
    # tgt_mask = tgt_mask.to(device)

    print(f"Beginning training at {datetime.now()}")
    if start_epoch == 1:
        with open(save_dir + "results.txt", mode='a') as f:
            f.write("epoch,dev_loss\n")
    # Train epochs
    all_step_cnt = 0
    warmup_steps = 4000
    for epoch in range(start_epoch, n_epochs + 1):
        avg_loss = 0.0  # for accumulating loss per reporting cycle over batches
        step_cnt = 0  # step counter until update
        for batch_num, batch in enumerate(train_loader):
            # Unpack batch objects
            src_tokens, src_key_padding_mask, src_lens, tgt_tokens, tgt_key_padding_mask, tgt_lens = batch
            max_src_len = torch.max(src_lens)
            src_tokens, src_key_padding_mask = src_tokens[:, :max_src_len], src_key_padding_mask[:, :max_src_len]
            max_tgt_len = torch.max(tgt_lens)
            tgt_tokens, tgt_key_padding_mask = tgt_tokens[:, :max_tgt_len], tgt_key_padding_mask[:, :max_tgt_len]
            src_tokens, src_key_padding_mask = src_tokens.to(device), src_key_padding_mask.to(device)
            tgt_tokens, tgt_key_padding_mask = tgt_tokens.to(device), tgt_key_padding_mask.to(device)
            tgt_mask = model.transformer.generate_square_subsequent_mask(sz=tgt_tokens.size(1) - 1)
            tgt_mask = tgt_mask.to(device)

            # Update weights
            loss = compute_loss(model, src_tokens, tgt_tokens, src_mask=None, tgt_mask=tgt_mask, memory_mask=None,
                                src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask,
                                memory_key_padding_mask=src_key_padding_mask, criterion=criterion)
            # Update only after every 4 batches
            loss_ = loss / 4
            loss_.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), 1)
            step_cnt += 1
            if step_cnt == 4:
                all_step_cnt += 1
                optimizer.step()
                optimizer.zero_grad()
                step_cnt = 0
                lr = (d_model ** (-0.5)) * min(all_step_cnt ** (-0.5), all_step_cnt * (warmup_steps ** (-1.5)))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

            # Accumulate loss for reporting
            avg_loss += loss.item()
            if report_freq and (batch_num + 1) % report_freq == 0:
                print("learning rate: %.6f"%lr)
                print(f'Epoch: {epoch}\tBatch: {batch_num + 1}\tTrain loss: {avg_loss / report_freq:.4f}\t\
                      {datetime.now()}')
                avg_loss = 0.0

            # Cleanup
            torch.cuda.empty_cache()
            del batch, src_tokens, src_key_padding_mask, tgt_tokens, tgt_key_padding_mask, loss, loss_

        # Evaluate epoch
        dev_loss = eval_loss(model, dev_loader, max_len, criterion, device)
        print(f'Epoch {epoch} complete.\tDev loss: {dev_loss:.4f}\t{datetime.now()}')
        with open(save_dir + "results.txt", mode='a') as f:
            f.write(f"{epoch},{dev_loss}\n")

        if scheduler:
            scheduler.step(dev_loss)

        # save epoch checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'dev_loss': dev_loss
        }
        torch.save(checkpoint, save_dir + f"checkpoint_{epoch}_{dev_loss:.4f}.pth")
    print(f"Finished training at {datetime.now()}")


def eval_loss(model, data_loader, max_len, criterion, device='gpu'):
    """
    Evaluates the loss of the model on a given dataset
    :param model: The model being evaluated
    :param data_loader: A dataloader for the data over which to evaluate
    :param max_len: The maximum length of an output sequence
    :param criterion: The loss criterion being computed
    :param device: The torch device used for processing the training
    :return: The average loss per sentence
    """
    model.eval()
    # tgt_mask = model.transformer.generate_square_subsequent_mask(sz=max_len - 1)
    loss_accum = 0.0
    batch_count = 0

    with torch.no_grad():
        for batch in data_loader:
            src_tokens, src_key_padding_mask, src_lens, tgt_tokens, tgt_key_padding_mask, tgt_lens = batch
            max_src_len = torch.max(src_lens)
            src_tokens, src_key_padding_mask = src_tokens[:, :max_src_len], src_key_padding_mask[:, :max_src_len]
            max_tgt_len = torch.max(tgt_lens)
            tgt_tokens, tgt_key_padding_mask = tgt_tokens[:, :max_tgt_len], tgt_key_padding_mask[:, :max_tgt_len]
            src_tokens, src_key_padding_mask = src_tokens.to(device), src_key_padding_mask.to(device)
            tgt_tokens, tgt_key_padding_mask = tgt_tokens.to(device), tgt_key_padding_mask.to(device)
            tgt_mask = model.transformer.generate_square_subsequent_mask(sz=tgt_tokens.size(1) - 1)
            tgt_mask = tgt_mask.to(device)

            loss = compute_loss(model, src_tokens, tgt_tokens, src_mask=None, tgt_mask=tgt_mask, memory_mask=None,
                                src_key_padding_mask=src_key_padding_mask,
                                tgt_key_padding_mask=tgt_key_padding_mask,
                                memory_key_padding_mask=src_key_padding_mask, criterion=criterion)
            loss_accum += loss
            batch_count += 1

            torch.cuda.empty_cache()
            del batch, src_tokens, src_key_padding_mask, tgt_tokens, tgt_key_padding_mask

    model.train()

    return loss_accum / batch_count


def eval_bleu(model, data_loader, idx_to_subword, sos_token, eos_token, max_len, beam_size=1, device='gpu'):
    """
    Evaluates the BLEU score of the model on a given dataset
    :param model: The model being evaluated
    :param data_loader: A dataloader for the data over which to evaluate
    :param idx_to_subword: The dictionary for the vocabulary of subword indices to subwords
    :param sos_token: The index of the start of sentence token
    :param eos_token: The index of the end of sentence token
    :param max_len: The maximum length of an output sequence
    :param beam_size: The beam size used for the beam search algorithm when decoding
    :param device: The torch device used for processing the training
    :return: The BLEU score out of 100
    """
    model.eval()

    hyps = []
    refs = []

    with torch.no_grad():
        for (i, batch) in enumerate(data_loader):
            if i > 10:
                break
            src_tokens, src_key_padding_mask, src_lens, tgt_tokens, tgt_key_padding_mask, tgt_lens = batch
            max_src_len = torch.max(src_lens)
            src_tokens, src_key_padding_mask = src_tokens[:, :max_src_len], src_key_padding_mask[:, :max_src_len]
            max_tgt_len = torch.max(tgt_lens)
            tgt_tokens, tgt_key_padding_mask = tgt_tokens[:, :max_tgt_len], tgt_key_padding_mask[:, :max_tgt_len]
            src_tokens, src_key_padding_mask = src_tokens.to(device), src_key_padding_mask.to(device)
            tgt_tokens, tgt_key_padding_mask = tgt_tokens.to(device), tgt_key_padding_mask.to(device)
            if beam_size == 1:
                hyp_batch = model.inference(src_tokens, src_key_padding_mask, sos_token, max_len)
            else:
                # only suuports batch size 1
                hyp_batch, _ = model.beam_search(src_tokens, src_key_padding_mask, sos_token, eos_token, max_len, beam_size) # (S, N, V)
            hyp_batch = idxs_to_sentences(hyp_batch, idx_to_subword)
            hyps.extend(hyp_batch)  # [N]
            ref_batch = idxs_to_sentences(tgt_tokens.cpu().numpy().tolist(), idx_to_subword)  # list of single-element lists
            refs.extend(ref_batch)

            torch.cuda.empty_cache()
            del batch, src_tokens, src_key_padding_mask, tgt_tokens, tgt_key_padding_mask
    model.train()
    print(hyps[:5])
    print(refs[:5])
    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    return bleu.score


def compute_loss(model, src_tokens, tgt_tokens, src_mask, tgt_mask, memory_mask, src_key_padding_mask,
                 tgt_key_padding_mask, memory_key_padding_mask, criterion):
    # drop last token for tgt_tokens input as decoder at each time step should attend to all tokens up to prev token
    # shift tgt_key_padding_mask left by one position for the same reason (the eos tag will end up being masked out
    outputs = model(src_tokens, tgt_tokens[:, :-1], src_mask=src_mask, tgt_mask=tgt_mask, memory_mask=memory_mask,
                    src_key_padding_mask=src_key_padding_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask[:, :-1],
                    memory_key_padding_mask=memory_key_padding_mask)
    outputs = outputs.transpose(0, 1).transpose(1, 2)
    loss = criterion(outputs, tgt_tokens[:, 1:].long())
    return loss

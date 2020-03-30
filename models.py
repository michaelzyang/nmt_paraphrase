import torch.nn as nn
import transformer
import torch
import torch.nn.functional as F
import math
from data_processing import EOS_TOKEN

def init_weights(m):
    raise NotImplementedError
    # if type(m) == nn.Conv2d or type(m) == nn.Linear:
    #     nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
    #     # nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')


class BeamCandidate():
    def __init__(self, eos, tokens=None, score=0):
        if tokens is None:
            self.tokens = []
        else:
            self.tokens = tokens
        self.eos = eos
        self.score = score
        self.ended = False

    def update(self, token, prob):
        if token.item() == self.eos:
            self.ended = True
        return BeamCandidate(self.eos, self.tokens + [token], self.score + prob)

    def is_end(self):
        return self.ended


class TransformerModel(nn.Module):
    """ Transformer Model """
    def __init__(self, src_vocab_size, tgt_vocab_size, hidden_dim, max_len, nhead, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, activation='relu', weight_tie=True, sinusoidal=True):
        super(TransformerModel, self).__init__()
        # Embedding layers
        self.src_token_embedding = nn.Embedding(src_vocab_size, hidden_dim)
        if weight_tie == True:
            assert src_vocab_size == tgt_vocab_size
            self.tgt_token_embedding = self.src_token_embedding
        else:
            self.tgt_token_embedding = nn.Embedding(tgt_vocab_size, hidden_dim)

        self.sinusoidal = sinusoidal
        if sinusoidal:
            self.position_embedding = PositionalEncoding(hidden_dim, dropout, max_len)
        else:
            self.position_embedding = nn.Embedding(max_len, hidden_dim)


        # Transformer model
        self.d_model = hidden_dim  # following notation from Vaswani et al.
        self.d_ff = dim_feedforward  # following notation from Vaswani et al.
        self.h = nhead  # following notation from Vaswani et al.
#         self.transformer = nn.Transformer(d_model=hidden_dim, nhead=nhead, num_encoder_layers=num_encoder_layers,
#                                           num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward,
#                                           dropout=dropout, activation=activation)
        self.transformer = transformer.Transformer(d_model=hidden_dim, nhead=nhead, num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward,
                                          dropout=dropout, activation=activation)

        # Output layer
        self.linear = nn.Linear(hidden_dim, tgt_vocab_size)

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def embedding(self, tokens, side="src"):
        # tokens (S, N)
        tokens = tokens.transpose(0, 1)
        if side == "src":
            token_embeddings = self.src_token_embedding(tokens)
        else:
            token_embeddings = self.tgt_token_embedding(tokens)
        if self.sinusoidal:
            return self.position_embedding(token_embeddings * math.sqrt(token_embeddings.size(2)))
        else:
            max_len = tokens.size(0)
            position_idx = torch.arange(max_len).type_as(tokens) # (S, )
            position_embeddings = self.position_embedding(position_idx) # (S, E)
            position_embeddings = position_embeddings.unsqueeze(1).expand(-1, tokens.size(1), -1)
            return token_embeddings + 0.1 * position_embeddings
        
    def forward(self, src_tokens, tgt_tokens, src_mask=None, tgt_mask=None, memory_mask=None,
                src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # tokens (S, N)
        src_embeddings = self.embedding(src_tokens, side="src")
        tgt_embeddings = self.embedding(tgt_tokens, side="tgt")
        output = self.transformer(src_embeddings, tgt_embeddings, src_mask=src_mask, tgt_mask=tgt_mask, memory_mask=memory_mask,
                                  src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask,
                                  memory_key_padding_mask=memory_key_padding_mask)
        # output = self.transformer(src_embeddings, tgt_embeddings, src_mask=src_mask, tgt_mask=tgt_mask, memory_mask=memory_mask)
        # (T, N, E)
        return self.linear(output) # (T, N, V)

    def inference(self, src_tokens, src_key_padding_mask, sos_token, max_len):
        """ greedy decoding """
        src_embeddings = self.embedding(src_tokens, side="src")
        src_encoding = self.transformer.encoder(src_embeddings, src_key_padding_mask=src_key_padding_mask)
        sos_token = torch.ones(1).long().type_as(src_tokens) * sos_token
        batch_size = src_tokens.size(0)
        tgt_tokens = [sos_token.expand(batch_size)]
        is_ended = torch.zeros(batch_size, dtype=torch.bool, device=src_tokens.device)
        src_length = src_tokens.size(1)
        for i in range(min(2 * src_length, max_len)):
            tgt_embeddings = self.embedding(torch.stack(tgt_tokens, dim=1), side="tgt")
            tgt_mask = self.transformer.generate_square_subsequent_mask(sz=tgt_embeddings.size(0)).type_as(tgt_embeddings)
            output = self.transformer.decoder(tgt_embeddings, src_encoding, tgt_mask=tgt_mask, memory_key_padding_mask=src_key_padding_mask) # (T, N, V)
            output = self.linear(output)
            next_logits = output[-1]  # (N, V) logits of the last predicted word
            _, words = torch.max(next_logits, dim=1)  # highest likelihood word (indices)
            tgt_tokens.append(words)
            is_ended += torch.eq(words, EOS_TOKEN)
            if is_ended.sum().item() == batch_size:
                break
            # TODO return if <eos> predicted
        return torch.stack(tgt_tokens[1:], dim=1).cpu().numpy().tolist()

    def beam_search(self, src_tokens, src_key_padding_mask, sos_token, eos_token, max_len, beam_size=5, len_penalty=0.7):
        """ advanced beam search """
        batch_size = src_tokens.size(0)
        src_key_padding_mask = src_key_padding_mask.repeat_interleave(beam_size, dim=0)
        src_tokens = src_tokens.repeat_interleave(beam_size, dim=0)
        src_embeddings = self.embedding(src_tokens, side="src")
        src_encoding = self.transformer.encoder(src_embeddings, src_key_padding_mask=src_key_padding_mask)
        sos_token = torch.ones(1).long().type_as(src_tokens) * sos_token
        tgt_tokens = sos_token.expand(batch_size, beam_size).unsqueeze(dim=2) # [batch_size, beam_size, seq_len]
        scores = torch.zeros(batch_size, beam_size, device=src_tokens.device) # keep batch_size, beam_size scores
        lengths = torch.zeros(batch_size, beam_size, device=src_tokens.device) # keep batch_size, beam_size lengths
        is_ended = torch.zeros(batch_size, beam_size, dtype=torch.bool, device=src_tokens.device) # [batch_size, beam_size]
        src_length = src_tokens.size(1)
        for i in range(min(int(1.2 * src_length + 10), max_len)):
            partial_seq = tgt_tokens
            partial_seq = partial_seq.view(-1, tgt_tokens.size(2)) # [batch_size x beam_size, seq_len]
            tgt_embeddings = self.embedding(partial_seq, side="tgt")
            tgt_mask = self.transformer.generate_square_subsequent_mask(sz=tgt_embeddings.size(0)).type_as(tgt_embeddings)
            output = self.transformer.decoder(tgt_embeddings, src_encoding, tgt_mask=tgt_mask, memory_key_padding_mask=src_key_padding_mask) # (T, N, V)
            if self.weight_share:
                output = F.linear(output, self.tgt_token_embedding.weight)
            else:
                output = self.linear(output)
            next_logits = output[-1]
            next_logits = F.log_softmax(next_logits, dim=1) # [batch_size x beam_size, word_dim]
            log_probs, words = torch.topk(next_logits, dim=1, k=beam_size) # [batch_size x beam_size, beam_size]
            log_probs = torch.stack(log_probs.chunk(batch_size)) # [batch_size, beam_size, beam_size]
            log_probs = log_probs.view(batch_size, -1) # [batch_size, beam_size x beam_size]
            words = torch.stack(words.chunk(batch_size)) # [batch_size, beam_size, beam_size]
            words = words.view(batch_size, -1) # [batch_size, beam_size x beam_size]
            # calculate score
            _is_ended = is_ended.repeat_interleave(beam_size, dim=1) # [batch_size, beam_size x beam_size]
            _lengths = lengths.repeat_interleave(beam_size, dim=1) # [batch_size, beam_size x beam_size]
            _lengths += (~ _is_ended).float() # add 1 for not ended sequences

            scores = scores.repeat_interleave(beam_size, dim=1) # [batch_size, beam_size x beam_size]
            scores = scores.view(batch_size, beam_size, beam_size)
            mask_idx = _is_ended.clone()
            mask_idx = mask_idx.view(batch_size, beam_size, beam_size)
            mask_idx[:, :, 0] = 0
            scores.masked_fill_(mask_idx, float('-inf'))
            scores = scores.view(batch_size, -1)
            scores += (log_probs * (~ _is_ended).float()) # do not update score for ended sequences
            _scores = scores / torch.pow(_lengths, len_penalty) # normalizing with length
            _scores = scores
            if i == 0:
                _scores = _scores[:, :beam_size]
            _, idxs = torch.topk(_scores, dim=1, k=beam_size) # [batch_size, beam_size]

            next_word = torch.gather(words, dim=1, index=idxs) # [batch_size, beam_size]
            prev_tokens = tgt_tokens.repeat_interleave(beam_size, dim=1) # [batch_size, beam_size x beam_size, seq_len]
            first_idx = torch.arange(batch_size).to(prev_tokens.device).repeat_interleave(beam_size)
            second_idx = idxs.view(-1)
            prev_tokens = prev_tokens[first_idx, second_idx]
            prev_tokens = prev_tokens.view(batch_size, beam_size, -1)
            tgt_tokens = torch.cat((prev_tokens, next_word.unsqueeze(2)), dim=2)
            scores = torch.gather(scores, dim=1, index=idxs) # [batch_size, beam_size]
            lengths = torch.gather(_lengths, dim=1, index=idxs) # [batch_size, beam_size]
            is_ended = torch.gather(_is_ended, dim=1, index=idxs) # [batch_size, beam_size]
            is_ended += torch.eq(next_word, EOS_TOKEN)
            if is_ended.sum().item() == batch_size * beam_size:
                break
        _scores = scores / lengths # normalizing with length
        scores, idxs = torch.max(_scores, dim=1) # [batch_size]
        sentences = tgt_tokens
        sentences = sentences[torch.arange(batch_size).to(sentences.device), idxs] # [batch_size, seq_len]
        return sentences[:, 1:].cpu().numpy().tolist(), scores.cpu().numpy().tolist()

# Temporarily leave PositionalEncoding module here. Will be moved somewhere else.
class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens
        in the sequence. The positional encodings have the same dimension as
        the embeddings, so that the two can be summed. Here, we use sine and cosine
        functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


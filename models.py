import torch.nn as nn
import torch
import torch.nn.functional as F


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
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, activation='relu', weight_tie=True):
        super(TransformerModel, self).__init__()
        self.src_token_embedding = nn.Embedding(src_vocab_size, hidden_dim)
        if weight_tie == True:
            assert src_vocab_size == tgt_vocab_size
            self.tgt_token_embedding = self.src_token_embedding
        else:
            self.tgt_token_embedding = nn.Embedding(tgt_vocab_size, hidden_dim)
        self.position_embedding = nn.Embedding(max_len, hidden_dim)
        self.transformer = nn.Transformer(d_model=hidden_dim, nhead=nhead, num_encoder_layers=num_encoder_layers,
                                          num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward,
                                          dropout=dropout, activation=activation)
        self.linear = nn.Linear(hidden_dim, tgt_vocab_size)

    def embedding(self, tokens, side="src"):
        # tokens (S, N)
        if side == "src":
            token_embeddings = self.src_token_embedding(tokens)
        else:
            token_embeddings = self.tgt_token_embedding(tokens)
        max_len = tokens.size(0)
        position_idx = torch.arange(max_len).type_as(max_len) # (S, )
        position_embeddings = self.position_embedding(position_idx) # (S, E)
        position_embeddings = position_embeddings.unsqueeze(1).expand(-1, tokens.size(1), -1)
        return token_embeddings + position_embeddings
        
    def forward(self, src_tokens, tgt_tokens, src_mask=None, tgt_mask=None, memory_mask=None,
                src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # tokens (S, N)
        src_embeddings = self.embedding(src_tokens, side="src")
        tgt_embeddings = self.embedding(src_tokens, side="tgt")
        output = self.transformer(src_embeddings, tgt_embeddings, src_mask=src_mask, tgt_mask=tgt_mask, memory_mask=memory_mask,
                                  src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask,
                                  memory_key_padding_mask=memory_key_padding_mask)
        # (T, N, E)
        return self.linear(output) # (T, N, V)

    def inference(self, src_tokens, src_key_padding_mask, sos_token, max_len):
        """ greedy decoding """
        src_embeddings = self.embedding(src_tokens, side="src")
        src_encoding = self.transformer.encoder(src_embeddings, src_key_padding_mask=src_key_padding_mask)
        tgt_tokens = [sos_token]
        for i in range(max_len):
            tgt_embeddings = self.embedding(torch.stack(tgt_tokens, dim=0), side="tgt")
            output = self.transformer.decoder(tgt_embeddings, src_encoding, memory_key_padding_mask=src_key_padding_mask) # (T, N, V)
            next_logits = output[-1]  # (N, V) logits of the last predicted word
            _, words = torch.max(next_logits, dim=1)  # highest likelihood word (indices)
            tgt_tokens.append(words)
            # TODO return if <eos> predicted
        return torch.stack(tgt_tokens[1:], dim=0).cpu().numpy().tolist()

    def beam_search(self, src_tokens, src_key_padding_mask, sos_token, eos_token, max_len, beam_size=5):
        topk = []
        probs = []
        src_embeddings = self.embedding(src_tokens, side="src")
        src_encoding = self.transformer.encoder(src_embeddings, src_key_padding_mask=src_key_padding_mask)
        tgt_tokens = [sos_token]
        for i in range(max_len):
            if i == 0:
                tgt_embeddings = self.embedding(torch.stack(tgt_tokens, dim=0), side="tgt")
                output = self.transformer.decoder(tgt_embeddings, src_encoding, memory_key_padding_mask=src_key_padding_mask)
                next_logits = output[-1]  # (N, V) logits of the last predicted word
                result = F.log_softmax(next_logits, 1)
                probs, idxs = torch.topk(result, beam_size, dim=1)

                for j in range(beam_size):
                    topk.append(BeamCandidate(eos_token, tgt_tokens + [idxs[:, j]], probs[:, j].item()))
            else:
                candidates = []
                for j in range(beam_size):
                    if topk[j].is_end():
                        candidates.append(topk[j])
                    else:
                        tgt_embeddings = self.embedding(torch.stack(topk[j].tokens, dim=0), side="tgt")
                        output = self.transformer.decoder(tgt_embeddings, src_encoding, memory_key_padding_mask=src_key_padding_mask)
                        next_logits = output[-1]
                        result = F.log_softmax(next_logits, 1)
                        probs, idxs = torch.topk(result, beam_size, dim=1)

                        for q in range(beam_size):
                            candidates.append(topk[j].update(idxs[:, q], probs[:, q].item()))
                candidates = sorted(candidates, key=lambda x: x.score, reverse=True)
                topk = [candidates[_] for _ in range(beam_size)]
        return torch.stack(topk[0].tokens[1:], dim=0).cpu().numpy().tolist(), topk[0].score






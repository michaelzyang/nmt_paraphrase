import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F


class Resnet(nn.Module):
    raise NotImplementedError
    # def __init__(self, in_channels, block_channels, layer_blocks, kernel_sizes, strides, pool_size, num_classes):
    #     """
    #     Adapted from Recitation 6 code
    #     :param in_channels: the number of channels in the input data
    #     :param block_channels: the number of channels in each layer
    #     :param layer_blocks: the number of consecutive blocks in each layer
    #     :param strides: stride at the end of each layer
    #     :param num_classes:
    #     :param feat_dim:
    #     """
    #     super(Resnet, self).__init__()
    #     assert len(block_channels) == len(layer_blocks), \
    #         f"# block channels {block_channels} needs to equal # layer_blocks {layer_blocks}."
    #
    #     # Initial layers
    #     self.layers = []
    #     conv1 = nn.Conv2d(in_channels, block_channels[0], kernel_size=3, stride=1, padding=1, bias=False)
    #     self.layers.append(conv1)
    #     self.layers.append(nn.BatchNorm2d(block_channels[0]))
    #     self.layers.append(nn.ReLU(inplace=True))
    #
    #     # Residual block layers
    #     for i in range(len(block_channels)):
    #         num_blocks = layer_blocks[i]
    #         in_block_channels = block_channels[i] if i == 0 else block_channels[i - 1]
    #         block_layer = self._block_layer(in_block_channels, block_channels[i], kernel_sizes[i], strides[i],
    #                                         num_blocks)
    #         self.layers.append(block_layer)
    #
    #     self.net = nn.Sequential(*self.layers)
    #
    #     # pooling layer
    #     self.avg_pool = nn.AvgPool2d(pool_size) # (block_channels[-1], 16/pool, 16/pool)
    #
    #     # linear output layer
    #     pooled_feature_map_size = (32 // np.product(strides) // pool_size) ** 2
    #     self.linear_label = nn.Linear(block_channels[-1] * pooled_feature_map_size, num_classes, bias=False)
    #
    #
    # def forward(self, x, evalMode=False):
    #     embedding = self.net(x)
    #
    #     output = self.avg_pool(embedding)
    #     output = output.reshape(output.shape[0], -1)
    #
    #     label_output = self.linear_label(output)
    #     # label_output = label_output/torch.norm(self.linear_label.weight, dim=1)
    #
    #     return output, label_output
    #
    # def _block_layer(self, in_channels, block_channels, kernel_size, stride, num_blocks):
    #     assert num_blocks >= 2, f"At least 2 blocks per layer required; {num_blocks} given."
    #
    #     block_layer = []
    #     # first block
    #     block_layer.append(
    #         BasicBlock(in_channels, block_channels, kernel_size, stride=1)
    #     )
    #     # intermediate blocks
    #     for _ in range(num_blocks - 2):
    #         block_layer.append(
    #             BasicBlock(block_channels, block_channels, kernel_size, stride=1)
    #         )
    #     # downsample if necessary by striding
    #     block_layer.append(
    #         BasicBlock(block_channels, block_channels, kernel_size, stride=stride)
    #     )
    #
    #     return nn.Sequential(*block_layer)


class BasicBlock(nn.Module):
    raise NotImplementedError
    # def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
    #     super(BasicBlock, self).__init__()
    #     padding = int(kernel_size // 2)  # preserve image size
    #     self.reshape = stride > 1 or in_channels != out_channels  # whether x needs to be reshaped before adding
    #
    #     self.straight = nn.Sequential(
    #         nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
    #         nn.BatchNorm2d(out_channels),
    #         nn.ReLU(),
    #         nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
    #         nn.BatchNorm2d(out_channels),
    #     )
    #
    #     if self.reshape:
    #         # self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
    #         self.shortcut = nn.Sequential(
    #             nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
    #             nn.BatchNorm2d(out_channels)
    #         )
    #     else:
    #         self.shortcut = nn.Identity()
    #
    #     self.relu = nn.ReLU()
    #
    # def forward(self, x):
    #     out = self.straight(x) + self.shortcut(x)  # add residual
    #     out = self.relu(out)
    #     return out


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
    def __init__(self, src_word_num, tgt_word_num, hidden_dim, max_len, nhead, num_encoder_layers=6,
     num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, activation='relu', weight_tie=True):
        super(TransformerModel, self).__init__()
        self.src_token_embedding = nn.Embedding(src_word_num, hidden_dim)
        if weight_tie == True:
            assert src_word_num == tgt_word_num
            self.tgt_token_embedding = self.src_token_embedding
        else:
            self.tgt_token_embedding = nn.Embedding(tgt_word_num, hidden_dim)
        self.position_embedding = nn.Embedding(max_len, hidden_dim)
        self.transformer = nn.Transformer(d_model=hidden_dim, nhead=nhead, num_encoder_layers=num_encoder_layers,
         num_decoder_layers=num_decoder_layers, dim_feedforward=dim_feedforward, dropout=dropout, activation=activation)
        self.linear = nn.Linear(hidden_dim, tgt_word_num)

    def embedding(self, tokens, side="src"):
        # tokens [S X N]
        if side == "src":
            token_embeddings = self.src_token_embedding(tokens)
        else:
            token_embeddings = self.tgt_token_embedding(tokens)
        max_len = tokens.size(0)
        position_idx = torch.arange(max_len).type_as(max_len) # S
        position_embeddings = self.position_embedding(position_idx) # S X E
        position_embeddings = position_embeddings.unsqueeze(1).expand(-1, tokens.size(1), -1)
        return token_embeddings + position_embeddings
        
    def forward(self, src_tokens, tgt_tokens, src_mask=None, tgt_mask=None, memory_mask=None, 
     src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        # tokens [S, N]
        src_embeddings = self.embedding(src_tokens, side="src")
        tgt_embeddings = self.embedding(src_tokens, side="tgt")
        output = self.transformer(src_embeddings, tgt_embeddings, src_mask=src_mask, tgt_mask=tgt_mask, memory_mask=memory_mask,
         src_key_padding_mask=src_key_padding_mask, tgt_key_padding_mask=tgt_key_padding_mask, memory_key_padding_mask=memory_key_padding_mask)
        # T x N x E
        return self.linear(output)

    def inference(self, src_tokens, src_key_padding_mask, start_token, max_len):
        """ greedy decoding """
        src_embeddings = self.embedding(src_tokens, side="src")
        src_encoding = self.transformer.encoder(src_embeddings, src_key_padding_mask=src_key_padding_mask)
        tgt_tokens = [start_token]
        for i in range(max_len):
            tgt_embeddings = self.embedding(torch.stack(tgt_tokens, dim=0), side="tgt")
            output = self.transformer.decoder(tgt_embeddings, src_encoding, memory_key_padding_mask=src_key_padding_mask)
            next_logits = output[-1]
            _, words = torch.max(next_logits, dim=1)
            tgt_tokens.append(words)
        return torch.stack(tgt_tokens[1:], dim=0).cpu().numpy().tolist()

    def beam_search(self, src_tokens, src_key_padding_mask, start_token, max_len, eos_token, beam_size=5):
        topk = []
        probs = []
        src_embeddings = self.embedding(src_tokens, side="src")
        src_encoding = self.transformer.encoder(src_embeddings, src_key_padding_mask=src_key_padding_mask)
        tgt_tokens = [start_token]
        for i in range(max_len):
            if i == 0:
                tgt_embeddings = self.embedding(torch.stack(tgt_tokens, dim=0), side="tgt")
                output = self.transformer.decoder(tgt_embeddings, src_encoding, memory_key_padding_mask=src_key_padding_mask)
                next_logits = output[-1]
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






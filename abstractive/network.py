# model architecture
import math

import torch
import torch.nn as nn
from torch.nn import Transformer

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:x.size(0), :]


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_attention_heads,
            num_encoder_layers, num_decoder_layers, intermediate_size,
            max_len, dropout=0.1):
        super(TransformerModel, self).__init__()

        self.token_embeddings = nn.Embedding(vocab_size, d_model)
        self.position_embeddings = PositionalEncoding(d_model, max_len)
        self.hidden_size = d_model
        self.dropout = nn.Dropout(p=dropout)

        self.transformer = Transformer(
                d_model=d_model,
                nhead=num_attention_heads,
                num_encoder_layers=num_encoder_layers,
                num_decoder_layers=num_decoder_layers,
                dim_feedforward=intermediate_size,
                dropout=dropout
        )

        self.decoder_embeddings = nn.Linear(d_model, vocab_size)
        self.decoder_embeddings.weight = self.token_embeddings.weight

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.token_embeddings.weight.data.uniform_(-initrange, initrange)
        self.decoder_embeddings.bias.data.zero_()
        self.decoder_embeddings.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, tgt, src_key_padding_mask=None, tgt_key_padding_mask=None):
        src_embeddings = self.token_embeddings(src) * math.sqrt(self.hidden_size) + self.position_embeddings(src)
        src_embeddings = self.dropout(src_embeddings)

        tgt_embeddings = self.token_embeddings(tgt) * math.sqrt(self.hidden_size) + self.position_embeddings(tgt)
        tgt_embeddings = self.dropout(tgt_embeddings)

        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(0)).to(tgt.device)
        output = self.transformer(src_embeddings, tgt_embeddings,
                tgt_mask=tgt_mask,
                src_key_padding_mask=src_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask)

        output = self.decoder_embeddings(output)
        return output

    def encode(self, src, src_key_padding_mask=None):
        src_embeddings = self.token_embeddings(src) * math.sqrt(self.hidden_size) + self.position_embeddings(src)
        src_embeddings = self.dropout(src_embeddings)

        memory = self.transformer.encoder(src_embeddings, src_key_padding_mask=src_key_padding_mask)
        return memory

    def decode(self, tgt, memory, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt_embeddings = self.token_embeddings(tgt) * math.sqrt(self.hidden_size) + self.position_embeddings(tgt)
        tgt_embeddings = self.dropout(tgt_embeddings)
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt.size(0)).to(tgt.device)

        output = self.transformer.decoder(tgt_embeddings, memory, tgt_mask=tgt_mask,
                                          tgt_key_padding_mask=tgt_key_padding_mask,
                                          memory_key_padding_mask=memory_key_padding_mask)
        output = self.decoder_embeddings(output)
        return output

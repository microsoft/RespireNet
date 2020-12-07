#!/usr/bin/python                                                       
# Author: Siddhartha Gairola (t-sigai at microsoft dot com)                 
# Transformer part has been referred from the following resource:
# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
import math
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet34, vgg16

# Transformer Model
class TransformerModel(nn.Module):

    def __init__(self, ninp, nhead, nhid, nlayers, dropout=0.5, nu_classes=10):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)

        self.ninp = ninp
        self.linear = nn.Sequential(nn.Linear(ninp, 128), nn.ReLU(True))
        self.classifier = nn.Linear(128, nu_classes) 

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src):
        src = src.float()
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        #src = self.encoder(src) * math.sqrt(self.ninp)
        src = src*math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = output.mean(axis=1)
        output = self.linear(output)
        output = self.classifier(output)

        return output


######################################################################
# ``PositionalEncoding`` module injects some information about the
# relative or absolute position of the tokens in the sequence. The
# positional encodings have the same dimension as the embeddings so that
# the two can be summed. Here, we use ``sine`` and ``cosine`` functions of
# different frequencies.
#

class PositionalEncoding(nn.Module):

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
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class model(nn.Module):
    def __init__(self, num_classes=10, drop_prob=0.5, pretrained=True):
        super(model, self).__init__()
        
        resnet = resnet34(pretrained=True)
        self.conv_feats = nn.Sequential(*list(resnet.children())[:8])
        self.avgpool = resnet.avgpool
        self.reduction = nn.Sequential(nn.Linear(128*24, 128), nn.ReLU(True), nn.BatchNorm1d(128))

        #num_ftrs = self.embeddings.fc.in_features
        #self.embeddings.fc = nn.Sequential(nn.Dropout(drop_prob), nn.Linear(num_ftrs, 128), nn.ReLU(True),
        #        nn.Dropout(drop_prob), nn.Linear(128, 128), nn.ReLU(True))
        # transformer model
        self.trfr_classifier = TransformerModel(128, 2, 200, 4, drop_prob, num_classes)

    def forward(self, x):
        
        # first extract features from the CNN
        x = self.conv_feats(x)

        # reshape the conv features and preserve the width as time
        # initial shape of x is batch x channels x height x width
        b, c, h, w = x.shape
        x = x.view(b, -1, w)
        x = x.transpose(1,2) # x: batch x width x num_features; num_features = (channels x height)

        # reducing the higher dimension num_features to 128
        x_red = []
        for t in range(x.size(1)):
            x_red.append(self.reduction(x[:,t,:]))

        x_red = torch.stack(x_red, dim=0).transpose_(0,1) # batch x time x reduced_num_features

        # pass x through the transformer
        return self.trfr_classifier(x_red)

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch
import numpy as np
import re
import os



class BaseModel(nn.Module):
    def __init__(self, pretrained_emb=False,
                 emb_dim=300, output_dim=46, vocab_size=2000):
        super().__init__()
        if pretrained_emb:
            emb_matrix = torch.load("./data/emb_matrix_ft.pt")
            self.embedding = nn.Embedding.from_pretrained(emb_matrix, 
                freeze=False)
        else: 
            self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.final_layer = nn.Linear(emb_dim, output_dim)
        
    def forward(self, x):
        seq, raw_text = x
        emb = self.embedding(seq).sum(dim=0) # sum of word embedding
        preds = self.final_layer(emb)
        return preds


class BaseModelWithLabel(nn.Module):
    """add label features"""
    def __init__(self, pretrained_emb=False, device="cuda",
                 emb_dim=300, hidden_dim=300, output_dim=46, vocab_size=2044):
        super().__init__()
        if pretrained_emb:
            emb_matrix = torch.load("./data/emb_matrix_ft.pt")
            self.embedding = nn.Embedding.from_pretrained(emb_matrix, 
                freeze=False)
        else: 
            self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.label_vocab = np.load("./data/label_vocab.npy")       
        self.labels = np.load("./data/labels.npy") 
        self.final_layer1 = nn.Linear(emb_dim, output_dim)
        #self.linear_layer = nn.Linear(emb_dim, hidden_dim)
        self.final_layer2 = nn.Linear(len(self.labels), output_dim)
        self.combination_layer = nn.Linear(2*output_dim, output_dim)
        self.device = device
        self.output_dim = output_dim
 
    def forward(self, x):
        seq, raw_text = x
        # emb dimension: batch_size * emb_dim
        emb = self.embedding(seq).sum(dim=0) # sum of word embedding
        # label_feature: batch_size * label_vocab
        label_features = self.create_label_feature(raw_text)
        preds1 = self.final_layer1(emb)
        preds2 = self.final_layer2(label_features)
        scaling_bias = torch.rand(self.output_dim, requires_grad=True).to(self.device)
        combined_preds = self.combination_layer(torch.cat((preds1, preds2), dim=1))
        return combined_preds

    def create_label_feature(self, text):
        """ a label get additional boosts if a word in text overlaps with it
            boost num is the number of words which overlaps with this label
        """
        label_feature = torch.zeros(len(text), self.output_dim)
        for i, each_line in enumerate(text):
            for word in re.split("'| ", each_line):
                if word in self.label_vocab:
                    for j, each_label in enumerate(self.labels):
                        if word in each_label:
                            label_idx = np.where(self.labels==each_label)[0][0]
                            label_feature[i, label_idx] += 1
        if self.device == "cuda":
            return label_feature.cuda()
        elif self.device == "cpu":
            return label_feature

    def create_cosim_feature(self, text):
        """ cosine similarity between text and label words"""
        pass


class GRU(nn.Module):
    def __init__(self, pretrained_emb=False, device="cuda", path='data',
                 hidden_unit=200,
                 emb_dim=300, output_dim=46, vocab_size=2000, 
                 bi=True):
        super().__init__()
        if pretrained_emb:
            emb_matrix = torch.load(os.path.join(path, "emb_matrix_ft.pt"))
            self.embedding = nn.Embedding.from_pretrained(emb_matrix, 
                freeze=False)
        else: 
            self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.gru = nn.GRU(emb_dim, hidden_unit, bidirectional=bi)
        self.hidden_unit = hidden_unit
        self.device = device
        self.bi = bi
        if self.bi:
            self.h0_bi = 2
        else:
            self.h0_bi = 1
        self.final_layer1 = nn.Linear(self.h0_bi * hidden_unit, output_dim)
        
    def forward(self, x):
        seq, raw_text = x
        x_emb, x_ngram = seq
        batch_size = len(raw_text)
        # emb shape: sequence_length, batch_size, emb_dim
        emb = self.embedding(x_emb)
        h0 = torch.rand((self.h0_bi,batch_size,self.hidden_unit), requires_grad=True)
        h0 = (h0 - 0.5)/self.hidden_unit
        if self.device == "cuda":
            h0 = h0.cuda()
        self.gru.flatten_parameters()
        output, h  = self.gru(emb, h0)
        preds = self.final_layer1(output[-1, :, :])
        return preds


class BaseModelNGram(nn.Module):
    """add label features"""
    def __init__(self, ngram=1, device="cuda", path="data",
                 hidden_dim=400, output_dim=46):
        super().__init__()
        input_dim = len(np.load(os.path.join(path, "{}grams.npy".format(ngram))))
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.device = device
        self.output_dim = output_dim
        self.batchnorm1 = nn.BatchNorm1d(hidden_dim)

    def forward(self, x):
        (x_emb, x_ngram), raw_text = x
        hidden = F.relu(self.batchnorm1(self.fc1(x_ngram)))
        preds = self.fc3(hidden)
        return preds


class MultiLayerMLP(nn.Module):
    def __init__(self, pretrained_emb=False, emb_matrix=None,
                 emb_dim=300, hidden_dim=100, 
                 output_dim=46, vocab_size=2043,
                 num_middle_layer=3, p_dropout=0.2):
        super().__init__()
        if pretrained_emb:
            self.embedding = nn.Embedding.from_pretrained(emb_matrix, freeze=False)
        else: 
            self.embedding = nn.Embedding(vocab_size, emb_dim)

        self.first_layer = nn.Linear(emb_dim, hidden_dim)
        self.middle_layers = []
        for _ in range(num_middle_layer):
            self.middle_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.middle_layers = nn.ModuleList(self.middle_layers)
        self.final_layer = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(p=p_dropout)
        
    def forward(self, seq):
        emb = self.embedding(seq).sum(dim=0) # sum of word embedding
        hidden = F.relu(self.dropout(self.first_layer(emb)))
        for i, middle_layer in enumerate(self.middle_layers):
            if i == len(self.middle_layers) - 1:
                hidden = F.relu(self.dropout(middle_layer(hidden)))
            else:
                hidden = F.relu(middle_layer(hidden))
                
        preds = self.final_layer(hidden)
        return preds


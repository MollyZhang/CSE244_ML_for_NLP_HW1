import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch
import numpy as np
import re
import os
from transformers import BertTokenizer, BertForSequenceClassification


class BertMovie(nn.Module):
    def __init__(self, output_dim=46, device="cuda"):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
        self.bert = BertForSequenceClassification.from_pretrained(
            'bert-large-uncased', num_labels=output_dim)
        self.device = device

    def forward(self, x):
        _, raw_text = x
        input_ids, attention, segment = self.prepare_data(raw_text)
        outputs = self.bert(input_ids, attention_mask=attention, token_type_ids=segment)
        return outputs[0]
    
    def prepare_data(self, raw_text):
        data = []
        for text in raw_text:
            text = "[CLS] " + text + " [SEP]"
            tokenized_text = self.tokenizer.tokenize(text)
            data.append(tokenized_text)
        longest = max([len(i) for i in data])
        indices = []
        attention_masks = []
        segment_ids = []
        for text in data:
            attention_mask = [1] * len(text) + [0] * (longest - len(text))
            text = text + ["[PAD]"] * (longest - len(text))
            segment_id = [1] * len(text)
            text_index = self.tokenizer.convert_tokens_to_ids(text)
            indices.append(text_index)
            attention_masks.append(attention_mask)
            segment_ids.append(segment_id)
        return (torch.tensor(indices).to(self.device), 
                torch.tensor(attention_masks).to(self.device), 
                torch.tensor(segment_ids).to(self.device))



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



class GRU(nn.Module):
    def __init__(self, ft=True, device="cuda", path='data',
                 hidden_unit=200, num_layer=1,
                 emb_dim=300, output_dim=46, vocab_size=2000, 
                 bi=True):
        super().__init__()
        if ft:
            print("use fasttext pretained embedding")
            emb_matrix = torch.load(os.path.join(path, "emb_matrix_ft.pt"))
            self.embedding = nn.Embedding.from_pretrained(emb_matrix, 
                freeze=False)
        else: 
            print("use random initialized embedding")
            self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.gru = nn.GRU(emb_dim, hidden_unit, num_layers=num_layer, bidirectional=bi)
        self.hidden_unit = hidden_unit
        self.device = device
        self.bi = bi
        self.num_layer=num_layer
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
        h0 = torch.rand((self.h0_bi*self.num_layer,batch_size,self.hidden_unit), 
            requires_grad=True)
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
        input_dim = 0
        for i in range(ngram):
            input_dim += len(np.load(os.path.join(path, "{}grams.npy".format(i+1))))
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
    def __init__(self, pretrained_emb=False, path="data",
                 emb_dim=300, hidden_dim=100,  
                 output_dim=46, vocab_size=2043,
                 num_middle_layer=3, p_dropout=0.2):
        super().__init__()
        if pretrained_emb:
            print("use fasttext pretained embedding")
            emb_matrix = torch.load(os.path.join(path, "emb_matrix_ft.pt"))
            self.embedding = nn.Embedding.from_pretrained(emb_matrix, freeze=False)
        else: 
            self.embedding = nn.Embedding(vocab_size, emb_dim)

        self.first_layer = nn.Linear(emb_dim, hidden_dim)
        self.middle_layers = []
        for _ in range(num_middle_layer):
            self.middle_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.middle_layers = nn.ModuleList(self.middle_layers)
        self.final_layer = nn.Linear(hidden_dim, output_dim)
        self.batchnorm1 = nn.BatchNorm1d(hidden_dim)
        self.batchnorm2 = nn.BatchNorm1d(hidden_dim)
        
    def forward(self, seq):
        (emb, ngram), extra = seq
        emb = self.embedding(emb).sum(dim=0) # sum of word embedding
        hidden = F.relu(self.batchnorm1(self.first_layer(emb)))
        for i, middle_layer in enumerate(self.middle_layers):
            hidden = F.relu(self.batchnorm2(middle_layer(hidden)))
                
        preds = self.final_layer(hidden)
        return preds



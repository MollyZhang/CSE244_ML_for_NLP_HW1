import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch
import numpy as np
import re



class BaseModel(nn.Module):
    def __init__(self, pretrained_emb=False,
                 emb_dim=300, output_dim=46, vocab_size=2044):
        super().__init__()
        if pretrained_emb:
            emb_matrix = torch.load("./data/emb_matrix_ft.pt")
            self.embedding = nn.Embedding.from_pretrained(emb_matrix, 
                freeze=False)
        else: 
            self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.final_layer = nn.Linear(emb_dim, output_dim)
        
    def forward(self, seq):
        emb = self.embedding(seq).sum(dim=0) # sum of word embedding
        preds = self.final_layer(emb)
        return preds


class BaseModelWithLabel(nn.Module):
    """add label features"""
    def __init__(self, pretrained_emb=False,
                 emb_dim=300, output_dim=46, vocab_size=2044):
        super().__init__()
        if pretrained_emb:
            emb_matrix = torch.load("./data/emb_matrix_ft.pt")
            self.embedding = nn.Embedding.from_pretrained(emb_matrix, 
                freeze=False)
        else: 
            self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.label_vocab = np.load("./data/label_vocab_restrict.npy")       
        self.final_layer1 = nn.Linear(emb_dim, output_dim)
        self.final_layer2 = nn.Linear(len(self.label_vocab), output_dim)
        self.combination_layer = nn.Linear(2*output_dim, output_dim)
 
    def forward(self, x):
        seq, raw_text = x
        # emb dimension: batch_size * emb_dim
        emb = self.embedding(seq).sum(dim=0) # sum of word embedding
        # label_feature: batch_size * label_vocab
        label_features = self.create_label_feature(raw_text)
        preds1 = self.final_layer1(emb)
        preds2 = self.final_layer2(label_features)
        combined_preds = self.combination_layer(torch.cat((preds1, preds2), dim=1))
        return combined_preds

    def create_label_feature(self, text):
        label_feature = torch.zeros(len(text), len(self.label_vocab)) 
        for i, each_line in enumerate(text):
            for word in re.split("'| ", each_line):
                if word in self.label_vocab:
                    idx = np.where(self.label_vocab==word)[0][0]
                    label_feature[i, idx] = 1
        return label_feature.cuda()


class GRU(nn.Module):
    def __init__(self, pretrained_emb=False, hidden_unit=100,
                 emb_dim=300, output_dim=46, vocab_size=200):
        super().__init__()
        if pretrained_emb:
            emb_matrix = torch.load("./data/emb_matrix_ft.pt")
            self.embedding = nn.Embedding.from_pretrained(emb_matrix, 
                freeze=False)
        else: 
            self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.gru = nn.LSTM(emb_dim, hidden_unit)
        self.final_layer = nn.Linear(hidden_unit, output_dim)
        
    def forward(self, x):
        seq, raw_text = x
        # emb shape: sequence_length, batch_size, emb_dim
        emb = self.embedding(seq)
        output, (h, c) = self.gru(emb)
        preds = self.final_layer(h[-1, :, :])
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


class OneHotMLP2LSTM(nn.Module):
    """ v1 is to flatten the one hot vector first """
    def __init__(self, vocab_size=2043, emb_dim=300, lstm_unit=300, output_dim=46):
        super().__init__()
        self.fc1 = nn.Linear(vocab, emb_dim)
        self.lstm = nn.LSTM(emb_dim, lstm_unit)
        self.fc2 = nn.Linear(lstm_unit, output_dim)
        
    def forward(self, seq):
        emb = F.relu(self.fc1(seq.view(seq.shape[1], seq.shape[0], -1)))
        hidden, _ = self.lstm(emb.view(emb.shape[1], emb.shape[0], -1))
        preds = self.fc2(hidden[-1, :, :])
        return preds


class LSTM_ONEHOT(nn.Module):
    def __init__(self, onehot_dim=2043, lstm_unit=100, output_dim=46):
        super().__init__() 
        self.lstm = nn.LSTM(onehot_dim, lstm_unit)
        self.linear = nn.Linear(lstm_unit, output_dim)
    
    def forward(self, seq):
        hidden, _ = self.lstm(seq)        
        preds = self.linear(hidden[-1, :, :])
        return preds


class BILSTM(nn.Module):
    def __init__(self, emb_dim=300, lstm_unit=100, vocab_size=2043,
                 num_layers=1, dropout=0.2, output_dim=46):
        super().__init__() 
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, lstm_unit, num_layers=num_layers, 
                            dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(lstm_unit*2, output_dim)
            
    def forward(self, seq):
        emb = self.embedding(seq)
        lstm_out, _ = self.lstm(emb)
        preds = self.fc(lstm_out[-1, :, :])
        return preds



class BILSTM_WITH_LEN(nn.Module):
    def __init__(self, emb_dim=100, vocab_size=2043, 
                 lstm_unit=100, num_layers=1, dropout=0.2, bi=True, output_dim=46):
        super().__init__() 
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, lstm_unit, num_layers=num_layers, 
                            dropout=dropout, bidirectional=bi)
        if bi:
            self.fc = nn.Linear(lstm_unit*2 + 1, output_dim)
        else: 
            self.fc = nn.Linear(lstm_unit + 1, output_dim)
            
    def forward(self, seq):
        emb = self.embedding(seq[0])        
        lstm_out, _ = self.lstm(emb)
        lstm_out_and_len = torch.cat((lstm_out[-1, :, :], 
                                      seq[1].float().reshape(-1, 1)), dim=1)
        preds = self.fc(lstm_out_and_len)
        return preds


class LSTM(nn.Module):
    def __init__(self, emb_dim=300, 
                 pretrained_emb=True, emb_matrix=None,
                 lstm_unit=100, vocab_size=2044, output_dim=46):
        super().__init__() 
        if pretrained_emb:
            assert emb_matrix is not None
            self.embedding = nn.Embedding.from_pretrained(emb_matrix, freeze=False)
        else: 
            self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, lstm_unit)
        self.linear = nn.Linear(lstm_unit, output_dim)
    
    def forward(self, seq):
        emb = self.embedding(seq)
        output, (h, c) = self.lstm(emb)
        preds = self.linear(output[-1, :, :])
        return preds




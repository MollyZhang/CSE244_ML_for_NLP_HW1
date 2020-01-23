import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable


class MultiLayerMLP(nn.Module):
    def __init__(self, emb_dim=100, output_dim=46, vocab_size=2043,
                 batch_norm=False, num_layers=1, p_dropout=0.1):
        super().__init__() 
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.fcs = []
        for _ in range(self.num_layers):
            self.fcs.append(nn.Linear(emb_dim, emb_dim))
        self.fcs = nn.ModuleList(self.fcs)
        self.final_layer = nn.Linear(emb_dim, output_dim)
        #self.bn = nn.BatchNorm1d(emb_dim)
        self.batch_norm = batch_norm
        self.dropout = nn.Dropout(p=p_dropout)
        
    def forward(self, seq):
        hidden = self.embedding(seq).sum(dim=0) # sum of word embedding
        for fc in self.fcs:
            hidden = F.relu(self.dropout(fc(hidden)))
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
    def __init__(self, emb_dim=100, lstm_unit=100, vocab_size=2043,
                 num_layers=1, dropout=0.2, bi=True, output_dim=46):
        super().__init__() 
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, lstm_unit, num_layers=num_layers, 
                            dropout=dropout, bidirectional=bi)
        if bi:
            self.fc = nn.Linear(lstm_unit*2, output_dim)
        else: 
            self.fc = nn.Linear(lstm_unit, output_dim)
            
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
    def __init__(self, emb_dim=100, lstm_unit=100, vocab_size=2043, output_dim=46):
        super().__init__() 
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, lstm_unit)
        self.linear = nn.Linear(lstm_unit, output_dim)
    
    def forward(self, seq):
        emb = self.embedding(seq)
        output, (h, c) = self.lstm(emb)
        print(output.shape)
        
        
        preds = self.linear(output[-1, :, :])
        return preds




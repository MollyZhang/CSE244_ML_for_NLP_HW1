from torchtext.data import TabularDataset, Field, RawField, BucketIterator, Iterator
import torch
import numpy as np


def prep_train_val(path, train_file, val_file, batch_size=64):
    tokenize = lambda x: x.split()
    text_field = Field(sequential=True, tokenize=tokenize, 
                       lower=True, include_lengths=True)
    labeler = lambda x: np.array([int(i) for i in list(x)])
    label_field = RawField(preprocessing=lambda x: labeler)

    datafields = [("ID", None), ("text", text_field), ("label", label_field)]
    trn, vld = TabularDataset.splits(
        path=path, train=train_file, validation=val_file,
        format='csv', skip_header=True,
        fields=datafields)
    
    text_field.build_vocab(trn, vld)
    vocab_size = len(text_field.vocab)
    train_iter, val_iter = BucketIterator.splits(
        (trn, vld),
        batch_sizes=(batch_size, batch_size),
        device=torch.device("cuda"), 
        sort_key=lambda x: len(x.text), 
        sort_within_batch=False,
        repeat=False)
    train_data = BatchWrapper(train_iter, 
        vocab_size=vocab_size, batch_size=batch_size)
    val_data = BatchWrapper(val_iter, 
        vocab_size=vocab_size, batch_size=batch_size)
    return train_data, val_data


class BatchWrapper(object):
    def __init__(self, data, onehot=False, 
                 length=False, vocab_size=-1, batch_size=-1):
        self.data = data
        self.onehot = onehot
        self.batch_size = batch_size
        self.include_len = length
        if self.onehot:
            assert(vocab_size > 0)
            self.vocab_size = vocab_size
    
    def one_hot(self, seq_batch,vocab_size):
        out = torch.zeros(seq_batch.size()+torch.Size([vocab_size]))
        dim = len(seq_batch.size())
        index = seq_batch.view(seq_batch.size()+torch.Size([1]))
        return out.scatter_(dim,index,1)
    
    def __iter__(self):
        for batch in self.data:
            x = batch.text
            if self.onehot:
                x = one_hot(x.cpu(), vocab_size).cuda()            
            if not self.include_len:
                x = x[0]
            y = np.stack(batch.label, axis=0)
            yield (x, y)
    
    def __len__(self):
        return len(self.data)
 

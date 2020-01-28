from torchtext.data import TabularDataset, Field, RawField, BucketIterator, Iterator
import torch
import numpy as np
import re


def prep_all_data(batch_size=64, use_holdout_test=False, device="cuda"):
    path = "data"
    train_file = "train_real.csv"
    val_file = "val.csv"
    holdout_test_file = "holdout_test.csv"
    test_file = "test.csv"
    train_val_file = 'train_val.csv'
    
    tokenize = lambda x: re.split("'| ", x.lower())
    text_field = Field(sequential=True, tokenize=tokenize, 
                       lower=True, include_lengths=True)
    labeler = lambda x: torch.tensor([int(i) for i in list(x)])
    #labeler = lambda x: x.split(" ")
    label_field = RawField(preprocessing=lambda x: labeler(x), is_target=True)
    #label_field = Field(sequential=False, use_vocab=False,
    #    preprocessing=lambda x: labeler(x), is_target=True)

    test_to_use = test_file
    if use_holdout_test:
        test_to_use = holdout_test_file
    trn, vld, tst = TabularDataset.splits(path=path, 
        train=train_file, validation=val_file, test=test_to_use,
        format='csv', skip_header=True,
        fields=[("ID", RawField(preprocessing=lambda x:int(x))), 
                ("text", text_field), 
                ("raw_text", RawField()),
                ("label", label_field),
                ("raw_label", RawField())])

    vocab = np.load("./data/vocab.npy")
    text_field.build_vocab([vocab])
    vocab_size = len(text_field.vocab)

    train_iter, val_iter, test_iter = BucketIterator.splits(
        (trn, vld, tst),
        batch_sizes=(batch_size, batch_size, batch_size),
        device=torch.device(device), 
        sort_key=lambda x: len(x.text), 
        sort_within_batch=False,
        repeat=False)
    
    train_data = BatchWrapper(train_iter, text_field=text_field, 
        sample_size=len(trn), vocab_size=vocab_size, batch_size=batch_size)
    val_data = BatchWrapper(val_iter, text_field=text_field, 
        sample_size=len(vld), vocab_size=vocab_size, batch_size=batch_size)
    test_data = BatchWrapper(test_iter, text_field=text_field, 
        sample_size=len(tst), vocab_size=vocab_size, batch_size=batch_size)
    
    return train_data, val_data, test_data


class BatchWrapper(object):
    def __init__(self, data, onehot=False, text_field=None, device="cuda",
                 length=False, sample_size=-1, vocab_size=-1, batch_size=-1):
        self.data = data
        self.onehot = onehot
        self.batch_size = batch_size
        self.include_len = length
        self.sample_size = sample_size
        self.text_field = text_field
        self.device = device
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
            #print(batch.label)
            y = torch.stack(batch.label, axis=0)#.to(self.device)
            ID = batch.ID
            raw_text = batch.raw_text
            raw_label = batch.raw_label
            extra = {"ID": batch.ID, "raw_text": batch.raw_text, 
                     "raw_label": batch.raw_label}
            yield (x, y, extra)
    
    def __len__(self):
        return self.sample_size


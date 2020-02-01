from torchtext.data import TabularDataset, Field, RawField, BucketIterator, Iterator
import torch
import numpy as np
import re
import os


def prep_all_data(batch_size=64, path="data", char_level=False,
                  device="cuda", ngram=1, 
                  train_file = "train_real.csv",
                  val_file = "val.csv",
                  test_file = "test.csv"):
    
    tokenize = lambda x: re.split("'| ", x.lower())
    if char_level:
        tokenize = lambda x: list(x)

    text_field = Field(sequential=True, tokenize=tokenize, 
                       lower=True, include_lengths=True)
    labeler = lambda x: torch.tensor([int(i) for i in list(x)])
    label_field = RawField(preprocessing=lambda x: labeler(x), is_target=True)

    trn, vld, tst = TabularDataset.splits(path=path, 
        train=train_file, validation=val_file, test=test_file,
        format='csv', skip_header=True,
        fields=[("ID", RawField(preprocessing=lambda x:int(x))), 
                ("text", text_field), 
                ("raw_text", RawField()),
                ("label", label_field),
                ("raw_label", RawField())])

    vocab = np.load(os.path.join(path, "vocab.npy"))
    text_field.build_vocab([vocab])
    vocab_size = len(text_field.vocab)

    train_iter, val_iter, test_iter = BucketIterator.splits(
        (trn, vld, tst),
        batch_sizes=(batch_size, batch_size, batch_size),
        device=torch.device(device), 
        sort_key=lambda x: len(x.text), 
        sort_within_batch=False,
        repeat=False)

    train_data = BaseWrapper(train_iter, text_field=text_field, path=path, 
        sample_size=len(trn), ngram=ngram, batch_size=batch_size)
    val_data = BaseWrapper(val_iter, text_field=text_field, path=path,
        sample_size=len(vld), ngram=ngram, batch_size=batch_size)
    test_data = BaseWrapper(test_iter, text_field=text_field, path=path,
        sample_size=len(tst), ngram=ngram, batch_size=batch_size)
    return train_data, val_data, test_data


class BaseWrapper(object):
    def __init__(self, data, text_field=None, path="data", device="cuda", ngram=1,
                 sample_size=None, ngram_vocab_size=None, batch_size=None):
        self.data = data
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.text_field = text_field
        self.device = device
        self.ngram = ngram
        self.vocabs = []
        for gram in range(1, ngram+1):
            self.vocabs.append(np.load(os.path.join(path, "{}grams.npy".format(gram))))
        self.ngram_vocab_size = sum([len(i) for i in self.vocabs])

    def __iter__(self):
        for batch in self.data:
            x_emb = batch.text[0]
            
            raw_text = batch.raw_text
            xs = []
            for gram in range(1, self.ngram+1):
                xs.append(self.get_gram_x(gram, raw_text))
            x_gram = torch.cat(xs, dim=1)

            y = torch.stack(batch.label, axis=0)
            ID = batch.ID
            raw_text = batch.raw_text
            raw_label = batch.raw_label
            extra = {"ID": batch.ID, "raw_text": batch.raw_text, 
                     "raw_label": batch.raw_label}
            yield (x_emb, x_gram), y, extra
    
    def __len__(self):
        return self.sample_size

    def get_gram_x(self, n, raw_text):
        vocab = self.vocabs[n-1]
        x = torch.zeros(len(raw_text), len(vocab), device=self.device)
        for i, each_line in enumerate(raw_text):
            words = re.split("'| ", each_line)
            for j in range(len(words) - (n-1)):
                ngram = "_".join(words[j:j+n])
                if ngram in vocab:
                    k = np.where(vocab==ngram)[0][0]
                    x[i, k] += 1
        return x



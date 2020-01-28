from torchtext.data import TabularDataset, Field, RawField, BucketIterator, Iterator
import torch
import numpy as np
import re


def prep_all_data(batch_size=64, use_holdout_test=False, 
                  device="cuda", x_type="embedding", ngram=None):
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
    label_field = RawField(preprocessing=lambda x: labeler(x), is_target=True)

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


    if x_type == "embedding":
        train_data = EmbeddingWrapper(train_iter, text_field=text_field, 
            sample_size=len(trn), vocab_size=vocab_size, batch_size=batch_size)
        val_data = EmbeddingWrapper(val_iter, text_field=text_field, 
            sample_size=len(vld), vocab_size=vocab_size, batch_size=batch_size)
        test_data = EmbeddingWrapper(test_iter, text_field=text_field, 
            sample_size=len(tst), vocab_size=vocab_size, batch_size=batch_size)
    elif x_type == "ngram":
        assert(ngram is not None)
        train_data = NgramWrapper(train_iter, text_field=text_field, 
            sample_size=len(trn), ngram=ngram, batch_size=batch_size)
        val_data = NgramWrapper(val_iter, text_field=text_field, 
            sample_size=len(vld), ngram=ngram, batch_size=batch_size)
        test_data = NgramWrapper(test_iter, text_field=text_field, 
            sample_size=len(tst), ngram=ngram, batch_size=batch_size)
    else:
        raise(Exception("invalid x_type"))
    return train_data, val_data, test_data


class BaseWrapper(object):
    def __init__(self, data, text_field=None, device="cuda",
                 length=False, sample_size=None, vocab_size=None, batch_size=None):
        self.data = data
        self.batch_size = batch_size
        self.include_len = length
        self.sample_size = sample_size
        self.text_field = text_field
        self.device = device
    
    def __len__(self):
        return self.sample_size


class EmbeddingWrapper(BaseWrapper):
    def __init__(self, data, text_field=None, device="cuda",
                 length=False, sample_size=None, vocab_size=None, batch_size=None):
        super().__init__(data, text_field=text_field, device=device,
                         length=length, sample_size=sample_size, 
                         vocab_size=vocab_size, batch_size=batch_size)
    
    def __iter__(self):
        for batch in self.data:
            x = batch.text
            if not self.include_len:
                x = x[0]
            y = torch.stack(batch.label, axis=0)#.to(self.device)
            ID = batch.ID
            raw_text = batch.raw_text
            raw_label = batch.raw_label
            extra = {"ID": batch.ID, "raw_text": batch.raw_text, 
                     "raw_label": batch.raw_label}
            yield (x, y, extra)
    

class NgramWrapper(BaseWrapper):
    def __init__(self, data, text_field=None, device="cuda", ngram=1,
                 length=False, sample_size=None, batch_size=None):
        super().__init__(data, text_field=text_field, device=device,
                         length=length, sample_size=sample_size, 
                         batch_size=batch_size)
        self.ngram = ngram
        self.vocabs = []
        for gram in range(1, ngram+1):
            self.vocabs.append(np.load("./data/{}grams.npy".format(gram)))
        self.vocab_size = sum([len(i) for i in self.vocabs])

    def __iter__(self):
        for batch in self.data:
            raw_text = batch.raw_text
            xs = []
            for gram in range(1, self.ngram+1):
                xs.append(self.get_gram_x(gram, raw_text))
            x = torch.cat(xs, dim=1)
            y = torch.stack(batch.label, axis=0)
            ID = batch.ID
            raw_text = batch.raw_text
            raw_label = batch.raw_label
            extra = {"ID": batch.ID, "raw_text": batch.raw_text, 
                     "raw_label": batch.raw_label}
            yield (x, y, extra)

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




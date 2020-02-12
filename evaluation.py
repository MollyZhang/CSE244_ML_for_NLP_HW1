import torch
import numpy as np


def calculate_f1(data, m):
    m.eval()
    total_f1 = 0
    num_samples = 0
    for x, y, _ in data:
        if y.shape[1] == 1:
            raise("no label in data")
        pred = m((x, _["raw_text"]))
        y_pred = (torch.sigmoid(pred) > 0.5).int().cpu().numpy()
        total_f1 += f1(y_pred, y)
        num_samples += y.shape[0]
    return(total_f1/num_samples)


def accuracy(y_pred, y_true):
    return -1

    
def f1(y_pred, y_true):
    total_f1 = 0
    y_true = y_true.cpu().numpy()
    batch_size, num_class = y_true.shape
    for sample_idx in range(batch_size):
        true_idx = np.arange(num_class)[(y_true[sample_idx] == 1).astype('bool')]
        pred_idx = np.arange(num_class)[(y_pred[sample_idx] == 1).astype('bool')]

        # make sure at least to predict one
        assert (y_true[sample_idx].sum() > 0)
        if len(pred_idx) == 0:
            pred_idx = [np.argmax(y_pred[sample_idx]).item()]

        tp = len(np.intersect1d(true_idx, pred_idx))        
        precision = tp/len(pred_idx)
        recall = tp/len(true_idx)
        if (precision + recall) == 0:
            f1_score = 0
        else:
            f1_score = 2 * precision * recall/(precision + recall)
        total_f1 += f1_score
    return total_f1


def simple_f1(y_pred=None, y_true=None):
    total_f1 = 0
    for true_label, pred_label in zip(y_true, y_pred):
        true = set(true_label.split(" "))
        pred = set(pred_label.split(" "))
        tp = len(true.intersection(pred))
        precision = tp/len(pred)
        recall = tp/len(true)
        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * precision * recall/(precision + recall)
        total_f1 += f1
    return total_f1/len(y_true)


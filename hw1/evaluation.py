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
        total_f1 += f1(pred, y)
        num_samples += y.shape[0]
    return(total_f1/num_samples)

    
def f1(y_pred, y_true):
    total_f1 = 0
    y_pred = (torch.sigmoid(y_pred) > 0.5).int().cpu().numpy()
    y_true = y_true.cpu().numpy()
    batch_size, num_class = y_true.shape
    for sample_idx in range(batch_size):
        true_idx = np.arange(num_class)[(y_true[sample_idx] == 1).astype('bool')]
        pred_idx = np.arange(num_class)[(y_pred[sample_idx] == 1).astype('bool')]

        # remove other or NO_REL if they show up as a co-occuring label
        #if len(pred_idx) >= 2:
        #    pred_idx = np.delete(pred_idx, [0, 37])

        # make sure at least to predict one
        assert (y_true[sample_idx].sum() > 0)
        if len(pred_idx) == 0:
            pred_idx = [np.argmax(y_pred[sample_idx]).item()]
        #if pred_idx[0] == 0 and len(pred_idx) == 1:
        #    pred_idx = np.array([37])

        tp = len(np.intersect1d(true_idx, pred_idx))        
        precision = tp/len(pred_idx)
        recall = tp/len(true_idx)
        if (precision + recall) == 0:
            f1_score = 0
        else:
            f1_score = 2 * precision * recall/(precision + recall)
        total_f1 += f1_score
    return total_f1

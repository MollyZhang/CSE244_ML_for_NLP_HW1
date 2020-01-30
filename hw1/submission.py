import pandas as pd
import numpy as np
import torch


def get_submission(model=None, data=None):
    model.eval()
    labels = np.load("./data/labels.npy")
    final_labels = []
    for x, y, _ in data:
        pred = (torch.sigmoid(model((x, _["raw_text"]))) > 0.5).int().cpu().numpy()
        batch_size, num_class = pred.shape
        for i in range(batch_size):
            pred_idx = np.arange(num_class)[(pred[i] == 1).astype('bool')] 
            if len(pred_idx) == 0:
                pred_idx = [np.argmax(pred[i]).item()]
            pred_label = [labels[j] for j in pred_idx]
            #if "NO_REL" in pred_label and len(pred_label) > 1:
            #   pred_label.remove("NO_REL")
            final_labels.append(
                {"ID": int(_["ID"][i]),
                 "CORE RELATIONS": " ".join(pred_label)})
    return pd.DataFrame(final_labels).sort_values(by="ID").set_index("ID")

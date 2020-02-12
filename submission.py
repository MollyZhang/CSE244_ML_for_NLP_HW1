import pandas as pd
import numpy as np
import datetime
import torch

import evaluation


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


class Ensemble(object):
    def __init__(self, models=None, 
                       val_data=None,
                       test_data=None):
        self.models = models
        self.num_models = len(models)
        self.val_data = val_data
        self.test_data = test_data
        self.df_val = pd.read_csv(
            "./data/val.csv", index_col="ID")[["text", "raw_label"]]
        self.df_test = pd.read_csv(
            "./data/test.csv", index_col="ID")[["text", "raw_label"]]


    def save_submission_file(self, name):
        today = datetime.datetime.now().strftime("%b%d")
        self.ensemble_df.to_csv("./data/submissions/{}_{}.csv".format(today, name))

    def get_ensemble_result(self, submission=False):
        if submission:
            data = self.test_data
        else:
            data = self.val_data
        dfs = []
        for i in range(self.num_models): 
            dfs.append(self.get_preds(self.models[i], data[i]))
        ensemble_df = self.calculate_ensemble_f1(dfs)

        if submission:
            df_labeled = ensemble_df.join(self.df_test)
            self.ensemble_df = ensemble_df.copy()
        else:    
            df_labeled = ensemble_df.join(self.df_val, how="inner")
            f1 = evaluation.simple_f1(y_true=df_labeled["raw_label"], 
                                      y_pred=df_labeled["CORE RELATIONS"])
            print("ensemble f1 val score:", f1)
        return df_labeled

    def get_preds(self, m, data):
        m.eval()
        preds = []
        ids = []
        for x, y, extra in data:
            pred = torch.sigmoid(m((x, extra["raw_text"]))).cpu().detach().numpy()
            preds.append(pred)
            ids.append(extra["ID"])        
        preds = np.concatenate(preds, axis=0)
        ids = np.concatenate(ids, axis=0)
        return pd.DataFrame(preds, index=ids).sort_index()
    
    def calculate_ensemble_f1(self, dfs):
        df = sum(dfs)/self.num_models
        preds = df.as_matrix()
        ids = df.index
        num_class = df.shape[1]
        labels = np.load("./data/labels.npy")
        final_labels = []
        for i in range(df.shape[0]):
            pred_idx = np.arange(num_class)[(preds[i, :] > 0.5).astype(bool)]
            if len(pred_idx) == 0:
                pred_idx = [np.argmax(preds[i]).item()]
            pred_label = [labels[j] for j in pred_idx]
            if "NO_REL" in pred_label and len(pred_label) > 1:
                pred_label.remove("NO_REL")
            final_labels.append(
                {"ID": ids[i],
                 "CORE RELATIONS": " ".join(pred_label)})
        return pd.DataFrame(final_labels).sort_values(by="ID").set_index("ID")


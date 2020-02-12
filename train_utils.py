import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import evaluation


def train(train_data, val_data, model,
          lr=1e-3, patience=10, max_epoch=100,
          print_freq=10, device="cuda"):
    no_improvement = 0
    best_val_f1 = 0
    loss_func = nn.BCEWithLogitsLoss(reduction='sum')
    if device=="cpu":
        model.cpu()
    else:
        model.cuda()
    opt = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=opt, factor=0.1, patience=patience)
    for epoch in range(max_epoch):
        if no_improvement > patience:
            break
        running_loss = 0.0
        model.train() # turn on training mode
        for x, y, _ in train_data: 
            opt.zero_grad()
            preds = model((x, _["raw_text"]))        
            loss = loss_func(preds, y.type_as(preds))
            loss.backward()
            opt.step()
            running_loss += loss.item() * y.shape[0]
        epoch_loss = running_loss / train_data.sample_size

        val_loss = calculate_loss(val_data, model, loss_func) 
        val_f1 = evaluation.calculate_f1(val_data, model)
        
        if val_f1 > best_val_f1:
            no_improvement = 0
            best_val_f1 = val_f1
            best_model = copy.deepcopy(model)
        else:
            no_improvement += 1
        scheduler.step(val_loss)
        if epoch % print_freq == 0:
            print('Epoch: {}, LR: {}, Train Loss: {:.4f}, Val Loss: {:.4f}, Val f1 {:.3f}'.format(
                epoch, opt.param_groups[0]['lr'], epoch_loss, val_loss, val_f1))
    train_f1 = evaluation.calculate_f1(train_data, best_model)
    result = {"trained_model": best_model, 
              "train f1 score": train_f1, 
              "val f1 score": best_val_f1, 
              "train loss": epoch_loss, 
              "val loss": val_loss}
    return result


def calculate_loss(val_data, model, loss_func):
    model.eval() 
    val_loss = 0.0
    for x, y, _ in val_data:
        preds = model((x, _["raw_text"]))
        loss = loss_func(preds, y.type_as(preds))
        val_loss += loss.item() * y.shape[0]
    val_loss /= val_data.sample_size
    return val_loss
    

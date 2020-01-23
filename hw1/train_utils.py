

def train(train_data, val_data, model,
          lr=1e-3, patience=10, max_epoch=100,
          print_freq=10):
    no_improvement = 0
    best_val_f1 = 0
    best_model = model
    loss_func = nn.BCEWithLogitsLoss()
    model.cuda()
    opt = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=opt, factor=0.1, patience=patience)
    for epoch in range(max_epoch):
        if no_improvement > patience:
            break
        running_loss = 0.0
        model.train() # turn on training mode
        for x, y in train_data: 
            opt.zero_grad()
            preds = model(x)        
            loss = loss_func(preds, torch.tensor(y).type_as(preds))
            loss.backward()
            opt.step()
            running_loss += loss.item() * BATCH_SIZE
        epoch_loss = running_loss / len(trn)

        model.eval() # turn on evaluation mode
        val_loss = 0.0
        for x, y in val_data:
            preds = model(x)
            loss = loss_func(preds, torch.tensor(y).type_as(preds))
            val_loss += loss.item() * BATCH_SIZE
        val_loss /= len(vld)
        
        val_f1 = evaluate(val_data, model)
        
        if val_f1 > best_val_f1:
            no_improvement = 0
            best_val_f1 = val_f1
            best_model = model
        else:
            no_improvement += 1
        scheduler.step(val_loss)
        if epoch % 10 == 0:
            print('Epoch: {}, LR: {}, Train Loss: {:.4f}, Val Loss: {:.4f}, Val F1 score {:.4f}'.format(
                epoch, opt.param_groups[0]['lr'], epoch_loss, val_loss, val_f1))
    return best_model, best_val_loss, best_val_f1




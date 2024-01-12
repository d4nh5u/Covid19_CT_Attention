import torch 
import torch.nn as nn
from torch import optim
from src.utils.plot_curve import plot_train_curve, plot_confusion_matrix
import time
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt

def initialize_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


# train
def train(train_loader, val_loader, net, Param, folder_path:str):
    train_losses = []
    train_accuracy = []
    val_losses = []
    val_accuracy = []
    lr_epoch = Param['LR']
    EPOCH = Param['EPOCH']
    best_acc = 0
    best_loss = 1000
    
    device =  torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    model = net.to(device)
    model.apply(initialize_weights)
    loss_func = nn.CrossEntropyLoss().to(device)
    
    if Param['optim'] == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=Param['LR'])
    elif Param['optim'] == 'AdamW':
        optimizer = optim.AdamW(model.parameters(), lr=Param['LR'], weight_decay=Param['L2_REG'])
        
    if Param['LRScheduler'] == 'CosineAnnealingLR':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Param['LRScheduler_Tmax'])
    elif Param['LRScheduler'] == 'ReduceLROnPlateau':
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=Param['LRScheduler_factor'], 
                                                                  patience=Param['LRScheduler_patience'], min_lr=Param['LRScheduler_min_lr'])
    bestmodel_path = folder_path + f'Best_{net.name}.pkl'
    lastmodel_path = folder_path + f'Last_{net.name}.pkl'
    
    epoch_start_time = time.time()
    for epoch in range(EPOCH):
        train_loss = 0
        train_correct = 0
        val_loss = 0
        val_correct = 0
        
        """ Training """
        model.train()
        for batch_index, batch_samples in enumerate(train_loader):
            data, label = batch_samples['img'].to(device), batch_samples['label'].to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = loss_func(output, label)
            loss.backward()
            optimizer.step()
            
            pred = output.argmax(dim=1, keepdim=True)
            train_correct += pred.eq(label.long().view_as(pred)).sum().item()
            train_loss += loss.item()

        """ Validation """
        model.eval()
        with torch.no_grad():
            for batch_index, batch_samples in enumerate(val_loader):
                data, label = batch_samples['img'].to(device), batch_samples['label'].to(device)
                output = model(data)
                val_loss += loss_func(output, label).item()
                #score = F.softmax(output, dim=1)
                pred = output.argmax(dim=1, keepdim=True)
                val_correct += pred.eq(label.long().view_as(pred)).sum().item()
          
                
        train_losses.append(train_loss / len(train_loader.dataset))
        train_accuracy.append(train_correct / len(train_loader.dataset))
        val_losses.append(val_loss / len(val_loader.dataset))
        val_accuracy.append(val_correct / len(val_loader.dataset))
        
        if (val_losses[epoch] < best_loss) and epoch>(EPOCH/4):
            best_acc = val_accuracy[epoch]
            best_loss = val_losses[epoch]
            best_epoch = epoch
            torch.save(model.eval(), bestmodel_path)
            
        if epoch == (EPOCH-1):
            torch.save(model.eval(), lastmodel_path)
            
        epoch_end_time = time.time()
        time_interval = epoch_end_time - epoch_start_time
        print(
            '\r[{:03d}/{:03d} Time: {:4.2f}s LR: {:3.8f}] Train Acc: {}/{}({:.2f}%) Loss: {:.4f} | Val Acc: {}/{}({:.2f}%) loss: {:.4f}'.format(
            epoch, EPOCH, time_interval, lr_epoch,
            train_correct, len(train_loader.dataset), 100.0*train_accuracy[epoch], train_losses[epoch],
            val_correct, len(val_loader.dataset), 100.0*val_accuracy[epoch], val_losses[epoch]),
            end=''
            )
        
        if Param['LRScheduler'] == 'CosineAnnealingLR':
            lr_scheduler.step()
        elif Param['LRScheduler'] == 'ReduceLROnPlateau' and epoch>Param['LRScheduler_start']:
            lr_scheduler.step(val_losses[epoch])
            
        lr_epoch = optimizer.param_groups[-1]['lr']


    print('\nThe best epoch: {}th  -->  Val Acc: {:.2f} | Val Loss: {:.4f}'.format(
        best_epoch, 100.0*val_accuracy[best_epoch], val_losses[best_epoch]))
    
    plot_train_curve(EPOCH, train_accuracy, val_accuracy,
                     curve_type='Accuracy', title=net.name, folder_path=folder_path, fig_size=(8,6), save_img=True)
    plot_train_curve(EPOCH, train_losses, val_losses,
                     curve_type='Loss', title=net.name, folder_path=folder_path, fig_size=(8,6), save_img=True)


# test
def test(test_loader, model, folder_path:str, plot_name:str):
    logits = []
    y_true = []
    y_pred = []
    test_loss = 0
    test_correct = 0
    
    device =  torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    loss_func = nn.CrossEntropyLoss().to(device)
    model.eval()
    with torch.no_grad():
        for batch_index, batch_samples in enumerate(test_loader):
            data, label = batch_samples['img'].to(device), batch_samples['label'].to(device)
            output = model(data)
            test_loss += loss_func(output, label).item()
            pred = output.argmax(dim=1, keepdim=True)
            test_correct += pred.eq(label.long().view_as(pred)).sum().item()
            
            # for evaluation
            logits.append(output.detach().cpu().numpy())
            y_true.append(label.detach().cpu().numpy())
            y_pred.append(pred.squeeze(1).detach().cpu().numpy())

    """ Classificatio Report and Confusion Matrix """
    print('Testset Acc: {}/{} ({:.2f}%)'.format(test_correct, len(test_loader.dataset), 100.0*test_correct/len(test_loader.dataset)))
    print('Testset Loss: {:.4f}\n'.format(test_loss/len(test_loader.dataset)))
    
    logits = np.concatenate(logits, axis=0)
    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)    
    CM = confusion_matrix(y_true, y_pred)
    class_name = ['COVID', 'Non-COVID']
    print(classification_report(y_true, y_pred, target_names=class_name, digits=4))
    print('\nConfusion Matrix: \n', CM)
    
    plt.figure(figsize=(6, 6))
    plot_confusion_matrix(  CM, classes=class_name, normalize=True, title='Confusion Matrix For Testing Data' )
    plt.tight_layout()
    CM_img_name =  folder_path + 'CM.png'
    plt.savefig(CM_img_name, bbox_inches='tight')
    
    """ ROC curve """



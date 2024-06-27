import torch
import torch.nn as nn
from logger import *
import time
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
from torch.optim.lr_scheduler import ReduceLROnPlateau

total_model=[]
total_val_f1=[]
total_test_f1=[]
test_cm=[]
total_val_loss=dict()
total_training_loss=dict()
detailed_training_f1=dict()
detailed_val_f1=dict()
model_state=dict()

def f1table():
    table=pd.DataFrame({'Model':total_model,
                    'Val f1':total_val_f1,
                    'Test f1':total_test_f1,})
    tablelog(table)

def cmtable():
    for i, cm in enumerate(test_cm):
        keylog('Confusion Matrix', f'{total_model[i]}')
        keylog(np.array2string(cm))

def getdetails():
    keylog('training_loss_details= ', total_training_loss)
    keylog('val_loss_details= ', total_val_loss)
    keylog('training_f1_details= ', detailed_training_f1)
    keylog('val_f1_details= ', detailed_val_f1)

def trainer(model, num_ep, learning_rate, train_loader, val_loader, test_loader):
    start_time=time.time()
    crit=nn.CrossEntropyLoss()
    optimiser=torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler=ReduceLROnPlateau(optimiser, 'min', patience=0, factor=0.2)
    training_loss_list=[]
    val_loss_list=[]
    training_f1_list=[]
    val_f1_list=[]

    for epoch in range(num_ep):
        target=[]
        output=[]
        training_loss=0.0

        for i, (img, label) in enumerate(train_loader):
            img=img.view(img.size(0), 299, -1)
            out=model(img)
            loss=crit(out, label)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            _, pred=torch.max(out.data, 1)
            target.extend(label.numpy())
            output.extend(pred.numpy())
            training_loss+=loss.item()

            if i % 46 == 45:
                training_f1=round(f1_score(target, output, average='weighted'), 4)
                training_f1_list.append(training_f1)
                training_loss/=46
                training_loss=round(training_loss, 5)
                training_loss_list.append(training_loss)
                print(model.__class__.__name__, f'training_f1 of model: {training_f1}')
                print(model.__class__.__name__, f'epoch: {epoch+1}, step: {i+1}, loss: {training_loss}')
                
                with torch.no_grad():
                    output=[]
                    target=[]
                    val_running_loss=0.0
                    for img, label in val_loader:
                        img=img.view(img.size(0), 299, -1)
                        out=model(img)
                        _, pred=torch.max(out, 1)
                        target.extend(label.numpy())
                        output.extend(pred.numpy())
                        loss=crit(out, label)
                        val_running_loss+=loss.item()

                    val_running_loss/=len(val_loader)
                    val_running_loss=round(val_running_loss, 5)
                    print('Validation loss =', val_running_loss)
                    val_loss_list.append(val_running_loss)
                    val_f1=round(f1_score(target, output, average='weighted'), 4)
                    val_f1_list.append(val_f1)
                    print(model.__class__.__name__, f'val_f1 of model: {val_f1}')
                    training_loss=0.0

        scheduler.step(val_running_loss)
        curr_lr=optimiser.param_groups[0]['lr']
        print(f'current LR= {curr_lr}')

        with torch.no_grad():
            output=[]
            target=[]
            for img, label in val_loader:
                img=img.view(img.size(0), 299, -1)
                out=model(img)
                _, pred=torch.max(out, 1)
                target.extend(label.numpy())
                output.extend(pred.numpy())

    total_training_loss[model.__class__.__name__]=training_loss_list
    total_val_loss[model.__class__.__name__]=val_loss_list
    detailed_training_f1[model.__class__.__name__]=training_f1_list
    detailed_val_f1[model.__class__.__name__]=val_f1_list

    final_val_f1=round(f1_score(target, output, average='weighted'), 4)
    total_val_f1.append(final_val_f1)
    print(model.__class__.__name__, f'val_f1 of model: {final_val_f1}')

    total_model.append(model.__class__.__name__)
    time_taken=round(time.time() - start_time, 2)
    print('time taken', "--- %s seconds ---" % (time_taken))
    test_f1(model, test_loader)

def test_f1(model, test_loader):
    with torch.no_grad():
        output=[]
        target=[]
        for img, label in test_loader:
            img=img.view(img.size(0), 299, -1)
            out=model(img)
            _, pred=torch.max(out, 1)
            target.extend(label.numpy())
            output.extend(pred.numpy())

        test_f1=round(f1_score(target, output, average='weighted'), 4)
        total_test_f1.append(test_f1)
        print(model.__class__.__name__, f'test_f1 of model: {test_f1}')

        tcm=confusion_matrix(target, output)
        test_cm.append(tcm)

def savemodel(path):
    torch.save(model_state, path)
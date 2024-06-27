import os
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
import torch.nn as nn
from rnn_trainer import *
from Rnn import *
from logger import *

def helper(batch_size, num_ep, learning_rate, aug_bright, aug_contrast, aug_saturation, aug_hue):
    path=os.getcwd()
    keylog('\nParameters', f"Batch size= {batch_size}, num_ep= {num_ep}, learning_rate= {learning_rate}")
    keylog('Augments', f"aug_bright= {aug_bright}, aug_contrast= {aug_contrast}, aug_saturation= {aug_saturation}, aug_hue= {aug_hue}")

    #augmentations
    aug_transform=transforms.Compose([transforms.Resize((299, 299)),
                                    transforms.ToTensor(),
                                    transforms.ColorJitter(brightness=aug_bright, contrast=aug_contrast, saturation=aug_saturation, hue=aug_hue),])
    
    #transforms to tensor and normalize
    transform=transforms.Compose([transforms.Resize((299, 299)),
                                transforms.ToTensor(),])

    #load data
    aug_train_ds= ImageFolder(root=path+'/train', transform=aug_transform)
    norm_train_ds= ImageFolder(root=path+'/train', transform=transform)
    val_ds= ImageFolder(root=path+'/val', transform=transform)
    test_ds= ImageFolder(root=path+'/test', transform=transform)
    train_ds= ConcatDataset([norm_train_ds, aug_train_ds])

    train_loader= DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True, num_workers=6, drop_last=True)
    val_loader= DataLoader(dataset=val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader= DataLoader(dataset=test_ds, batch_size=batch_size, shuffle=False, num_workers=4)

    print('running Rnn')
    trainer(model=Rnn(), num_ep=num_ep, learning_rate=learning_rate, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader)
    print('running biRnn')
    trainer(model=biRnn(), num_ep=num_ep, learning_rate=learning_rate, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader)
    print('running Lstm')
    trainer(model=Lstm(), num_ep=num_ep, learning_rate=learning_rate, train_loader=train_loader, val_loader=val_loader, test_loader=test_loader)

    f1table()#get F1 score in keylogs file
    cmtable()#get confusion matric in keylogs file
    getdetails()#get loss and f1 details dict in keylogs file
    savemodel(path+'/Rnn_model/Rnnstate.pt')#save model state


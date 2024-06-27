import torch
import os
import matplotlib.pyplot as plt
from torchinfo import summary
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from Rnn import *


path=(os.getcwd())
test_ds= ImageFolder(root=path+'/test', transform=ToTensor())

#Paste logged details here
training_loss_details=  {'Rnn': [1.02869, 0.85903, 0.82802, 0.75051, 0.75992, 0.73391, 0.70335, 0.72664, 0.71454, 0.67802, 0.66956, 0.67693, 0.66417, 0.65279], 'biRnn': [1.07194, 0.90851, 0.82078, 0.77956, 0.75794, 0.7307, 0.71525, 0.76344, 0.69382, 0.71167, 0.71648, 0.68546, 0.6625, 0.69708], 'Lstm': [1.0398, 0.98435, 0.90372, 0.84661, 0.79583, 0.78658, 0.7535, 0.73816, 0.75117, 0.74217, 0.70659, 0.72059, 0.67869, 0.71375]}
val_loss_details=  {'Rnn': [0.92552, 0.92039, 0.81382, 0.75519, 0.76007, 0.74008, 0.71787, 0.7685, 0.7014, 0.68766, 0.68128, 0.68703, 0.67619, 0.69839], 'biRnn': [0.98901, 0.90998, 0.81999, 0.81461, 0.76364, 0.78303, 0.75278, 0.72881, 0.75301, 0.8573, 0.70751, 0.70898, 0.69547, 0.69965], 'Lstm': [1.02577, 0.99623, 0.89357, 0.8371, 0.82991, 0.78723, 0.76169, 0.74932, 0.74594, 0.73033, 0.75033, 0.72298, 0.69925, 0.7163]}
training_f1_details=  {'Rnn': [0.46, 0.6353, 0.6546, 0.6703, 0.6737, 0.6713, 0.707, 0.6741, 0.6815, 0.692, 0.7092, 0.6953, 0.7047, 0.7165], 'biRnn': [0.4899, 0.6187, 0.6534, 0.6506, 0.6675, 0.6899, 0.6982, 0.6531, 0.699, 0.6795, 0.685, 0.7041, 0.7185, 0.6809], 'Lstm': [0.4755, 0.4908, 0.5809, 0.6244, 0.6528, 0.6587, 0.6847, 0.6765, 0.6764, 0.6761, 0.6918, 0.6934, 0.6982, 0.6972]}
val_f1_details=  {'Rnn': [0.6023, 0.493, 0.6194, 0.6541, 0.6354, 0.6562, 0.6677, 0.606, 0.6527, 0.6795, 0.69, 0.6674, 0.6855, 0.6566], 'biRnn': [0.5699, 0.5481, 0.6161, 0.6144, 0.6743, 0.6197, 0.6287, 0.6648, 0.661, 0.5666, 0.6758, 0.6724, 0.6671, 0.6791], 'Lstm': [0.4376, 0.5348, 0.5692, 0.617, 0.6049, 0.6444, 0.6373, 0.6549, 0.6568, 0.655, 0.6545, 0.6777, 0.6745, 0.6698]}

def seeloss(graph1, graph2, model):
    num_epochs = len(graph1)
    steps = range(1, num_epochs + 1)
    scaled_steps = [step * 46 for step in steps]
    plt.figure(figsize=(12,8))
    plt.plot(scaled_steps, graph1, label='Training Loss')
    plt.plot(scaled_steps, graph2, linestyle='--', label='Validation Loss')
    plt.xlabel('steps')
    plt.ylabel('Loss')
    plt.title(f'{model} Training and Validation Loss Curves')
    plt.legend()
    plt.grid(True)
    plt.show()

def seef1(graph1, graph2, model):
    num_epochs = len(graph1)
    steps = range(1, num_epochs + 1)
    scaled_steps = [step * 46 for step in steps]
    plt.figure(figsize=(12,8))
    plt.plot(scaled_steps, graph1, label='Training f1')
    plt.plot(scaled_steps, graph2, linestyle='--', label='Validation f1')
    plt.xlabel('steps')
    plt.ylabel('f1')
    plt.title(f'{model} Training and Validation f1 Curves')
    plt.legend()
    plt.grid(True)
    plt.show()

for model in training_f1_details.keys():
    seeloss(training_loss_details[model], val_loss_details[model], model)
    seef1(training_f1_details[model], val_f1_details[model], model)

path=(os.getcwd()+'/Cnn_model/Cnnstate.pt')
model1=Rnn()
model2=biRnn()
model3=Lstm()
checkpoint=torch.load(path)
model1.load_state_dict(checkpoint['Rnn'])
model2.load_state_dict(checkpoint['biRnn'])
model3.load_state_dict(checkpoint['Lstm'])

print(f'Model 1 state dict size: {sum(p.numel() for p in model.parameters()) * 4 / 1e6} MB')
print("Model's state_dict:")
for param_tensor in model1.state_dict():
    print(param_tensor, "\t", model1.state_dict()[param_tensor].size())
summary(model, (3,299,299))
for param_tensor in model1.state_dict():
    print(param_tensor, "\t", model1.state_dict()[param_tensor].size())
print('model 1 summary')
summary(model1, (3,299,299))

print(f'Model 2 state dict size: {sum(p.numel() for p in model.parameters()) * 4 / 1e6} MB')
print("Model's state_dict:")
for param_tensor in model2.state_dict():
    print(param_tensor, "\t", model2.state_dict()[param_tensor].size())
summary(model, (3,299,299))
for param_tensor in model2.state_dict():
    print(param_tensor, "\t", model2.state_dict()[param_tensor].size())
print('model 2 summary')
summary(model2, (3,299,299))

print(f'Model 3 state dict size: {sum(p.numel() for p in model.parameters()) * 4 / 1e6} MB')
print("Model's state_dict:")
for param_tensor in model3.state_dict():
    print(param_tensor, "\t", model3.state_dict()[param_tensor].size())
summary(model, (3,299,299))
for param_tensor in model3.state_dict():
    print(param_tensor, "\t", model3.state_dict()[param_tensor].size())
print('model 3 summary')
summary(model3, (3,299,299))
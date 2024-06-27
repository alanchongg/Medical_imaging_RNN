import torch.nn as nn
import torch

class Rnn(nn.Module):
    def __init__(self):
        super(Rnn, self).__init__()
        self.input_size = 3 * 299
        self.hidden_size = 128
        self.num_layers = 1
        self.num_classes = 3
        self.rnn = nn.RNN(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.rnn(x, h0)
        out = out[:, -1, :]
        out = self.fc(out)
        return out
    
class biRnn(nn.Module):
    def __init__(self):
        super(biRnn, self).__init__()
        self.input_size = 3 * 299
        self.hidden_size = 128
        self.num_layers = 1
        self.num_classes = 3
        self.rnn = nn.RNN(self.input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(self.hidden_size*2, self.num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size)
        out, _ = self.rnn(x, h0)
        out = out[:, -1, :]
        out = self.fc(out)
        return out
    
class Lstm(nn.Module):
    def __init__(self):
        super(Lstm, self).__init__()
        self.input_size = 3 * 299
        self.hidden_size = 128
        self.num_layers = 1
        self.num_classes = 3
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        return out
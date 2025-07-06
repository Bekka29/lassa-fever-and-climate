# %%
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable



class EpidemiologyModelAllVariable(nn.Module):
    def __init__(self, feature_dim=8, hidden_size=128, num_lstm_layers = 2, bidirection=False, device='cpu'):
        super().__init__()
        self.device = device
        self.bidirection = bidirection
        self.hidden_size = hidden_size
        self.num_lstm_layers = num_lstm_layers
        self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=self.hidden_size, dropout=0.2, bidirectional=self.bidirection, num_layers=self.num_lstm_layers, batch_first=True, device=self.device)
        if bidirection:
            self.regressor = nn.Linear(2*self.hidden_size, feature_dim, device= self.device)
        else:
            self.regressor = nn.Linear(self.hidden_size, feature_dim, device= self.device)
        self.relu = m = F.relu
    def forward(self, x):
        self.lstm.flatten_parameters()
        if self.bidirection:
            h_0 = Variable(torch.zeros(2*self.num_lstm_layers, x.size(0), self.hidden_size, device= self.device)) #hidden state
            c_0 = Variable(torch.zeros(2*self.num_lstm_layers, x.size(0), self.hidden_size, device = self.device)) #internal state
        else:
            h_0 = Variable(torch.zeros(self.num_lstm_layers, x.size(0), self.hidden_size, device = self.device)) #hidden state
            c_0 = Variable(torch.zeros(self.num_lstm_layers, x.size(0), self.hidden_size, device= self.device)) #internal state

        out, (h_n, c_o) = self.lstm(x, (h_0, c_0))  # output, (hidden, cell_state)
        out = self.relu(out)
        x = self.regressor(out)

        return x[:, :, :6], torch.exp(x[:,:, -1])

class EpidemiologyOneOutput(nn.Module):
    def __init__(self, feature_dim=6, output_dim = 1, hidden_size=128, num_lstm_layers = 2, bidirection=False, device='cpu'):
        super().__init__()
        self.device = device
        self.bidirection = bidirection
        self.hidden_size = hidden_size
        self.num_lstm_layers = num_lstm_layers
        self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=self.hidden_size, dropout=0.2, bidirectional=self.bidirection, num_layers=self.num_lstm_layers, batch_first=True, device=self.device)
        if bidirection:
            self.regressor = nn.Linear(2*self.hidden_size, output_dim, device= self.device)
        else:
            self.regressor = nn.Linear(self.hidden_size, output_dim, device= self.device)
        self.relu = m = F.relu
    def forward(self, x):
        self.lstm.flatten_parameters()
        if self.bidirection:
            h_0 = Variable(torch.zeros(2*self.num_lstm_layers, x.size(0), self.hidden_size, device= self.device)) #hidden state
            c_0 = Variable(torch.zeros(2*self.num_lstm_layers, x.size(0), self.hidden_size, device = self.device)) #internal state
        else:
            h_0 = Variable(torch.zeros(self.num_lstm_layers, x.size(0), self.hidden_size, device = self.device)) #hidden state
            c_0 = Variable(torch.zeros(self.num_lstm_layers, x.size(0), self.hidden_size, device= self.device)) #internal state

        out, (h_n, c_o) = self.lstm(x, (h_0, c_0))  # output, (hidden, cell_state)
        out = self.relu(out)
        out = self.regressor(out)
        out = torch.exp(out)       
        return out

class MARModel(nn.Module):
    def __init__(self, d, p):
        super().__init__()
        self.d = d
        self.p = p
        self.linear = nn.Linear(d * p, d, bias=False)

    def forward(self, x):
        # x: [batch_size, p, d]
        x = x.view(x.size(0), -1)  # Flatten to [batch_size, d * p]
        return self.linear(x)

class MARModelOneOutput(nn.Module):
    def __init__(self, d, p):
        super().__init__()
        self.d = d
        self.p = p
        self.linear = nn.Linear(d * p, 1, bias=False)

    def forward(self, x):
        # x: [batch_size, p, d]
        x = x.view(x.size(0), -1)  # Flatten to [batch_size, d * p]
        return torch.exp(self.linear(x))

# -------------------------
# MAR Model as Linear Regression
# -------------------------
class MARModel(nn.Module):
    def __init__(self, d, p):
        super().__init__()
        self.d = d
        self.p = p
        self.linear = nn.Linear(d * p, d, bias=False)

    def forward(self, x):
        # x: [batch_size, p, d]
        x = x.view(x.size(0), -1)  # Flatten to [batch_size, d * p]
        x= self.linear(x)
        return x[:, :6], torch.exp(x[:, -1])


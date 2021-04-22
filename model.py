import torch
import torch.nn as nn
import torch.nn.functional as F


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device):
        super(RNN, self).__init__()
        self.device = device

        self.hidden_size = hidden_size

        self.linear1 = nn.Linear(input_size + hidden_size, input_size + hidden_size)
        self.linear2 = nn.Linear(input_size + hidden_size, input_size + hidden_size)
        self.linear_output = nn.Linear(input_size + hidden_size, output_size)
        self.linear_hidden = nn.Linear(input_size + hidden_size, hidden_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, hidden_state):
        x = torch.cat((x, hidden_state), 0)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        output = self.linear_output(x)
        hidden = self.linear_hidden(x)
        return output, hidden

    def initiate_hidden_state(self):
        return torch.zeros(self.hidden_size, device=self.device)

from model import RNN
from databes import OurDataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
import os

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)

curr_path = os.path.abspath(os.getcwd())

batch_size = 128
dataloader = DataLoader(OurDataset(os.path.join(curr_path, 'data'), device), batch_size=batch_size, shuffle=True,
                        num_workers=0, drop_last=True)
model = RNN(1, 7, 5, device)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

for epoch in range(30):
    running_loss = 0.0
    for i, data in enumerate(dataloader):
        batch_inputs, batch_labels = data
        transposed_inputs = torch.transpose(batch_inputs, 0, 1)
        transposed_labels = torch.transpose(batch_labels, 0, 1)
        optimizer.zero_grad()
        hidden = model.initiate_hidden_state(batch_size)
        outputs = torch.zeros((1500, batch_size, 5), device=device)
        for j, x in enumerate(transposed_inputs):
            output, hidden = model(x.view(batch_size, 1), hidden)
            outputs[j] = output

        # forward + backward + optimize
        loss = criterion(outputs[0], transposed_labels[0])
        for output, labels in zip(outputs[1:], transposed_labels[1:]):
            loss += criterion(output, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / (i + 1)))

print('Finished Training')

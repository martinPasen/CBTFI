from model import RNN
from databes import OurDataset
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
import torch
import os

curr_path = os.path.abspath(os.getcwd())

dataloader = DataLoader(OurDataset(os.path.join(curr_path, 'data')), batch_size=1, shuffle=True, num_workers=0)
model = RNN(1, 7, 5)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

for epoch in range(30):
    running_loss = 0.0
    for i, data in enumerate(dataloader):
        batch_inputs, batch_labels = data
        for inputs, labels in zip(batch_inputs, batch_labels):
            optimizer.zero_grad()
            hidden = model.initiate_hidden_state()
            outputs = torch.zeros((len(inputs), 5))
            for j, x in enumerate(inputs):
                x = x.flatten()
                output, hidden = model(x, hidden)
                outputs[j] = output

            # forward + backward + optimize
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / (i + 1)))

print('Finished Training')

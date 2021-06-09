from model import RNN
from databes import OurDataset, INPUT_SIZE
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import torch
import os
from transforms import To256


def run(model, dataloader, criterion, optimizer=None):
    running_loss = 0.0
    if optimizer is not None:
        model.train()
    else:
        model.eval()
    i = 0
    total = 0
    correct = 0
    for i, data in enumerate(dataloader):
        batch_inputs, batch_labels = data
        transposed_inputs = torch.transpose(batch_inputs, 0, 1)
        transposed_labels = torch.transpose(batch_labels, 0, 1)
        if optimizer is not None:
            optimizer.zero_grad()
        hidden = model.initiate_hidden_state(batch_size)
        outputs = torch.zeros((INPUT_SIZE, batch_size, 5), device=device)
        for j, x in enumerate(transposed_inputs):
            output, hidden = model(x.view(batch_size, 256), hidden)
            outputs[j] = output

        # forward + backward + optimize
        loss = criterion(outputs[0], transposed_labels[0])
        for output, labels in zip(outputs[1:], transposed_labels[1:]):
            loss += criterion(output, labels)
            total += labels.size(0)
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == labels).sum().item()

        if optimizer is not None:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
            optimizer.step()

        # print statistics
        running_loss += loss.item()
        print('[%2d, %5d] loss: %.3f, accuraccy: %.2f' % (
            epoch + 1, i + 1, running_loss / (i + 1), correct / total))
    return running_loss / (i + 1), correct / total


if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)

curr_path = os.path.abspath(os.getcwd())

batch_size = 128
dataset = OurDataset(os.path.join(curr_path, 'data'), device, To256())
training_set = torch.utils.data.Subset(dataset, (range(0, (len(dataset) * 4) // 5)))
validation_set = torch.utils.data.Subset(dataset, (range((len(dataset) * 4) // 5, len(dataset))))

training_dataloader = DataLoader(training_set, batch_size=batch_size, shuffle=True,
                                 num_workers=0, drop_last=True)

validation_dataloader = DataLoader(validation_set, batch_size=batch_size, shuffle=True,
                                   num_workers=0, drop_last=True)

model = RNN(256, 7, 5, device)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

running_losses = []
valid_losses = []
acuracy = []

for epoch in range(5):
    print('Training')
    run(model, training_dataloader, criterion, optimizer)

    print('Testing with training set')
    loss, _ = run(model, training_dataloader, criterion)
    running_losses.append(loss)

    print('Testing with validation set')
    loss, acc = run(model, validation_dataloader, criterion)
    valid_losses.append(loss)
    acuracy.append(acc)
    torch.save(model, 'model.pt')
    torch.save(criterion, 'criterion.pt')
    torch.save(optimizer, 'optimizer.pt')
plt.figure(1)
plt.plot(running_losses, label='Running loss')
plt.plot(valid_losses, label='Validation loss')
plt.legend()
plt.figure(2)
plt.plot(acuracy, label='Accuracy')
plt.legend()
plt.show()
print('Finished Training')

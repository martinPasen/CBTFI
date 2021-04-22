from torch.utils.data import Dataset
import random
import torch
import os


class OurDataset(Dataset):
    def __init__(self, dir_path, device):
        self.dir_path = dir_path
        _, _, file_names = next(os.walk(dir_path))
        data = []
        labels = {}
        counter = 0
        for i, file_name in zip(range(len(file_names)), file_names):
            label, _ = str.split(file_name, '.')
            bytes = list(open(os.path.join(dir_path, file_name), 'rb').read())
            max_size = len(bytes) // 300
            data.append([bytes, max_size, 0])
            counter += max_size
            labels[label] = i
        self.input_data = []
        self.targets = []
        while len(labels) != 0:
            input_data = []
            target = []
            for i in range(5):
                if (len(labels) == 0):
                    break
                rand_label = random.choice(list(labels.keys()))
                rand = labels[rand_label]
                bytes, _, cur = data[rand]
                input_data += bytes[cur * 300:(cur + 1) * 300]
                target += [rand] * 300
                data[rand][2] += 1
                counter -= 1
                if data[rand][1] - 1 == data[rand][2]:
                    labels.pop(rand_label)
            if len(input_data) == 1500:
                self.input_data.append(input_data)
                self.targets.append(target)
        self.input_data = torch.tensor(self.input_data, device=device)
        self.targets = torch.tensor(self.targets, device=device)

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, index):
        return self.input_data[index], self.targets[index]

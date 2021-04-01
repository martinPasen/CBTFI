from torch.utils.data import Dataset
import torch


class OurDataset(Dataset):
    def __init__(self):
        self.input_data = torch.randint(256, (10, 30))
        self.targets = torch.randint(5, (10, 30))

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, index):
        return self.input_data[index], self.targets[index]

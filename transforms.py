import torch


class To256(object):
    def __call__(self, input, device):
        tn = torch.zeros((1500, 256), device=device, dtype=torch.bool)
        for i, j in enumerate(input):
            tn[i][j] = True
        return tn

    def __repr__(self):
        return self.__class__.__name__ + '()'

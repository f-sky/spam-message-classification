import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from easydict import EasyDict
from progressbar import progressbar
from torch.autograd import Variable


class AverageMeter(EasyDict):
    """Computes and stores the average and current value"""

    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class History:
    def __init__(self, metrics):
        metrics = list(set(metrics))
        self.keys = metrics
        self.records = {}
        for k in metrics:
            self.records[k] = {'train': [], 'dev': []}

    def load_dict(self, other):
        for k in self.keys:
            self.records[k] = other.metrics[k]

    def plot(self):
        epochs = len(self.records[self.keys[0]]['train'])
        for k in self.keys:
            plt.plot(range(1, 1 + epochs), self.records[k]['train'], label='train_{}'.format(k))
            plt.plot(range(1, 1 + epochs), self.records[k]['dev'], label='dev_{}'.format(k))
        plt.legend()
        plt.show()


def load_model(model, optim, model_dir, epoch=-1, history: History = None):
    if not os.path.exists(model_dir):
        return 0

    pths = [int(pth.split('.')[0]) for pth in os.listdir(model_dir)]
    if len(pths) == 0:
        return 0
    if epoch == -1:
        pth = max(pths)
    else:
        pth = epoch
    pretrained_model = torch.load(os.path.join(model_dir, '{}.pth'.format(pth)))
    model.load_state_dict(pretrained_model['net'])
    optim.load_state_dict(pretrained_model['optim'])
    if history is not None:
        history.load_dict(pretrained_model['history'])
    return pretrained_model['epoch'] + 1


def save_model(net, optim, epoch, model_dir, history=None):
    os.system('mkdir -p {}'.format(model_dir))
    obj = {
        'net': net.state_dict(),
        'optim': optim.state_dict(),
        'epoch': epoch
    }
    if history is not None:
        obj['history'] = history
    torch.save(obj, os.path.join(model_dir, '{}.pth'.format(epoch)))

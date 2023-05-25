from sklearn.model_selection import KFold
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import torch
import numpy as np
from omegaconf import OmegaConf
from utils import instantiation, get_logger
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import divide_train_val
import json
import argparse

def evaluate(model, val_dataloader, logger, device):
    with torch.no_grad():
        for i, (imgs, labels) in enumerate(tqdm(val_dataloader)):
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            if i == 0:
                all_outputs = outputs
                all_labels = labels
            else:   
                all_outputs = torch.cat((all_outputs, outputs), dim = 0)
                all_labels = torch.cat((all_labels, labels), dim = 0)
        all_outputs = all_outputs.cpu().numpy()
        all_labels = all_labels.cpu().numpy()
        all_outputs = np.argmax(all_outputs, axis = 1)
        acc = np.sum(all_outputs == all_labels) / len(all_labels)
        return acc

    
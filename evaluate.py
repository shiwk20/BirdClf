import torch
import numpy as np
from omegaconf import OmegaConf
from utils import instantiation, get_logger
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
import argparse

def evaluate(model, dataloader, logger, device):
    with torch.no_grad():
        all_labels = []
        all_outputs = []
        for batch in dataloader:
            imgs, labels = batch
            labels = labels.to(device)
            outputs = model(imgs.to(device))
            all_labels.append(labels)
            all_outputs.append(outputs)
        all_labels = torch.cat(all_labels, dim = 0)
        all_outputs = torch.cat(all_outputs, dim = 0)
        all_labels = all_labels.cpu().numpy()
        all_outputs = all_outputs.cpu().numpy()
        all_outputs = np.argmax(all_outputs, axis = 1)
        acc = np.mean(all_labels == all_outputs)
        return acc

    
    
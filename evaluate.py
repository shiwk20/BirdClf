import torch
import numpy as np
from omegaconf import OmegaConf
from utils import instantiation, get_logger
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from sklearn.metrics import classification_report
import argparse

def evaluate(model, dataloader, logger, device, type):
    model.eval()
    with torch.no_grad():
        # 进行测试，获取预测结果及标签
        labels = []
        preds = []
        for batch in tqdm(dataloader, leave=False):
            input, label = batch
            label = label.to(device)
            outputs = model(input.to(device))
            labels.append(label)
            preds.append(outputs)
        labels = torch.cat(labels, dim = 0)
        preds = torch.cat(preds, dim = 0)
        labels = labels.cpu().numpy()
        preds = preds.cpu().numpy()
        preds = np.argmax(preds, axis = 1)
        
        # 进行评估，计算Accuray, Precision, Recall, F1_score等指标
        Accuray = np.mean(labels == preds)
        logger.info('{} Accuray: {:.6f}'.format(type, Accuray))
        
        output_dict = classification_report(labels, preds, output_dict=True, digits=4)
        logger.info(f'{type} macro avg: {output_dict["macro avg"]}')
        logger.info(f'{type} weighted avg: {output_dict["weighted avg"]}')
        
        return Accuray


    
    
import os
import numpy as np
import time
import logging
import importlib
import torch
import random
import json


def get_folder_num(path):
    count1 = 0
    count2 = 0
    name_list = os.listdir(path)
    count1 = len(name_list)
    for name in name_list:
        img_list = os.listdir(os.path.join(path, name))
        for img in img_list:
            count2 += 1
    return count1, count2   

# compute the area of bounding box
def area(mtcnn):
    return -np.prod(mtcnn['box'][-2:])

# b, c
def L2_dist(vec1, vec2):
    return torch.sqrt(torch.sum(torch.square(vec1 - vec2), dim = 1))

def get_logger(type, log_path = 'log/logs'):
    logger = logging.getLogger()
    logfile = os.path.join(log_path, '{}_{}.log'.format(type, time.strftime('%m-%d-%H-%M-%S')))
    logging.basicConfig(level = logging.INFO, format = \
        '%(asctime)s - %(levelname)s %(filename)s(%(lineno)d): %(message)s', filename = logfile)
    logging.root.addHandler(logging.StreamHandler())
    return logger

def instantiation(config, args = {}):
    assert 'dest' in config, 'No dest key in config'
    dest, name = config["dest"].rsplit(".", 1)
    module = importlib.import_module(dest)
    return getattr(module, name)(**config.get("paras", dict()), **args)
    
def get_train_indexes(align_type):
    tmp = {}
    if align_type == 'landmark':
        train_img_path = 'data/train/landmark/align112x112'
    else:
        train_img_path = 'data/train/mtcnn/align112x96'
    for root, dirs, files in os.walk(train_img_path):
        if root == train_img_path:
            continue
        tmp[root] = files
    tmp = sorted(tmp.items(), key = lambda x: x[0].lower())
    train_indexes = {}
    for train_data in tmp:
        train_indexes[train_data[0]] = train_data[1]
    return train_indexes

def divide_train_val(seed, align_type, train_ratio = 0.85, device = 0):
    all_indexes = get_train_indexes(align_type)
    
    random.seed(seed)
    train_indexes = random.sample(list(all_indexes.keys()), int(len(all_indexes) * train_ratio))
    val_indexes = list(set(all_indexes.keys()) - set(train_indexes))
    
    train_indexes_tmp = {}
    for key in train_indexes:
        train_indexes_tmp[key] = all_indexes[key]
        
    val_indexes_tmp = {}
    for key in val_indexes:
        val_indexes_tmp[key] = all_indexes[key]
    
    if device == 0:
        json.dump(train_indexes_tmp, open(f'data/train/{align_type}/indexes/train_indexes.json', 'w'), indent=4)
        json.dump(val_indexes_tmp, open(f'data/train/{align_type}/indexes/val_indexes.json', 'w'),  indent=4)
    
    return train_indexes_tmp, val_indexes_tmp

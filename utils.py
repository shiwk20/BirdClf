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
    print(path, 'folder num: {}, img num: {}'.format(count1, count2))
    
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

def get_logger(type, log_path = 'log/logs'):
    os.makedirs(log_path, exist_ok=True)
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


if __name__ == '__main__':
    get_folder_num('data/train')
    get_folder_num('data/valid')
    get_folder_num('data/test')
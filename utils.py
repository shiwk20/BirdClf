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
    
# 使得结果可复现，见https://blog.csdn.net/weixin_43135178/article/details/118768531
def set_seed(seed=42):
    random.seed(seed)  # Python的随机性
    os.environ['PYTHONHASHSEED'] = str(seed)  # 设置Python哈希种子，为了禁止hash随机化，使得实验可复现
    np.random.seed(seed)  # numpy的随机性
    torch.manual_seed(seed)  # torch的CPU随机性，为CPU设置随机种子
    torch.cuda.manual_seed(seed)  # torch的GPU随机性，为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.   torch的GPU随机性，为所有GPU设置随机种子
    torch.backends.cudnn.deterministic = True # 选择确定性算法
    torch.backends.cudnn.benchmark = False # if benchmark=True, deterministic will be False

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
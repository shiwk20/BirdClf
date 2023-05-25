import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import torch
import numpy as np
from utils import L2_dist, divide_train_val
from omegaconf import OmegaConf
from utils import instantiation, get_logger
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
import json
from evaluate import get_dists, find_best_threhold, get_accuracy

# given threhold, cal the accuracy of all dataset
# or find the best threhold and cal the accuracy of all dataset
def eval_all(model, dataloader, device, threshold = -1):
    dists, labels = get_dists(model, dataloader, device)
    
    print(len(dists))
    if threshold == -1:
        threshold, accuracy = find_best_threhold(np.arange(0, 10, 0.01), dists, labels)
    else:
        accuracy = get_accuracy(threshold, dists, labels)
    return accuracy, threshold

def test(model, test_dataloader, all_dataloader, logger, device, align_type):
    train_accuracy, best_threhold = eval_all(model, all_dataloader, device)
    logger.info('best threshold: {}'.format(best_threhold))
    logger.info('train accuracy: {}'.format(train_accuracy))
    
    dists, labels = get_dists(model, test_dataloader, device)
    pred = {}
    for idx, dist in enumerate(dists):
        pred[labels[idx]] = 1 if dist < best_threhold else 0
    
    f = open(f"test_{align_type}.txt", 'w') 
    for i in range(len(pred)):
        print(pred[i], file = f)
    f.close()

if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument('-t', '--type', help = 'align type(mtcnn or landmark), default: mtcnn', type = str, default = 'mtcnn')
    args = parse.parse_args()
    logger = get_logger('test')
    logger.info('align_type: {}'.format(args.type))
    
    device = 'cuda:1'
    config = OmegaConf.load(f'configs/test_{args.type}.yaml')
    model = instantiation(config.model)
    if os.path.isfile(model.ckpt_path):
        _, _ = model.load_ckpt(model.ckpt_path, logger)
    model.eval()
    model.to(device)
    
    test_dataset = instantiation(config.data.test)
    test_dataloader = DataLoader(test_dataset,
                                batch_size = config.data.batch_size,
                                shuffle = False,
                                num_workers = config.data.num_workers,
                                pin_memory = True,
                                drop_last = False)

    all_indexes = json.load(open(f'data/train/{args.type}/indexes/all_indexes.json', 'r'))
    all_indexes = {'val_indexes': all_indexes}
    
    all_dataset = instantiation(config.data.validation, all_indexes)
    all_dataloader = DataLoader(all_dataset,
                                batch_size = config.data.batch_size,
                                shuffle = True, 
                                num_workers = config.data.num_workers,
                                pin_memory = True,
                                drop_last = False)
    
    test(model, test_dataloader, all_dataloader, logger, device, args.type)
    
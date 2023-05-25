from sklearn.model_selection import KFold
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
import torch
import numpy as np
from utils import L2_dist
from omegaconf import OmegaConf
from utils import instantiation, get_logger
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import divide_train_val
import json
import argparse

def get_dists(model, val_dataloader, device):
    labels = []
    dists = []
    
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc='get distance', leave=False):
            A_img, B_img = batch['A'].to(device), batch['B'].to(device)
            A_fea, B_fea = model(A_img), model(B_img)
            distance = L2_dist(A_fea, B_fea)
            
            dists += distance.cpu().numpy().tolist()
            labels += batch['label'].cpu().numpy().tolist()
    return dists, labels
            
def find_best_threhold(thresholds, dists, labels):
    best_accuracy = 0
    for threshold in thresholds:
        accuracy = get_accuracy(threshold, dists, labels)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_threshold = threshold
    return best_threshold, best_accuracy
    
def get_accuracy(threshold, dists, labels):
    pred = []
    for dist in dists:
        pred.append(1) if dist < threshold else pred.append(0)
    pred, labels = np.array(pred), np.array(labels)
    accuracy = np.mean(pred == labels)
    return accuracy

def evaluate(model, val_dataloader, logger, device):
    dists, labels = get_dists(model, val_dataloader, device)

    Accuracies = []
    Thresholds = []
    kf = KFold(n_splits=10, shuffle=True)
    thresholds = np.arange(0, 10, 0.01)
    
    for train_indexes, test_indexes in kf.split(dists):
        best_threshold, best_accuracy = find_best_threhold(thresholds, np.array(dists)[train_indexes], np.array(labels)[train_indexes])
        logger.info('best acc on train_set: {}, best threshold on train set: {}'.format(best_accuracy, best_threshold))
        Accuracies.append(get_accuracy(best_threshold, np.array(dists)[test_indexes], np.array(labels)[test_indexes]))
        Thresholds.append(best_threshold)
    logger.info('test accuracy: {}, test threshold: {}'.format(Accuracies, Thresholds))
    logger.info('Accuracy: {:.4f} Threshold: {:.4f}'.format(np.mean(Accuracies), np.mean(Thresholds)))
    return np.mean(Accuracies), np.mean(Thresholds)

if __name__ == '__main__':
    logger = get_logger('evaluate')
    device = 'cuda:1'
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--type', help = 'align type(mtcnn or landmark), default: mtcnn', type = str, default = 'mtcnn')
    parser.add_argument('-r', '--train_ratio', help = 'align type(mtcnn or landmark), default: 0.85', type = int, default = 0.85)
    parser.add_argument('-s', '--seed', help = 'align type(mtcnn or landmark), default: 0', type = int, default = 0)
    args = parser.parse_args()
    config_path = f'configs/evaluate_{args.type}.yaml'
    config = OmegaConf.load(config_path)
    
    model = instantiation(config.model)
    if os.path.isfile(model.ckpt_path):
        _, _ = model.load_ckpt(model.ckpt_path, logger)
    model.eval()
    model.to(device)

    _, val_indexes = divide_train_val(seed = 0, align_type = args.type, train_ratio = args.train_ratio)
    val_indexes = {'val_indexes': val_indexes}
    val_dataset = instantiation(config.data.validation, val_indexes)
    val_dataloader = DataLoader(val_dataset,
                                batch_size = config.data.batch_size,
                                shuffle = True, 
                                num_workers = config.data.num_workers,
                                pin_memory = True,
                                drop_last = False)
    
    accuracy, threshold = evaluate(model, val_dataloader, logger, device)
    logger.info('Accuracy: {:.4f}, Threshold: {:.4f}'.format(accuracy, threshold))
    
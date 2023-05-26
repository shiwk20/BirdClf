import torch
import numpy as np
from omegaconf import OmegaConf
from utils import instantiation, get_logger
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from utils import get_logger, instantiation, set_seed
from sklearn.metrics import classification_report
import argparse
import matplotlib.pyplot as plt

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

# 训练记录曲线, 包括loss, acc和lr曲线
def plot_curve(val_interval, epoch, train_accs, val_accs, train_losses, lrs, save_path):
    save_path = os.path.join(save_path, "imgs")
    os.makedirs(save_path, exist_ok = True)
    
    # 作acc + lr曲线
    plt.rcdefaults()
    _, plot1 = plt.subplots(dpi=200)
    # 以val_interval为间隔，从0到epoch生成等差数列
    val_epochs = np.linspace(0, epoch, num=epoch // val_interval + 1)
    train_epochs = np.linspace(0, epoch, num=epoch+1)
    plot1.plot(val_epochs, train_accs, color='tab:blue', label='train_accs')
    plot1.plot(val_epochs, val_accs, color='tab:red', label='val_accs')
    plot1.set_xlabel('Epoch')
    plot1.set_ylabel('Accuracy')
    plot1.legend(loc='upper left')
    plot1.set_ylim(min(train_accs[0], val_accs[0]) * 0.9, max(train_accs[-1], val_accs[-1]) * 1.1)
    
    plot2 = plot1.twinx()
    plot2.plot(train_epochs, lrs, color='tab:green', label='learning rate', ls='--')
    plot2.set_ylabel('Learning rate')
    plot2.legend(loc='upper right')
    plot2.set_ylim(min(lrs) * 0.9, max(lrs) * 1.1)
    plot2.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    plot1.grid(True)
    plt.title("Train and Valid Accuracies & Learning Rates")
    plt.box(True)
    plt.savefig(os.path.join(save_path, "accs_lrs.png"), bbox_inches = 'tight')
    plt.close()

    # 作loss曲线
    plt.rcdefaults()
    plt.figure(dpi=200)
    # 运用指数平滑，使loss曲线更加平滑
    alpha = 0.8
    loss_smooth = [train_losses[0]]
    for i in range(1, len(train_losses)):
        loss_smooth.append(loss_smooth[-1] * alpha + train_losses[i] * (1 - alpha))
    plt.plot(train_epochs, loss_smooth, "b-", lw=0.5)
    plt.title("Train Losses(smooth=%.1f)"%(alpha))
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.box(True)
    plt.savefig(os.path.join(save_path, "losses.png"), bbox_inches = 'tight')
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--config', type=str, default='', help='test config file path')
    parser.add_argument('--device', type=int, default=0, help='device id')
    args = parser.parse_args()
    
    device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu')
    config = OmegaConf.load(args.config)
    set_seed(config.seed)
    
    assert config.resume and os.path.isfile(config.ckpt_path), 'must resume from a valid checkpoint to test'
    res_path = res_path = os.path.dirname(os.path.dirname(config.ckpt_path))
    log_path = os.path.join(res_path, 'logs')
    
    logger = get_logger(log_path, 'test')
    logger.info('config: {}'.format(config.pretty()))
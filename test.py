import torch
import numpy as np
from omegaconf import OmegaConf
from utils import instantiation, get_logger
import os
from summary import summary_string
import warnings
import cv2
from torchvision import transforms
from PIL import Image
from torch import nn
warnings.filterwarnings("ignore")
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from dataset import TestDataset
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

# 利用Grad-CAM进行可视化，参考: https://blog.csdn.net/sinat_37532065/article/details/103362517
def CAM(model, N_imgs = 4, img_size = 224):
    train_mean = [0.4740, 0.4693, 0.3956]
    train_std = [0.2039, 0.2007, 0.2058]
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(train_mean, train_std)]
    )
    
    imgs_path = os.path.join(config.data.data_path, "test")
    save_path = os.path.join(res_path, "imgs")
    
    # 从img_path中随机选取N_imgs * N_imgs类，随后从每一类中随机选取一张图片进行可视化
    labels = os.listdir(imgs_path)
    labels.sort()
    choose_labels = np.random.choice(os.listdir(imgs_path), size = N_imgs * N_imgs, replace = False)
    imgs = []
    titles = []
    colors = []

    for i, label in enumerate(tqdm(choose_labels)):
        img_file = np.random.choice(os.listdir(os.path.join(imgs_path, label)))
        img_path = os.path.join(imgs_path, label, img_file)

        img = Image.open(img_path).convert('RGB')
        img = transform(img).to(device)
        
        # auxiliary function
        def extract(g):
            global features_grad
            features_grad = g

        # inference
        model.eval()
        output, features = model(img.unsqueeze(0), visualize=True)
        output = nn.Softmax(dim=1)(output)  # (1, l)
        pred = torch.argmax(output, dim=1).item()
        pred_class = output[:, pred]
        pred_class_name = labels[pred]
        proba = output[0, pred].item()

        # generate grad-cam
        features.register_hook(extract)
        pred_class.backward()
        grads = features_grad
        pooled_grads = torch.nn.functional.adaptive_avg_pool2d(grads, (1, 1))
        pooled_grads = pooled_grads[0]
        features = features[0]
        for i in range(features.shape[0]):
            features[i, :, :] *= pooled_grads[i, :, :]
        heatmap = features.detach().cpu().numpy()
        heatmap = np.mean(heatmap, axis=0)
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[2]))
        heatmap = 255 - np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        show_img = ((img.cpu().numpy() * torch.tensor(train_std).unsqueeze(1).unsqueeze(1).numpy() + torch.tensor(train_mean).unsqueeze(1).unsqueeze(1).numpy()) * 255).transpose(1, 2, 0)
        superimposed_img = heatmap * 0.3 + show_img * 0.7
        imgs.append(superimposed_img.astype(np.uint8))
        title = "%s (%.2f%%)\n%s"%(pred_class_name, proba * 100, label)
        titles.append(title)
        if pred_class_name == label:
            colors.append("black")
        else:
            colors.append("red")
        
    # plot: 作图
    _, axes = plt.subplots(N_imgs, N_imgs, figsize=(15, 15), dpi=400)
    plt.subplots_adjust(wspace=0.5, hspace=0.01)
    plt.style.use('ggplot')
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        ax.imshow(img)
        ax.set_xlabel(titles[i], color=colors[i])
        ax.set_xticks([])
        ax.set_yticks([])
    plt.savefig(os.path.join(save_path, 'CAM.png'))
    logger.info("CAM visualization finished!")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--config', type=str, default='', help='test config file path')
    parser.add_argument('--device', type=int, default=0, help='device id')
    parser.add_argument('--N_imgs', type=int, default=5, help='num of imgs in CAM')
    args = parser.parse_args()
    
    device = torch.device('cuda:{}'.format(args.device) if torch.cuda.is_available() else 'cpu')
    config = OmegaConf.load(args.config)
    set_seed(config.random_seed)
    
    assert config.resume and os.path.isfile(config.ckpt_path), 'must resume from a valid checkpoint to test'
    res_path = res_path = os.path.dirname(os.path.dirname(config.ckpt_path))
    log_path = os.path.join(res_path, 'logs')
    model_name = os.path.basename(config.ckpt_path)[:-4]
    logger = get_logger(log_path, 'test_' + model_name)
    logger.info('config: {}'.format(config))
    
    # model
    model = instantiation(config.model)
    _, _, _, _, _, _ = model.load_ckpt(config.ckpt_path, logger)
    model.eval()
    model.to(device)
    
    CAM(model, N_imgs = args.N_imgs)
    
    # model summary: 计算模型参数量, 保存模型结构报告
    model_report = summary_string(model, input_size=(3, 224, 224), batch_size=-1)
    logger.info(model_report)
    
    # data
    test_dataset = TestDataset(config.data.img_size, config.data.data_path)
    test_dataloader = DataLoader(test_dataset, 
                                batch_size=config.data.batch_size, 
                                shuffle=True,
                                num_workers=config.data.num_workers,
                                pin_memory=True,
                                drop_last=False)
    
    evaluate(model, test_dataloader, logger, device, 'test')

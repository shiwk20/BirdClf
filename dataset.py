import torch
import numpy as np
from torchvision import datasets,  transforms
import os
import random
import math
from tqdm import tqdm
from matplotlib import pyplot as plt

def cal_mean_std():
    """
    计算训练集的均值和方差
    """
    data_path = "data"
    train_data = datasets.ImageFolder(os.path.join(data_path, "train"), transform=transforms.ToTensor())
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for img, _ in tqdm(train_data):
        mean += img.mean(dim=(1, 2))
        std += img.std(dim=(1, 2))
    mean /= len(train_data)
    std /= len(train_data)
    print("mean: ", mean)
    print("std: ", std)
train_mean = [0.4740, 0.4693, 0.3956]
train_std = [0.2039, 0.2007, 0.2058]

# 统计训练数据集每一类的个数，并作图
def plot_train_num(data_path = 'data'):
    train_path = os.path.join(data_path, 'train')
    train_num = []
    labels = os.listdir(train_path)
    labels.sort()
    for label in labels:
        train_num.append(len(os.listdir(os.path.join(train_path, label))))
    classes = np.linspace(0, len(labels)-1, len(labels))
    plt.figure(figsize=(25, 10))
    plt.bar(classes, train_num, width=0.5, align='center')
    plt.xticks(rotation=90)
    plt.savefig(os.path.join(data_path, 'train_num.png'))
    
def TrainDataset(img_size, data_path, augment=False, resized_crop=False, herizon_flip=False, vertical_flip=False, random_affine=False):
    data_path = os.path.join(data_path, "train")
    if not os.path.exists(data_path):
        raise FileNotFoundError("train dataset not found")
    
    if augment:
        transform = []
        if resized_crop:
            transform.append(transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)))
        if herizon_flip:
            transform.append(transforms.RandomHorizontalFlip(p=0.5))
        if vertical_flip:
            transform.append(transforms.RandomVerticalFlip(p=0.5))
        if random_affine:
            transform.append(transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.9, 1.1)))
        transform.append(transforms.CenterCrop(img_size))
        transform.append(transforms.ToTensor())
        transform.append(transforms.Normalize(train_mean, train_std))
        transform = transforms.Compose(transform)
    else:
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(train_mean, train_std)]
        )
    dataset = datasets.ImageFolder(data_path, transform=transform)
    return dataset

def ValDataset(img_size, data_path):
    data_path = os.path.join(data_path, "valid")
    if not os.path.exists(data_path):
        raise FileNotFoundError("valid dataset not found")
    
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(train_mean, train_std)]
    )
    dataset = datasets.ImageFolder(data_path, transform=transform)
    return dataset

def TestDataset(img_size, data_path):
    data_path = os.path.join(data_path, "test")
    if not os.path.exists(data_path):
        raise FileNotFoundError("test dataset not found")
    
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(train_mean, train_std)]
    )
    dataset = datasets.ImageFolder(data_path, transform=transform)
    return dataset

# 在训练集中选取class_num个类
def select_class(data_path = 'data', class_num = 16, select_num = 5):
    train_path = os.path.join(data_path, 'train')
    labels = os.listdir(train_path)
    labels.sort()
    # 将labels分为select_num组
    labels = np.array_split(labels, select_num)
    for i in range(select_num):
        save_path = os.path.join(data_path, 'test_class', str(i))
        os.makedirs(os.path.join(save_path, 'train'), exist_ok=True)
        os.makedirs(os.path.join(save_path, 'valid'), exist_ok=True)
        os.makedirs(os.path.join(save_path, 'test'), exist_ok=True)
        # 从每一组中选取class_num个类
        select_labels = random.sample(list(labels[i]), class_num)
        # 将选取的类放入select文件夹中
        for label in select_labels:
            if label.find(' ') != -1:
                label = label.replace(' ', '\ ')
            os.system('cp -r ' + os.path.join(train_path, label) + ' ' + os.path.join(save_path, 'train'))
            os.system('cp -r ' + os.path.join(data_path, 'valid', label) + ' ' + os.path.join(save_path, 'valid'))
            os.system('cp -r ' + os.path.join(data_path, 'test', label) + ' ' + os.path.join(save_path, 'test'))

# 在前面select_class 0的基础上，改变训练集类别的均衡度
# 参考了https://blog.csdn.net/javastart/article/details/99221361
def select_balance(root_path = 'data', data_path = 'data/test_class/0'):
    save_path = os.path.join(root_path, 'test_balance')
    method_num = 0
    ratio_lists = []
    # 此处使用5种方式改变样本的不平衡度
    labels = os.listdir(os.path.join(data_path, 'train'))
    labels.sort()
    imgs_num = [len(os.listdir(os.path.join(data_path, 'train', label))) for label in labels]
    min_num = min(imgs_num)
    
    # 求出每种方法下每一类的样本数
    ## 0 每一类的数据完全一样多
    ratio_list = [min_num] * len(labels)
    ratio_lists.append(ratio_list)
    method_num += 1
    
    ## 1 一半的类的数据是另一半的两倍
    ratio_list = [min_num // 2] * len(labels)
    ratio_list[:len(labels) // 2] = [i * 2 for i in ratio_list[:len(labels) // 2]]
    ratio_lists.append(ratio_list)
    method_num += 1
    
    ## 2 只有一类的数据比其他类多(3 倍)
    ratio_list = [min_num // 3] * len(labels)
    ratio_list[0] *= 3
    ratio_lists.append(ratio_list)
    method_num += 1
    
    ## 3 数据个数呈线性分布
    ratio_list = [int(i / len(labels) * min_num) for i in range(1, len(labels) + 1)]
    ratio_lists.append(ratio_list)
    method_num += 1
    
    ## 4 数据个数呈指数分布
    ratio_list = [int(math.exp(i / len(labels)) / math.exp(1) * min_num) for i in range(1, len(labels) + 1)]
    ratio_lists.append(ratio_list)
    method_num += 1

    for i in tqdm(range(method_num)):
        print(ratio_lists[i])
        os.makedirs(os.path.join(save_path, str(i), 'train'), exist_ok=True)
        os.system('cp -r ' + os.path.join(data_path, 'valid') + ' ' + os.path.join(save_path, str(i)))
        os.system('cp -r ' + os.path.join(data_path, 'test') + ' ' + os.path.join(save_path, str(i)))
        
        for j in tqdm(range(len(labels)), leave=False):
            label = labels[j]
            os.makedirs(os.path.join(save_path, str(i), 'train', label), exist_ok=True)
            imgs = os.listdir(os.path.join(data_path, 'train', label))
            
            if label.find(' ') != -1:
                label = label.replace(' ', '\ ')
            label_num = ratio_lists[i][j]
            imgs.sort()
            imgs = random.sample(imgs, label_num)
            for img in imgs:
                os.system('cp ' + os.path.join(data_path, 'train', label, img) + ' ' + os.path.join(save_path, str(i), 'train', label))
    


if __name__ == "__main__":
    cal_mean_std()
    # select_class()
    
    # plot_train_num('data/test_class/0')
    
    # select_balance()
    
    # plot_train_num('data/test_balance/0')
    # plot_train_num('data/test_balance/1')
    # plot_train_num('data/test_balance/2')
    # plot_train_num('data/test_balance/3')
    # plot_train_num('data/test_balance/4')
    
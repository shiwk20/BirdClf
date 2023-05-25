from torch.utils.data import Dataset
import torch
import numpy as np
import json
from torchvision import datasets,  transforms
import os
import random
from tqdm import tqdm
from PIL import Image

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

def TrainDataset(img_size, data_path, augment=False):
    data_path = os.path.join(data_path, "train")
    if not os.path.exists(data_path):
        raise FileNotFoundError("train dataset not found")
    
    if augment:
        transform = transforms.Compose([
            transforms.RandomResizedCrop(img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomAffine(degrees=30, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=15),
            transforms.CentorCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(train_mean, train_std)]
        )
    else:
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(train_mean, train_std)]
        )
    dataset = datasets.ImageFolder(data_path, transform=transform)
    return dataset

def ValidDataset(img_size, data_path):
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

if __name__ == "__main__":
    cal_mean_std()
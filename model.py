from torch import nn
import torch
import torch.nn.functional as F
import os
from torch.nn.utils import prune
from torchvision.models import convnext_base

def make_layer(in_channel, out_channel, block_num, stride):
    Blocks = []
    for i in range(block_num):
        if i == 0:
            Blocks.append(BottleNeck(in_channel, out_channel, stride, down_sample = True))
        else:
            Blocks.append(BottleNeck(out_channel, out_channel))
    return nn.Sequential(*Blocks)
    
class BottleNeck(nn.Module):
    def __init__(self, in_channel, out_channel, stride = 1, down_sample = False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel // 4, kernel_size = 1, stride = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(out_channel // 4)
        
        self.conv2 = nn.Conv2d(out_channel // 4, out_channel // 4, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(out_channel // 4)
        
        self.conv3 = nn.Conv2d(out_channel // 4, out_channel, kernel_size = 1, stride = 1, bias = False)
        self.bn3 = nn.BatchNorm2d(out_channel)
        
        if down_sample:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size = 1, stride = stride, bias = False),
                nn.BatchNorm2d(out_channel)
            )
        self.down_sample = down_sample

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.down_sample:
            out = out + self.downsample(x)
        else:
            out = out + x
        
        out = F.relu(out)
        return out

class BirdClf(nn.Module):
    def __init__(self, embed_size: int = 525, compress=False):
        super().__init__()
        self.compress = compress
        self.conv1 = nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        
        self.layer1 = make_layer(64, 256, block_num = 3, stride = 1)
        self.layer2 = make_layer(256, 512, block_num = 4, stride = 2)
        self.layer3 = make_layer(512, 1024, block_num = 6, stride = 2)
        self.layer4 = make_layer(1024, 2048, block_num = 3, stride = 2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.emb = nn.Linear(2048, embed_size)

    def forward(self, x, visualize=False):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        features = self.layer4(x)
        x = self.avgpool(features)
        
        x = x.flatten(1)
        x = self.emb(x)
        if visualize:
            return x, features
        return x
    
    def load_ckpt(self, ckpt_path, logger):
        states = torch.load(ckpt_path, map_location="cpu")
        if "state_dict" in list(states.keys()):
            state_dict = states["state_dict"]
        else:
            state_dict = states
        missing, unexpected = self.load_state_dict(state_dict, strict = False)
        logger.info(f"load checkpoint from {ckpt_path}, missing keys: {missing}, unexpected keys: {unexpected}")
        
        if "epoch" in list(states.keys()):
            cur_epoch = states["epoch"] + 1
        else:
            cur_epoch = 0
        if 'optimizer' in list(states.keys()):
            optim_state_dict = states['optimizer']
        else:
            optim_state_dict = None
        
        if 'train_accs' in list(states.keys()):
            train_accs = states['train_accs']
        else:
            train_accs = []
        if 'val_accs' in list(states.keys()):
            val_accs = states['val_accs']
        else:
            val_accs = []
        if 'train_losses' in list(states.keys()):
            train_losses = states['train_losses']
        else:
            train_losses = []
        if 'lrs' in list(states.keys()):
            lrs = states['lrs']
        else:
            lrs = []
        
        return cur_epoch, optim_state_dict, train_accs, val_accs, train_losses, lrs
        
    def save_ckpt(self, save_path, epoch, train_accs, val_accs, train_losses, lrs, optimizer, logger):
        os.makedirs(save_path, exist_ok = True)
        save_path = os.path.join(save_path, "epoch_{}_acc_{:4f}.pth".format(epoch, val_accs[-1]))
        
        states = {
            'epoch': epoch,
            'state_dict': self.state_dict(),
            'train_accs': train_accs,
            'val_accs': val_accs,
            'train_losses': train_losses,
            'lrs': lrs,
            'optimizer': optimizer.state_dict()
        }

        torch.save(states, save_path)
        logger.info('save ckpt to {}'.format(save_path)) 
        
    # 对模型进行压缩，主要是针对模型中的卷积层应用l1_unstructured剪枝, 剪枝比例为prune_ratio, 
    def model_compress(self, prune_ratio=0.3):
        paras_to_prune = [(self.conv1, 'weight')] + \
                                [(self.layer1[i].conv1, 'weight') for i in range(3)] + \
                                [(self.layer1[i].conv2, 'weight') for i in range(3)] + \
                                [(self.layer1[i].conv3, 'weight') for i in range(3)] + \
                                [(self.layer2[i].conv1, 'weight') for i in range(4)] + \
                                [(self.layer2[i].conv2, 'weight') for i in range(4)] + \
                                [(self.layer2[i].conv3, 'weight') for i in range(4)] + \
                                [(self.layer3[i].conv1, 'weight') for i in range(6)] + \
                                [(self.layer3[i].conv2, 'weight') for i in range(6)] + \
                                [(self.layer3[i].conv3, 'weight') for i in range(6)] + \
                                [(self.layer4[i].conv1, 'weight') for i in range(3)] + \
                                [(self.layer4[i].conv2, 'weight') for i in range(3)] + \
                                [(self.layer4[i].conv3, 'weight') for i in range(3)]
        for module, name in paras_to_prune:
            prune.l1_unstructured(module, name=name, amount=prune_ratio)
        
    
    # 移除模型中剪枝相关的key-value，便于测试和保存
    def remove_prune(self):
        paras_to_prune = [(self.conv1, 'weight')] + \
                                [(self.layer1[i].conv1, 'weight') for i in range(3)] + \
                                [(self.layer1[i].conv2, 'weight') for i in range(3)] + \
                                [(self.layer1[i].conv3, 'weight') for i in range(3)] + \
                                [(self.layer2[i].conv1, 'weight') for i in range(4)] + \
                                [(self.layer2[i].conv2, 'weight') for i in range(4)] + \
                                [(self.layer2[i].conv3, 'weight') for i in range(4)] + \
                                [(self.layer3[i].conv1, 'weight') for i in range(6)] + \
                                [(self.layer3[i].conv2, 'weight') for i in range(6)] + \
                                [(self.layer3[i].conv3, 'weight') for i in range(6)] + \
                                [(self.layer4[i].conv1, 'weight') for i in range(3)] + \
                                [(self.layer4[i].conv2, 'weight') for i in range(3)] + \
                                [(self.layer4[i].conv3, 'weight') for i in range(3)]
        for module, name in paras_to_prune:
            prune.remove(module, name)


if __name__ == "__main__":
    model = BirdClf()
    model.model_compress()
    model.remove_prune()
    x = torch.randn(64, 3, 224, 224)
    print(model(x).shape)
from torch import nn
import torch
import torch.nn.functional as F
import os


def make_layer(in_channel, out_channel, block_num, stride):
    Blocks = []
    for i in range(block_num):
        if i == 0:
            Blocks.append(Block(in_channel, out_channel, stride, down_sample = True))
        else:
            Blocks.append(Block(out_channel, out_channel))
    return nn.Sequential(*Blocks)
    
class Block(nn.Module):
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

# use ResNet50 as backbone
class BirdClf_R(nn.Module):
    def __init__(self, ckpt_path='', save_ckpt_path='log/ckpt', embed_size: int = 525):
        super().__init__()
        self.start_time = None
        self.ckpt_path = ckpt_path
        self.save_ckpt_path = save_ckpt_path
        self.conv1 = nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        
        self.layer1 = make_layer(64, 256, block_num = 3, stride = 1)
        self.layer2 = make_layer(256, 512, block_num = 4, stride = 2)
        self.layer3 = make_layer(512, 1024, block_num = 6, stride = 2)
        self.layer4 = make_layer(1024, 2048, block_num = 3, stride = 2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.emb = nn.Linear(2048, embed_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        
        x = x.flatten(1)
        x = self.emb(x)
        
        return x

    def load_ckpt(self, ckpt_path, logger):
        states = torch.load(ckpt_path, map_location="cpu")
        if "state_dict" in list(states.keys()):
            state_dict = states["state_dict"]
            self.start_time = os.path.dirname(ckpt_path).split('/')[-1]
        else:
            state_dict = states
        missing, unexpected = self.load_state_dict(state_dict, strict = False)
        logger.info(f"load checkpoint from {ckpt_path}, missing keys: {missing}, unexpected keys: {unexpected}")
        if "epoch" in list(states.keys()):
            cur_epoch = states["epoch"]
        else:
            cur_epoch = 0
        if 'optimizer' in list(states.keys()):
            optim_state_dict = states['optimizer']
        else:
            optim_state_dict = None
        
        return cur_epoch, optim_state_dict
        
    def save_ckpt(self, save_path, epoch, best_accuracy, optimizer, logger, start_time, is_final = False):
        states = {'state_dict': self.state_dict(), 'epoch': epoch, 
                'best_accuracy': best_accuracy, 'optimizer': optimizer.state_dict()}
        if self.start_time is not None:
            start_time = self.start_time
        save_path = os.path.join(save_path, start_time)
        os.makedirs(save_path, exist_ok = True)
        if is_final:
            save_path = os.path.join(save_path, "last_epoch_{}_acc_{:.4f}.pth".format(epoch, best_accuracy))
        else:
            save_path = os.path.join(save_path, "epoch_{}_acc_{:4f}.pth".format(epoch, best_accuracy))
        torch.save(states, save_path)
        logger.info('save ckpt to {}'.format(save_path))

if __name__ == "__main__":
    model = BirdClf_R()
    x = torch.randn(64, 3, 224, 224)
    print(model(x).shape)
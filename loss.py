from torch import nn
import torch
import numpy as np

class CrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, reduction='mean', label_smoothing=0.0):
        """
        weight: 每个类别的权重, (n_classes,)
        reduction: 'none', 'mean', 'sum'
        label_smoothing: 标签平滑, 用于解决过拟合问题
        """
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(weight=weight, reduction=reduction, label_smoothing=label_smoothing)

    def forward(self, logits, labels):
        """
        logits: (b, n_classes)
        labels: (b,)
        """
        return self.criterion(logits, labels)
    
if __name__ == '__main__':
    criterion = CrossEntropyLoss()
    logits = torch.randn(10, 5)
    labels = torch.randint(0, 5, (10,))
    loss = criterion(logits, labels)
    print(loss)
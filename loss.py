from torch import nn
from utils import L2_dist
import torch
import numpy as np

class TripletLoss(nn.Module):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, anc, pos, neg):
        pos_dist = L2_dist(anc, pos)
        neg_dist = L2_dist(anc, neg)
        
        hard_indexes = self.get_hard(pos_dist, neg_dist)
        
        pos_hard_dist = pos_dist[hard_indexes]
        neg_hard_dist = neg_dist[hard_indexes]

        return torch.sum(pos_hard_dist - neg_hard_dist + self.alpha) / max(1, len(hard_indexes[0])), len(hard_indexes[0])

    def get_hard(self, pos_dist, neg_dist):
        hard_mask = (pos_dist - neg_dist + self.alpha > 0).cpu().numpy().flatten()
        hard_indexes = np.where(hard_mask == True)
        return hard_indexes
    
if __name__ == '__main__':
    criterion = TripletLoss(1)
    a = torch.randn(64, 512)
    b = torch.randn(64, 512)
    c = torch.randn(64, 512)
    
    loss = criterion(a, b, c)
    print(loss)
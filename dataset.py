from torch.utils.data import Dataset
import numpy as np
import json
import os
import random
from tqdm import tqdm
from PIL import Image

def process_image(image):
    if not image.mode == 'RGB':
        image = image.convert('RGB')
    image = np.array(image).astype(np.uint8)
    image = (image / 127.5 - 1.0).astype(np.float32)   
    image = image.transpose(2, 0, 1)
    return image

# produce triplet images
class TripletTrainDataset(Dataset):
    def __init__(self, train_indexes, triplets_num, align_type, align_size) -> None:
        super().__init__()
        align_size_0 = int(align_size.split(',')[0])
        align_size_1 = int(align_size.split(' ')[1])
        self.img_path = f'data/train/{align_type}/align{align_size_0}x{align_size_1}'
        self.triplets_num = triplets_num
        self.train_indexes = train_indexes
        self.triplets = {}
        
        print('get triplets...')
        for i in tqdm(range(triplets_num), desc='get triplets', leave=False):
            tmp_triplet = self.get_triplet()
            self.triplets[i] = tmp_triplet
    
    def get_triplet(self):
        triplet = {}
        anc_name = random.choice(list(self.train_indexes.keys()))
        while len(self.train_indexes[anc_name]) < 2:
            anc_name = random.choice(list(self.train_indexes.keys()))
        anc_idx, pos_idx = random.sample(self.train_indexes[anc_name], 2) 
        
        neg_name = random.choice(list(self.train_indexes.keys()))
        while neg_name == anc_name:
            neg_name = random.choice(list(self.train_indexes.keys()))
        neg_idx = random.choice(self.train_indexes[neg_name])
        
        triplet['anc_path'] = os.path.join(os.path.basename(anc_name), anc_idx)
        triplet['pos_path'] = os.path.join(os.path.basename(anc_name), pos_idx)
        triplet['neg_path'] = os.path.join(os.path.basename(neg_name), neg_idx)
        return triplet
        
    def __len__(self):
        return self.triplets_num
            
    def __getitem__(self, index):
        batch = {}
        
        anc_image = Image.open(os.path.join(self.img_path, self.triplets[index]['anc_path']))
        anc_image = process_image(anc_image)
        pos_image = Image.open(os.path.join(self.img_path, self.triplets[index]['pos_path']))
        pos_image = process_image(pos_image)
        neg_image = Image.open(os.path.join(self.img_path, self.triplets[index]['neg_path']))
        neg_image = process_image(neg_image)
        
        batch['anc'], batch['pos'], batch['neg'] = anc_image, pos_image, neg_image
        return batch
    

# produce pair images
class TripletValDataset(Dataset):
    def __init__(self, val_indexes, pairs_num, align_type, align_size) -> None:
        super().__init__()
        align_size_0 = int(align_size.split(',')[0])
        align_size_1 = int(align_size.split(' ')[1])
        self.img_path = f'data/train/{align_type}/align{align_size_0}x{align_size_1}'
        self.val_indexes = val_indexes
        self.pairs_num = pairs_num
        self.pairs = {}
        
        print('get pairs...')
        for i in tqdm(range(pairs_num // 2), desc='get pos pairs', leave=False):
            tmp_pair = self.get_pos_pair()
            # remove duplication
            while tmp_pair in self.pairs.values():
                tmp_pair = self.get_pos_pair()
            self.pairs[i] = tmp_pair
            
        for i in tqdm(range(pairs_num // 2), desc='get neg pairs', leave=False):
            tmp_pair = self.get_neg_pair()
            # remove duplication
            while tmp_pair in self.pairs.values():
                tmp_pair = self.get_neg_pair()
            self.pairs[i + pairs_num // 2] = tmp_pair
    
    def get_pos_pair(self):
        pair = {}
        anc_name = random.choice(list(self.val_indexes.keys()))
        while len(self.val_indexes[anc_name]) < 2:
            anc_name = random.choice(list(self.val_indexes.keys()))
        anc_idx, pos_idx = random.sample(self.val_indexes[anc_name], 2)

        pair['A'] = os.path.join(os.path.basename(anc_name), anc_idx)
        pair['B'] = os.path.join(os.path.basename(anc_name), pos_idx)
        pair['label'] = 1
        return pair
        
    def get_neg_pair(self):
        pair = {}
        anc_name, neg_name = random.sample(list(self.val_indexes.keys()), 2)
        anc_idx = random.choice(self.val_indexes[anc_name])
        neg_idx = random.choice(self.val_indexes[neg_name])
        
        pair['A'] = os.path.join(os.path.join(os.path.basename(anc_name), anc_idx))
        pair['B'] = os.path.join(os.path.join(os.path.basename(neg_name), neg_idx))
        pair['label'] = 0
        return pair
        
    def __len__(self):
        return self.pairs_num
    
    def __getitem__(self, index):
        batch = {}
        
        A_image = Image.open(os.path.join(self.img_path, self.pairs[index]['A']))
        A_image = process_image(A_image)
        B_image = Image.open(os.path.join(self.img_path, self.pairs[index]['B']))
        B_image = process_image(B_image)
        
        batch['A'], batch['B'], batch['label'] = A_image, B_image, self.pairs[index]['label']
        return batch

# produce pair images
class TestDataset(Dataset):
    def __init__(self, align_type, align_size) -> None:
        super().__init__()
        align_size_0 = int(align_size.split(',')[0])
        align_size_1 = int(align_size.split(' ')[1])
        self.img_path = f'data/test/{align_type}/align{align_size_0}x{align_size_1}'
        
        self.pairs = {}
        print('get test pairs...')
        idx_list = os.listdir(self.img_path)
        idx_list = [int(i) for i in idx_list]
        idx_list.sort()
        for i in idx_list:
            pair = {}
            pair['A'] = os.path.join(str(i), 'A.jpg')
            pair['B'] = os.path.join(str(i), 'B.jpg')
            pair['label'] = i
            self.pairs[i] = pair
    
        
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, index):
        batch = {}
        
        A_image = Image.open(os.path.join(self.img_path, self.pairs[index]['A']))
        A_image = process_image(A_image)
        B_image = Image.open(os.path.join(self.img_path, self.pairs[index]['B']))
        B_image = process_image(B_image)
        
        batch['A'], batch['B'], batch['label'] = A_image, B_image, self.pairs[index]['label']

        return batch
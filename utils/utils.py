from torch.utils.data import Dataset
import pandas as pd
import os
import torch
from skimage import io, transform
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
import albumentations.augmentations.functional as F
from albumentations.pytorch import ToTensorV2
import numpy as np

class DatsetLoader(Dataset):
    def __init__(self, csv_file='TrainingDataSet.csv', root_dir='train2014', transform=None):
        self.name_csv = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.name_csv)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        ref_img_name = os.path.join(self.root_dir,self.name_csv.iloc[idx, 0])
        ref_image    = io.imread(ref_img_name)
        ref_image    = ref_image/np.max(ref_image)
        if(len(ref_image.shape)==2):
            ref_image = color.gray2rgb(ref_image)  
        ref_image = self.transform(image=ref_image)
        return ref_image['image']



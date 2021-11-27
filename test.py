from torch.utils.data import Dataset
import os
import torch
from skimage import io, transform
import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch
import torch.nn as nn
import torchvision.models as models
import albumentations as A
import albumentations.augmentations.functional as F
from albumentations.pytorch import ToTensorV2
from collections import OrderedDict
from models.StyleTransferModels import *
from models.vggmodule import *
from utils.trainingLoop import *
from utils.utils import *
from utils.getstyle import *

dtype = torch.float32
cpu = torch.device('cuda')

def getStylemodel(index,norm):
    if(index==0):
        styleImage   = torch.load('./pretrainedModels/style0.pt')
    elif(index==1):
        styleImage   = torch.load('./pretrainedModels/style1.pt')
    elif(index==2):
        styleImage   = torch.load('./pretrainedModels/style2.pt')
    elif(index==3):
        styleImage   = torch.load('./pretrainedModels/style3.pt')
    elif(index==4):
        styleImage   = torch.load('./pretrainedModels/style4.pt')
    elif(index==5):
        styleImage   = torch.load('./pretrainedModels/style5.pt')
    elif(index==6):
        styleImage   = torch.load('./pretrainedModels/style6.pt')
    elif(index==7):
        styleImage   = torch.load('./pretrainedModels/style7.pt')
    elif(index==8):
        styleImage   = torch.load('./pretrainedModels/style8.pt')
    elif(index==9):
        styleImage   = torch.load('./pretrainedModels/style9.pt')
    elif(index==10):
        styleImage   = torch.load('./pretrainedModels/style10.pt')
    return styleImage



def main(imageName,styleIndicator,norm):
    scratch    =''
    fileRoot   ='./data/'
 
    SampleTransform = A.Compose([
                                 A.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229,0.224,0.225]),
                                 ToTensorV2(),
                                ])
    if(norm==0):
        model = ModelStyleInstance()
    else:
        model = ModelStyleBatch()
    model = ModelStyleInstance()
    saved_state_dict =  getStylemodel(styleIndicator,norm)   
    model.load_state_dict(saved_state_dict, strict=True)
    model = model.to("cuda")
    model.eval()
    contentImage = io.imread(imageName)  
    contentImage = contentImage 
    contentImage = SampleTransform(image=contentImage)
    contentImage = contentImage['image'] 
    contentImage = contentImage.unsqueeze(0) 
    contentImage = contentImage.to('cuda')
    output = model(contentImage)
    output = output.squeeze()
    output = output.permute(1,2,0)
    output = output.squeeze().detach().cpu().numpy()
    io.imsave('output.png',output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Short sample app')
    parser.add_argument('-style', type=int,action="store", dest='style', default=0)
    parser.add_argument('-norm', type=int,action="store", dest='norm', default=0)
    parser.add_argument('-imageName', type=str,action="store", dest='imageName', default='sampleImages/univeristy-utah.jpg')
    args = parser.parse_args()

    main(args.imageName,args.style,args.norm)


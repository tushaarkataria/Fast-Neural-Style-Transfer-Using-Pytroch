from torch.utils.data import Dataset
import os
import torch
from skimage import io, transform
import matplotlib.pyplot as plt
import numpy as np
import argparse
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import torch.optim as optim
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from skimage.transform import SimilarityTransform
from skimage.transform import warp
import copy
from skimage.transform import resize
import albumentations as A
import albumentations.augmentations.functional as F
from albumentations.pytorch import ToTensorV2
from utils import *
from htmlutils import *
from collections import OrderedDict

dtype = torch.float32
cpu = torch.device('cuda')


def main(lr1,wd,epochs,styleIndicator,alpha,alphatv,batchSize,onlyTest):
    

    train_transform = A.Compose([A.Resize(256,256),
                                 A.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
                                 ToTensorV2(),
                                ])
    if(onlyTest==0):
        TrainingSet = NoiseDatsetLoader(csv_file='TrainingDataSet.csv', root_dir='train2014',transform=train_transform)
  
        ## DataLoader 
        batch_size=batchSize

        dataloader_train  = DataLoader(TrainingSet,batch_size=batch_size,num_workers=4)

        directoryName = 'style'+str(styleIndicator)+'alpha'+str(alpha)+'alphatv'+str(alphatv)+'batchSize'+str(batchSize)+'lr'+str(lr1)
        if not os.path.exists(directoryName):
            os.makedirs(directoryName)

        webpage = HTML(directoryName, 'lr = %s, alpha = %s,alphatv =%s, style = %s' % (str(lr1), str(alpha),str(alphatv), str(styleIndicator)))

        ## ************* Start of your Code *********************** ##
        model = ModelStyle()
        styleImage = getStyleImage(styleIndicator)

        optimizer = optim.Adam(model.parameters(), lr=lr1)
        ## ************ End of your code ********************   ##

        ## Train Your Model. Complete the implementation of trainingLoop function above 
        trainingLoop(dataloader_train, styleImage, model,optimizer,epochs,alpha,alphatv,directoryName,webpage)

    train_transform = A.Compose([
                                 A.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225]),
                                 ToTensorV2(),
                                ])
    model = ModelStyle()
    saved_state_dict = torch.load(directoryName+'/best_model_batch.pt')    
    model.load_state_dict(saved_state_dict, strict=True)
    model = model.to("cuda")
    model.eval()
    contentImage = io.imread('chicago.jpg')  
    contentImage = contentImage/np.max(contentImage) 
    contentImage = train_transform(image=contentImage)
    contentImage = contentImage['image'] 
    contentImage = contentImage.unsqueeze(0) 
    contentImage = contentImage.to('cuda')
    output = model(contentImage)
    output = output.squeeze()
    output = output.permute(1,2,0)
    output = output.squeeze().detach().cpu().numpy()
    io.imsave(directoryName+'/Chicago_output.png',output)
    contentImage = io.imread('hoovertowernight.jpg')  
    contentImage = contentImage/np.max(contentImage) 
    contentImage = train_transform(image=contentImage)
    contentImage = contentImage['image'] 
    contentImage = contentImage.unsqueeze(0) 
    contentImage = contentImage.to('cuda')
    output = model(contentImage)
    output = output.squeeze()
    output = output.permute(1,2,0)
    output = output.squeeze().detach().cpu().numpy()
    io.imsave(directoryName+'/hoovertower_output.png',output)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Short sample app')
    parser.add_argument('-lr', type=float,action="store", dest='learningRate', default=1e-3)
    parser.add_argument('-wd', type=float,action="store", dest='wd', default=1e-4)
    parser.add_argument('-epoch', type=int,action="store", dest='epoch', default=2)
    parser.add_argument('-style', type=int,action="store", dest='style', default=0)
    parser.add_argument('-alpha', type=float,action="store", dest='alpha', default=10)
    parser.add_argument('-alphatv', type=float,action="store", dest='alphatv', default=1)
    parser.add_argument('-batch', type=int,action="store", dest='batch', default=2)
    parser.add_argument('-t', type=int,action="store", dest='t', default=0)
    args = parser.parse_args()

    main(args.learningRate,args.wd,args.epoch,args.style,args.alpha,args.alphatv,args.batch,args.t)


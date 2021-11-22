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


def main(lr1,wd,epochs,styleIndicator,alpha,alphatv,batchSize,onlyTest,norm):
    #scratch    ='/scratch/general/nfs1/u1319435/StyleTransfer/'
    #fileRoot ='/scratch/general/nfs1/u1319435/StyleTransfer/'
    scratch    =''
    fileRoot   ='./data/'
 
    train_transform = A.Compose([A.Resize(256,256),
                                 A.CenterCrop(256,256),
                                 A.Normalize(mean=[103.939, 116.779, 123.68],std=[1,1,1], max_pixel_value=1.0),
                                 ToTensorV2(),
                                ])
    SampleTransform = A.Compose([
                                 A.Normalize(mean=[103.939, 116.779, 123.68],std=[1,1,1], max_pixel_value=1.0),
                                 ToTensorV2(),
                                ])
    if(onlyTest==0):
        TrainingSet = DatsetLoader(csv_file='./data/TrainingDataSet.csv', root_dir=fileRoot+'/train2014',transform=train_transform)
  
        ## DataLoader 
        batch_size=batchSize

        dataloader_train  = DataLoader(TrainingSet,batch_size=batch_size,num_workers=4,shuffle=True)

        directoryName = scratch+'style'+str(styleIndicator)+'/alpha'+str(alpha)+'/alphatv'+str(alphatv)+'/batchSize'+str(batchSize)+'/lr'+str(lr1)+'/wd'+str(wd)+'/norm'+str(norm)
        if not os.path.exists(directoryName):
            os.makedirs(directoryName)

        webpage = HTML(directoryName, 'lr = %s, alpha = %s,alphatv =%s, style = %s' % (str(lr1), str(alpha),str(alphatv), str(styleIndicator)))

        ## ************* Start of your Code *********************** ##
        if(norm==0):
            model = ModelStyleInstance()
        else:
            model = ModelStyleBatch()
        styleImage = getStyleImage(styleIndicator)

        optimizer = optim.Adam(model.parameters(), lr=lr1)
        ## ************ End of your code ********************   ##

        vggmodel = VGGModule()
        ## Train Your Model. Complete the implementation of trainingLoop function above 
        trainingLoop(dataloader_train, styleImage, model,optimizer,epochs,alpha,alphatv,directoryName,webpage,train_transform,SampleTransform,batchSize,vggmodel)

    SampleTransform = A.Compose([
                                 A.Normalize(mean=[103.939, 116.779, 123.68],std=[1,1,1], max_pixel_value=1.0),
                                 ToTensorV2(),
                                ])
    ## Inference for network saved in the last batch
    model = ModelStyleInstance()
    saved_state_dict = torch.load(directoryName+'/best_model_batch.pt')    
    model.load_state_dict(saved_state_dict, strict=True)
    model = model.to("cuda")
    model.eval()
    contentImage = io.imread('sampleImages/chicago.jpg')  
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
    contentImage = io.imread('sampleImages/hoovertowernight.jpg')  
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
    parser.add_argument('-wd', type=float,action="store", dest='wd', default=1e-9)
    parser.add_argument('-epoch', type=int,action="store", dest='epoch', default=2)
    parser.add_argument('-style', type=int,action="store", dest='style', default=0)
    parser.add_argument('-alpha', type=float,action="store", dest='alpha', default=10)
    parser.add_argument('-alphatv', type=float,action="store", dest='alphatv', default=1)
    parser.add_argument('-batch', type=int,action="store", dest='batch', default=3)
    parser.add_argument('-norm', type=int,action="store", dest='norm', default=0)
    parser.add_argument('-t', type=int,action="store", dest='t', default=0)
    args = parser.parse_args()

    main(args.learningRate,args.wd,args.epoch,args.style,args.alpha,args.alphatv,args.batch,args.t,args.norm)


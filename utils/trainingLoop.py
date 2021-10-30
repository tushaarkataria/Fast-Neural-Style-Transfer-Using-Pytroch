import torch
import torch.nn as nn
import albumentations as A
import albumentations.augmentations.functional as F
from albumentations.pytorch import ToTensorV2
from .loss_computation import *
from .htmlutils import *
from collections import OrderedDict
from skimage import io
import copy
dtype = torch.float32
cpu = torch.device('cuda')


def trainingLoop(loader, styleImage,model,optimizer,nepochs,alpha,alphatv,directoryName,webpage,SampleTransform, batchSize,vggmodel):
    ## VGG model to GPU
    vggmodel = vggmodel.to(device=cpu,dtype=dtype)
    vggmodel.eval()

    # Style Image Gram Matrix Precalculations
    styleImage = styleImage/np.max(styleImage) 
    styleImage = SampleTransform(image=styleImage)
    styleImage = styleImage['image'] 
    C, H, W    = styleImage.size()
    styleImage = styleImage.unsqueeze(0)
    
    styleImage = styleImage.expand([batchSize,C,H,W]) ## batchSize * Channels * height * width
    styleImage = styleImage.to(device=cpu,dtype=dtype)
    
    ## Style Image Activations and Gram Matrix
    styleActivations = vggmodel(styleImage)
    StyleGramMatrix  = gram_matrix(styleActivations)

    ## MSE loss and weights for different losses
    contentLoss = nn.MSELoss()
    k=0
    fullLossArray=[]
    contentLossArray =[]
    styleLossArray =[]
    tvLossArray =[]
    for e in range(nepochs):
        for temp in loader:
            ## Sending data to GPU
            temp = temp.to(device=cpu,dtype=dtype)
            
            ## Setting model and optimizer for training
            optimizer.zero_grad(set_to_none=True)
            model = model.to(device=cpu)    
            model.train()  

            ## Forward pass
            output  = model(temp)

            ## VGG features for output and input
            outputActivations = vggmodel(output)
            inputActivations = vggmodel(temp)
   
            ## Gram Matrix of the output batch 
            outputGramMatrix = gram_matrix(outputActivations) 
            
            style_loss = 0           
            for keys in outputGramMatrix.keys():
                outputGramMatrix[keys] = outputGramMatrix[keys].to(device=cpu,dtype=dtype)
                StyleGramMatrix[keys]  = StyleGramMatrix[keys].to(device=cpu,dtype=dtype)
                style_loss += torch.norm(outputGramMatrix[keys]-StyleGramMatrix[keys],p='fro')**2 
            
            content_loss = contentLoss(inputActivations['relu2_2'], outputActivations['relu2_2'])

            tvloss       =  total_variation_loss(output)
            loss   = content_loss + alpha*style_loss +  alphatv * tvloss 
            loss.backward(retain_graph=True)
            optimizer.step()
            if(k%100==0 and k>0):
                contentLossArray.append(content_loss.item())
                styleLossArray.append(style_loss.item())
                fullLossArray.append(loss.item())
                tvLossArray.append(tvloss.item())
                best_model = copy.deepcopy(model.state_dict())
                torch.save(best_model,directoryName+'/best_model_batch'+str(k)+'.pt')
                print("Full loss: ",loss.item())
                contentImage = io.imread('sampleImages/chicago.jpg')  
                contentImage = contentImage/np.max(contentImage) 
                contentImage = SampleTransform(image=contentImage)
                contentImage = contentImage['image'] 
                contentImage = contentImage.unsqueeze(0) 
                contentImage = contentImage.to('cuda')
                output = model(contentImage)
                output = output.squeeze()
                output = output.permute(1,2,0)
                output = output.squeeze().detach().cpu().numpy()
                io.imsave(directoryName+'/images/Chicago_output'+str(k)+'.png',output)
                contentImage = io.imread('sampleImages/hoovertowernight.jpg')  
                contentImage = contentImage/np.max(contentImage) 
                contentImage = SampleTransform(image=contentImage)
                contentImage = contentImage['image'] 
                contentImage = contentImage.unsqueeze(0) 
                contentImage = contentImage.to('cuda')
                output = model(contentImage)
                output = output.squeeze()
                output = output.permute(1,2,0)
                output = output.squeeze().detach().cpu().numpy()
                io.imsave(directoryName+'/images/hoovertower_output'+str(k)+'.png',output)
                ## Writing to Webpage
                visuals = OrderedDict()
                visuals={'chicago'+str(k):'Chicago_output'+str(k)+'.png','hoover'+str(k):'hoovertower_output'+str(k)+'.png'}
                save_images(webpage, visuals, directoryName+'/images', aspect_ratio=1,width=512)
                webpage.save()
            k=k+1
    best_model = copy.deepcopy(model.state_dict())
    torch.save(best_model,directoryName+'/best_model_batch.pt')
    fig, ax = plt.subplots(2,2)
    fig.set_size_inches(28,20)
    ax[0][0].plot(fullLossArray)
    ax[0][0].set_title("Total Loss")
    ax[0][1].plot(styleLossArray)
    ax[0][1].set_title("Style Loss")
    ax[1][0].plot(contentLossArray)
    ax[1][0].set_title("content Loss")
    ax[1][1].plot(tvLossArray)
    ax[1][1].set_title("TV Loss")
    plt.savefig(directoryName+"/FullLoss.png")    
    return loss


import torch
import torch.nn as nn

def total_variation_loss(img):
     bs_img, c_img, h_img, w_img = img.size()
     tv_h = torch.pow(img[:,:,1:,:]-img[:,:,:-1,:], 2).sum()
     tv_w = torch.pow(img[:,:,:,1:]-img[:,:,:,:-1], 2).sum()
     return (tv_h+tv_w)/(bs_img*c_img*h_img*w_img)

def gram_matrix(inputDict):
    GramMatrix ={}
    for key in inputDict.keys():
        input = inputDict[key]
        a,b,c,d = input.size()
        features = input.view(a,b,c*d)
        features_t = features.transpose(1,2) 
        G = torch.bmm(features, features_t)
        G = G.div(a*b*c*d)
        GramMatrix[key] = G
    return GramMatrix

def gram_matrix_1(inputDict):
    GramMatrix ={}
    for key in inputDict.keys():
        input = inputDict[key]
        a,b,c,d = input.size()
        G1 = torch.empty((a,b,b))
        for i in range(a):
            input1 = input[i,:,:,:]
            input1 = input1.unsqueeze(0)
            a1,b1,c1,d1 = input1.size()
            features = input1.view(a1*b1,c1*d1)
            G = torch.mm(features, features.t())
            G=G.div(a1*b1*c1*d1)
            G1[i,:,:] =G
        GramMatrix[key] = G
    return GramMatrix


def gram_matrix1(input,batchSize):
    a,b,c,d = input.size()
    G1 = torch.empty((a,b,b))
    for i in range(a):
        input1 = input[i,:,:,:]
        input1 = input1.unsqueeze(0)
        a1,b1,c1,d1 = input1.size()
        features = input1.view(a1*b1,c1*d1)
        G = torch.mm(features, features.t())
        G=G.div(a1*b1*c1*d1)
        G1[i,:,:] =G
    return G1



import torch
import torch.nn as nn
import numpy as np
class ResNetModule(nn.Module):
    def __init__(self,inCh,outCh,norm='inst'):
        super(ResNetModule,self).__init__()
        if(norm=='batch'):
            self.layer = nn.Sequential(nn.Conv2d(inCh, outCh,3),
                                       nn.BatchNorm2d(outCh),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(inCh, outCh,3),
                                       nn.BatchNorm2d(outCh))
        elif(norm=='inst'):
            self.layer = nn.Sequential(nn.Conv2d(inCh, outCh,3),
                                       nn.InstanceNorm2d(outCh),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(inCh, outCh,3),
                                       nn.InstanceNorm2d(outCh))

    def forward(self,X):
        out = self.layer(X)
        a, b, c, d = X.size()
        X = X[:,:,2:c-2,2:d-2]
        output = out+X
        return output

class ConvModule(nn.Module):
    def __init__(self,inCh,outCh,kernel_size,stride,norm='inst'):
        super(ConvModule,self).__init__()
        if(norm=='batch'):
            self.layer = nn.Sequential(nn.Conv2d(inCh, outCh,kernel_size,stride=stride,padding=np.int(kernel_size/2)),
                                      nn.BatchNorm2d(outCh),
                                      nn.ReLU())
        elif(norm=='inst'):
            self.layer = nn.Sequential(nn.Conv2d(inCh, outCh,kernel_size,stride=stride,padding=np.int(kernel_size/2)),
                                      nn.InstanceNorm2d(outCh),
                                      nn.ReLU())


    def forward(self,X):
        out = self.layer(X)
        return out

class ConvTransPoseModule(nn.Module):
    def __init__(self,inCh,outCh,kernel_size,stride,norm='inst'):
        super(ConvTransPoseModule,self).__init__()
        if(norm=='batch'):
            self.layer = nn.Sequential(nn.ConvTranspose2d(inCh, outCh,kernel_size,stride=stride,padding=np.int(kernel_size/2),output_padding=np.int(kernel_size/2)),
                                  nn.BatchNorm2d(outCh),
                                  nn.ReLU())
        elif(norm=='inst'):
            self.layer = nn.Sequential(nn.ConvTranspose2d(inCh, outCh,kernel_size,stride=stride,padding=np.int(kernel_size/2),output_padding=np.int(kernel_size/2)),
                                  nn.InstanceNorm2d(outCh),
                                  nn.ReLU())


    def forward(self,X):
        out = self.layer(X)
        return out



class ModelStyleBatch(nn.Module):
    def __init__(self,isTanh=False):
        super(ModelStyleBatch,self).__init__()
        self.pad    = nn.ReflectionPad2d(40)
        self.layer1 = ConvModule(3 ,32 ,9,1,norm='batch') # same size as input 
        self.layer2 = ConvModule(32,64 ,3,2,norm='batch') # size/2
        self.layer3 = ConvModule(64,128,3,2,norm='batch') # size/4
        self.RB1    = ResNetModule(128,128,norm='batch')  # -4
        self.RB2    = ResNetModule(128,128,norm='batch')  # -4
        self.RB3    = ResNetModule(128,128,norm='batch')  # -4
        self.RB4    = ResNetModule(128,128,norm='batch')  # -4
        self.RB5    = ResNetModule(128,128,norm='batch')  # -4
        self.CTM1   = ConvTransPoseModule(128,64,3,stride=2,norm='batch') #*2
        self.CTM2   = ConvTransPoseModule(64 ,32,3,stride=2,norm='batch')  # *4
        if(isTanh==True):
            self.conv   = nn.Sequential(nn.Conv2d(32, 3,9,stride=1,padding=4),nn.Tanh())
        else:
            self.conv   = nn.Sequential(nn.Conv2d(32, 3,9,stride=1,padding=4))
                   
    def forward(self, X):
        X    = self.pad(X)
        out1 = self.layer1(X)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.RB1(out3)
        out5 = self.RB2(out4)
        out6 = self.RB3(out5)
        out7 = self.RB4(out6)
        out7 = self.RB5(out7)
        out8 = self.CTM1(out7)
        out9 = self.CTM2(out8)
        out  = self.conv(out9)
        return out




class ModelStyleInstance(nn.Module):
    def __init__(self,isTanh=False):
        super(ModelStyleInstance,self).__init__()
        self.pad    = nn.ReflectionPad2d(40)
        self.layer1 = ConvModule(3,32,9,1) # same size as input 
        self.layer2 = ConvModule(32,64,3,2) # size/2
        self.layer3 = ConvModule(64,128,3,2) # size/4
        self.RB1    = ResNetModule(128,128)  # -4
        self.RB2    = ResNetModule(128,128)  # -4
        self.RB3    = ResNetModule(128,128)  # -4
        self.RB4    = ResNetModule(128,128)  # -4
        self.RB5    = ResNetModule(128,128)  # -4
        self.CTM1   = ConvTransPoseModule(128,64,3,stride=2) #*2
        self.CTM2   = ConvTransPoseModule(64,32,3,stride=2)  # *4
        self.conv   = nn.Sequential(nn.Conv2d(32, 3,9,stride=1,padding=4))
        if(isTanh==True):
            self.conv   = nn.Sequential(nn.Conv2d(32, 3,9,stride=1,padding=4),nn.Tanh())
        else:
            self.conv   = nn.Sequential(nn.Conv2d(32, 3,9,stride=1,padding=4))
                                            
    def forward(self, X):
        X    = self.pad(X)
        out1 = self.layer1(X)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.RB1(out3)
        out5 = self.RB2(out4)
        out6 = self.RB3(out5)
        out7 = self.RB4(out6)
        out7 = self.RB5(out7)
        out8 = self.CTM1(out7)
        out9 = self.CTM2(out8)
        out  = self.conv(out9)
        return out


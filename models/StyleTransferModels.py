import torch
import torch.nn as nn

class ResNetModule(nn.Module):
    def __init__(self,inCh,outCh):
        super(ResNetModule,self).__init__()
        self.layer = nn.Sequential(nn.Conv2d(inCh, outCh,3),
                                   nn.BatchNorm2d(outCh),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(inCh, outCh,3),
                                   nn.BatchNorm2d(outCh))

    def forward(self,X):
        out = self.layer(X)
        a, b, c, d = X.size()
        X = X[:,:,2:c-2,2:d-2]
        output = out+X
        return output

class ConvModule(nn.Module):
    def __init__(self,inCh,outCh,kernel_size,stride):
        super(ConvModule,self).__init__()
        self.layer = nn.Sequential(nn.Conv2d(inCh, outCh,kernel_size,stride=stride,padding=np.int(kernel_size/2)),
                                  nn.BatchNorm2d(outCh),
                                  nn.ReLU())


    def forward(self,X):
        out = self.layer(X)
        return out

class ConvTransPoseModule(nn.Module):
    def __init__(self,inCh,outCh,kernel_size,stride):
        super(ConvTransPoseModule,self).__init__()
        self.layer = nn.Sequential(nn.ConvTranspose2d(inCh, outCh,kernel_size,stride=stride,padding=np.int(kernel_size/2),output_padding=np.int(kernel_size/2)),
                                  nn.BatchNorm2d(outCh),
                                  nn.ReLU())


    def forward(self,X):
        out = self.layer(X)
        return out



class ModelStyleBatch(nn.Module):
    def __init__(self):
        super(ModelStyleBatch,self).__init__()
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

class ResNetModuleInstance(nn.Module):
    def __init__(self,inCh,outCh):
        super(ResNetModuleInstance,self).__init__()
        self.layer = nn.Sequential(nn.Conv2d(inCh, outCh,3),
                                   nn.LayerNorm(outCh),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(inCh, outCh,3),
                                   nn.LayerNorm(outCh))

    def forward(self,X):
        out = self.layer(X)
        a, b, c, d = X.size()
        X = X[:,:,2:c-2,2:d-2]
        output = out+X
        return output

class ConvModuleInstance(nn.Module):
    def __init__(self,inCh,outCh,kernel_size,stride):
        super(ConvModuleInstance,self).__init__()
        self.layer = nn.Sequential(nn.Conv2d(inCh, outCh,kernel_size,stride=stride,padding=np.int(kernel_size/2)),
                                  nn.LayerNorm(outCh),
                                  nn.ReLU())


    def forward(self,X):
        out = self.layer(X)
        return out

class ConvTransPoseModuleInstance(nn.Module):
    def __init__(self,inCh,outCh,kernel_size,stride):
        super(ConvTransPoseModuleInstance,self).__init__()
        self.layer = nn.Sequential(nn.ConvTranspose2d(inCh, outCh,kernel_size,stride=stride,padding=np.int(kernel_size/2),output_padding=np.int(kernel_size/2)),
                                  nn.LayerNorm(outCh),
                                  nn.ReLU())


    def forward(self,X):
        out = self.layer(X)
        return out



class ModelStyleInstance(nn.Module):
    def __init__(self):
        super(ModelStyleInstance,self).__init__()
        self.pad    = nn.ReflectionPad2d(40)
        self.layer1 = ConvModuleInstance(3,32,9,1) # same size as input 
        self.layer2 = ConvModuleInstance(32,64,3,2) # size/2
        self.layer3 = ConvModuleInstance(64,128,3,2) # size/4
        self.RB1    = ResNetModuleInstance(128,128)  # -4
        self.RB2    = ResNetModuleInstance(128,128)  # -4
        self.RB3    = ResNetModuleInstance(128,128)  # -4
        self.RB4    = ResNetModuleInstance(128,128)  # -4
        self.RB5    = ResNetModuleInstance(128,128)  # -4
        self.CTM1   = ConvTransPoseModuleInstance(128,64,3,stride=2) #*2
        self.CTM2   = ConvTransPoseModuleInstance(64,32,3,stride=2)  # *4
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


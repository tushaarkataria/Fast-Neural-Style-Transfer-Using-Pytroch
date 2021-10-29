import torch
import torch.nn as nn
import torchvision.models as models

class VGGModule(nn.Module):
    def __init__(self):
        super(VGGModule,self).__init__()
        vgg = models.vgg16(pretrained=True)
        self.relu1_2 =  vgg.features[0:4]
        self.relu2_2 =  vgg.features[0:9]
        self.relu3_3 =  vgg.features[0:16]
        self.relu4_3 =  vgg.features[0:24]

    def forward(self,X):
        relu1_2 = self.relu1_2(X)
        relu2_2 = self.relu2_2(X)
        relu3_3 = self.relu3_3(X)
        relu4_4 = self.relu4_3(X)
        return relu1_2, relu2_2, relu3_3, relu4_4

class VGGModule19(nn.Module):
    def __init__(self):
        super(VGGModule,self).__init__()
        vgg = models.vgg16(pretrained=True)
        self.relu1_2 =  vgg.features[0:4]
        self.relu2_2 =  vgg.features[0:9]
        self.relu3_3 =  vgg.features[0:16]
        self.relu4_3 =  vgg.features[0:24]

    def forward(self,X):
        relu1_2 = self.relu1_2(X)
        relu2_2 = self.relu2_2(X)
        relu3_3 = self.relu3_3(X)
        relu4_4 = self.relu4_3(X)
        return relu1_2, relu2_2, relu3_3, relu4_4



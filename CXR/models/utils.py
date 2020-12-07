import torch
import torch.nn as nn
from torchvision import models

from toolz         import *
from toolz.curried import *

def loadEncoder(encoderName, pretrained = True):
    
    def AlexNet(pretrained = True):
        return nn.Sequential(nn.Conv2d(1,3,1),
                             models.alexnet(pretrained=pretrained).features)

    def VGGNet(pretrained = True):
        return nn.Sequential(nn.Conv2d(1,3,1),
                             models.vgg16(pretrained=pretrained).features)
    
    def MobileNet(pretrained = True):
        return nn.Sequential(nn.Conv2d(1,3,1),
                             models.mobilenet_v2(pretrained=pretrained).features)
    
    def SqueezeNet(pretrained = True):
        return nn.Sequential(nn.Conv2d(1,3,1),
                             models.squeezenet1_0(pretrained=pretrained).features)

    
    return {"AlexNet"    : lambda : AlexNet    (pretrained=pretrained),
            "VGGNet"     : lambda : VGGNet     (pretrained=pretrained),
            "MobileNet"  : lambda : MobileNet  (pretrained=pretrained),
            "SqueezeNet" : lambda : SqueezeNet (pretrained=pretrained)}[encoderName]()

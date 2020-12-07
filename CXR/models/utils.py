import torch
import torch.nn as nn
from torchvision import models

from toolz         import *
from toolz.curried import *

def loadCNN(what, pretrained = True):
    
    """
    what       :: one of ["AlexNet", "VggNet", "mobileNet", "SqueezeNet"]    
    pretrained :: BOOL
    """
        
    # why add lambda? becuase I only want to load one of many.
    model = \
        {"AlexNet"    : lambda : nn.Sequential(models.alexnet(pretrained=pretrained).features),
         "VggNet"     : lambda : nn.Sequential(models.vgg16(pretrained=pretrained).features),
         "MobileNet"  : lambda : nn.Sequential(models.mobilenet_v2(pretrained=pretrained).features),
         "SqueezeNet" : lambda : nn.Sequential(models.squeezenet1_0(pretrained=pretrained).features),
        }[what]()
    
    featureN = {"AlexNet"    : 256 ,
                "VGGNet"     : 512,
                "MobileNet"  : 1280,
                "SqueezeNet" : 512}[what] 

        
    # the models above are all for RGB image.
    # so I add one 1X1 conv2d layer to map to channel size of 3.
    model = nn.Sequential(nn.Conv2d(1,3,1), nn.ReLU(),
                          model,
                          nn.Conv2d(featureN, 256, 1), nn.ReLU())
    
    print(f"loading {what}... pretrained : {str(pretrained)} ")
    
    return model

def loadAR(what, infeatureN, outFeatureN, unitN):
    
    """
    what        :: one of ["LSTM", "GRU", "RNN"]    
    infeatureN  :: size of input features
    outFeatureN :: size of output features
    unitN       :: number of ar computation units. (i think one is fine....)
    """
    
    # why add lambda? becuase I only want to load one of many.    
    model = \
        {"LSTM" : lambda : nn.LSTM(infeatureN, outFeatureN, unitN),
         "GRU"  : lambda : nn.GRU(infeatureN, outFeatureN, unitN),
         "RNN"  : lambda : nn.RNN(infeatureN, outFeatureN, unitN)
        }[what]()
    
    model = nn.Sequential(model)
    
    print(f"loading {what}... ")
    
    return model
    
    
    
    






import sys,os,re

from glob import glob 

import numpy as np

from toolz         import *
from toolz.curried import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from utils import loadEncoder


class model1(nn.Module):
    def __init__(self, config):
        super(model1, self).__init__()
                
        # main nets
        self.cnn  = loadEncoder(config.encoder, pretrained = False)
        self.lstm = nn.LSTM(256 + 14, 256, 1)
        
        neurons    = {"AlexNet"    : 256 ,
                      "VGGNet"     : 512,
                      "MobileNet"  : 1280,
                      "SqueezeNet" : 512}
                
        self.fc1      = nn.Linear(neurons[config.encoder], 256)        
        self.fc2      = nn.Linear(256, 1)
        
        self.avgPool  = nn.AdaptiveAvgPool2d(1)
        self.dropout  = nn.Dropout(p=0.5)

    def forward(self,batch):
        
        """            
        imgss  :: tensor(B,N,C,H,W)
        diagss :: tensor(B,N,14) we got 14 different diags
        ys     :: tensor(B)
        
        # to be done in future,
            * add relative time as a feature.
        """
                        
        imgss, diagss, ys = batch        
        B,N,C,H,W = imgss.shape
        imgss   = imgss.view(B*N,C,H,W)        
        
        # embed imageeeee yeahhhhhhhhhhh
        embedss = compose(lambda x : x.view(B,N,-1),
                          self.fc1,
                          lambda x : x.squeeze(),
                          self.avgPool,
                          self.cnn)(imgss)
        
        # cat to emb and diag yeahhhhhhhhhhh
        embDiagss = torch.cat([embedss.permute(1,0,-1),
                               diagss.permute(1,0,-1)],
                              dim = -1)
        
        #lstm yeahhhhhhhhhhh
        os, _ = self.lstm(embDiagss)
                        
        #logit yeahhhhhhhhhhh
        logit = compose(self.dropout,
                        self.fc2,
                        last)(os)
        
        return logit
    
if __name__ == "__main__":
    
    sys.path.append("..")
        
    from data.dataGen import dataGen
    from data.dataLoader import collateFn, augument
    from easydict import EasyDict
        
    config = EasyDict()
    config.pklPath = "../data/joined.pkl"
    
    config.L = 72 # hours
    config.W = 24 # hours
    config.dataPath = "../data" #"./data"

    config.batchSize  = 4
    config.numWorkers = 16    
    
    config.encoder = "AlexNet"
        
    gen = dataGen(config, augument)
    
    batch = collateFn([gen.__getitem__(i) for i in range(6)])
    
    net = model1(config)
    
    net(batch)
        

        
        
        
        
        
        


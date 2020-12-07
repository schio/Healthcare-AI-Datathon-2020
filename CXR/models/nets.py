import sys,os,re
import numpy as np

from toolz         import *
from toolz.curried import *

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from utils import loadCNN, loadAR

class baseLineNet(nn.Module):
    def __init__(self, config):
        super(baseLineNet, self).__init__()
                        
        self.cnn = loadCNN(config.cnn, pretrained = False)
        
        self.ar = loadAR(config.ar, 256 + 14, 256, 1)
                        
        self.fc = nn.Sequential(nn.Linear(256, 1),
                                nn.ReLU())
        
        self.avgPool  = nn.AdaptiveAvgPool2d(1)
        
        self.dropout  = nn.Dropout(p=0.5)
        
        print("loading baseLineNet ...")

    def forward(self,batch):
        
        """            
        imgss  :: tensor(B,N,C,H,W)
        diagss :: tensor(B,N,14) we got 14 different diags
        ys     :: tensor(B)
        
        where
            B : batchSize
            N : # of time stamps 
            C : channel size
            H : Height
            W : width
        
        # to be done in future,
            * add relative time as a feature.
        """
                        
        imgss, diagss, ys = batch        
        B,N,C,H,W = imgss.shape
        imgss   = imgss.view(B*N,C,H,W)        
        
        # embed imageeeee yeahhhhhhhhhhh
        embedss = compose(lambda x : x.view(B,N,-1),
                          lambda x : x.squeeze(),
                          self.avgPool,
                          self.cnn)(imgss)
        
        # cat to emb and diag yeahhhhhhhhhhh
        emb = torch.cat([embedss.permute(1,0,-1),
                         diagss.permute(1,0,-1)],
                        dim = -1)
        
        #auto regress yeahhhhhhhhhhh
        os, _ = self.ar(emb)
                        
        #logit yeahhhhhhhhhhh
        logit = compose(self.dropout,
                        self.fc,
                        last)(os)
        
        return logit
    
if __name__ == "__main__":
    
    sys.path.append("..")
        
    from data.dataGen import dataGen
    from data.dataLoader import collateFn, augument
    from easydict import EasyDict
        
    config = EasyDict()
    config.pklPath = "../data/metaData/joined.pkl"
    
    config.L = 72 # hours
    config.W = 24 # hours
    config.dataPath = "../data" #"./data"

    config.batchSize  = 4
    config.numWorkers = 16    
    
    config.cnn = "AlexNet"
    config.ar  = "LSTM"
        
    gen = dataGen(config, augument)
    
    batch = collateFn([gen.__getitem__(i) for i in range(6)])
    
    net = baseLineNet(config)
    
    net(batch)
        

        
        
        
        
        
        


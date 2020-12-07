import os, sys, random
import numpy as np

from toolz import *
from toolz.curried import *

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision.transforms as transforms

from .dataGen import dataGen

def makeDataLoader(config, flag):    
    
    return DataLoader(dataGen(config, augument),
                      batch_size  = config.batchSize,
                      shuffle     = True if flag == 'train' else False,
                      num_workers = config.numWorkers,
                      collate_fn  = collateFn)


@curry
def collateFn(batch):
    
    """
    this function is added as an argument to the makeDataLoader function.
    this function handles how the datapoints for every batch are merged.
    this is required for our case becuase input has variable sizes.
    A remedy to this is to padd zero to short ones.
    
    batch :: [ imgs, diags, y ]
    imgs  :: [np.array(H,W)]
    diags :: [np.array([int])]
    y     :: int    
    """
    
    batchSize = len(batch)
    
    N     = max( map(compose(len, first) )( batch) ) 
    C,H,W = batch[0][0][0].shape
    nDiag = len(batch[0][1][1])
                       
    imgss   = torch.zeros(batchSize,N,C,H,W)
    diagss  = torch.zeros(batchSize,N,nDiag)
    ys      = torch.zeros(batchSize)
    
    for b, (imgs, diags, y) in enumerate(batch):
        
        for i, (img, diag) in enumerate(zip(imgs,diags)):
            imgss[b,i,...] = img
            diagss[b,i,...] = torch.tensor(diag)
            
        ys[b] = y
        
    return torch.tensor(imgss), torch.tensor(diagss), torch.tensor(ys)
    
    
               
@curry
def augument (flag, images) :
               
    """
    flag   : one of ["train","test","validate"]
    images : [np.array(H,W)]
    
    it is a simple augmentation fucntion that ,I admit, is imperfect.
    One key major drawback is that I resize the image to a much smaller one.
    However, following the works done by google. this does not give a bad performance.
    https://github.com/GoogleCloudPlatform/healthcare/tree/master/datathon/datathon_etl_pipelines
    """
    
    if flag == "train":
        return \
            transforms.Compose(
                [transforms.ToPILImage(),
                 transforms.ToTensor(),
                 transforms.Resize((320,320)),
                 transforms.Normalize((0), (1)),
                 transforms.RandomRotation((-5,5)),
                ])(images)
               
    else: # ["test","validate"]
        return \
            transforms.Compose(
                [transforms.ToPILImage(),
                 transforms.ToTensor(),
                 transforms.Resize((320,320)),
                 transforms.Normalize((0,0), (1,1)),
                ])(images)


if __name__ == "__main__":
    
    from easydict import EasyDict
        
    config = EasyDict()
    config.pklPath = "./joined.pkl"
    
    config.L = 72 # hours
    config.W = 24 # hours
    config.dataPath = "." #"./data"

    config.batchSize  = 4
    config.numWorkers = 16    
    
    
    gen = dataGen(config, augument)
    
    batch = [gen.__getitem__(i) for i in range(6)]
    
    for x in makeDataLoader(config, "train"):        
        print(x)
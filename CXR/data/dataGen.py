import os, sys, random
import numpy as np
import torch

from toolz import *
from toolz.curried import *

from matplotlib import pyplot as plt

import torch
from torch.utils.data import Dataset

from datetime import datetime, date, time, timedelta

from dateutil.parser import parse as parseDateTime


# try to join the three tables (need to join Y table later)

class dataGen(Dataset):
    def __init__(self, config, 
                 augument    = lambda _ : identity,
                 flag        = "train"):
        
        self.config  = config
        self.flag    = flag        
        self.augment = augument(self.flag)
        self.dataPts = compose(list,
                               filter(lambda pt : pt["SPLIT"] == self.flag) ,
                               readPkl)(self.config.pklPath)
    
    def __len__(self): 
        return len(self.dataPts)
    
    def __getitem__(self, i):
        
           
        """
        /------/------ 0|1
            W    L
        
        """
                
        dataPt = self.dataPts[i]
        
        Xs     = dataPt["X"]         
        MVTime = None if dataPt["Y"] == 0  else parseDateTime(dataPt["Y"])
        L = timedelta(hours = self.config.L)
        W = timedelta(hours = self.config.W)
        
        Xs = pipe(Xs,
                   # I want only AP or PA
                   filter(lambda x : x["ViewPosition"] in ["AP","PA","0","",0]),                
                   # I want only the ones within the window size
                   filter(lambda x : x if MVTime == None else MVTime-L < x["studyDatetime"] ),
                   # order by  studyDatetime
                   partial(sorted, key = lambda x : x["studyDatetime"]),
                   list)
        
        paths, studyDatetimes, _, diags = zip(*map(lambda x : x.values())(Xs))
                
        imgs =  map(compose(self.augment,
                            plt.imread,
                            lambda path : f"{self.config.dataPath}/{path}"))(paths)
        
        y = 1 if MVTime is not None else 0

        return list(imgs), list(diags), int(y)

    

#FUNCTIONS
#########

import pickle

def readPkl(pklPath):
    return compose(pickle.load, partial(open, mode = "rb"))(pklPath)


if __name__ == "__main__":
    
    from easydict import EasyDict
    
    config = EasyDict()
    config.pklPath = "./joined.pkl"
    
    config.L = 72 # hours
    config.W = 24 # hours
    config.dataPath = "." #"./data"
    
    xs = readPkl(config.pklPath)
    
    gen = dataGen(config)
    
    
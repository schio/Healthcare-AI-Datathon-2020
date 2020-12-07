"""
Here I specicy config details.
"""

from datetime import datetime
from easydict import EasyDict


config = EasyDict()

config.dataPath   = "./data"    # where raw data is located
config.weightPath = "./weights" # where weights are located 

config.L = 72 # lead time
config.W = 24 # window size

config.batchSize  = 10   # batchsize
config.numWorkers = 16   # preprocessing workers    

config.cnn = ["AlexNet" "VGGNet", "MobileNet", "SqueezeNet"][0] # cnn
config.ar  = ["LSTM", "GRU", "RNN"][0] # autoregressive 

config.instanceName = "_".join([config.encoder,
                                config.ar,
                                str(datetime.today())])


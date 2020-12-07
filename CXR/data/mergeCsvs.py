"""
Here we merge 4 csv files :
    1) mimic-cxr-2.0.0-metadata.csv
    2) mimic-cxr-2.0.0-negbio.csv
    3) mimic-cxr-2.0.0-split.csv
    4) mv.csv    
    
1) mimic-cxr-2.0.0-metadata.csv :
    contains necessary meta information of patients such as view, timestamp, size

2) mimic-cxr-2.0.0-negbio.csv:
    14 diagnosis of each X-ray photo

3) mimic-cxr-2.0.0-split.csv:
    train | test | split

4) mv.csv:
    contains 1 ) binary label which indicate patient's ventilation history. 
             2 ) time when the ventilation was performed. if the label is 0 
                 
"""
import pickle
    
import os, sys, time

import numpy as np

import pandas as pd 

from glob import glob

from toolz import *
from toolz.curried import *

from operator import methodcaller

from datetime import datetime, date, time, timedelta

####################################

def parseDateTime(date, time) :
    
    def parseDate(val) :
        val = str(val)
        return datetime.strptime(val, "%Y%m%d").date()

    def parseTime(val) :
        val = str(val).split(".")[0].zfill(6)
        return datetime.strptime(val, "%H%M%S").time()
    
    return datetime.combine(parseDate(date),
                            parseTime(time))

def parseDiag(*args):
    return args


if __name__ == '__main__':

    parentDir = "CXRdata"
    metaDir   = "metaData"
    
    
    #imagePathDF
    ###############
    imagePaths = [(path.split("/")[-1].split(".")[0],path) 
          for path 
          in glob(f"{parentDir}/*/*/*/*/*jpg")]
    
    imagePathDF = pd.DataFrame(imagePaths,columns=["dicom_id", "path"])

    # some paths
    ###############
    metaPath    = f"{parentDir}/mimic-cxr-2.0.0-metadata.csv"
    bioPath     = f"{parentDir}/mimic-cxr-2.0.0-negbio.csv"
    splitPath   = f"{parentDir}/mimic-cxr-2.0.0-split.csv"
    mvPath      = f"{metaDir}/mv.csv"
    joinedPath  = f"{metaDir}/joined.pkl"
    
    
    meta, bio, split, mv = map(pd.read_csv)([metaPath, bioPath, splitPath, mvPath])     
    
    #meta, split join
    ###############        
    meta_split = \
        reduce(partial(pd.merge,
                       on = ["subject_id", "study_id", "dicom_id"],
                       how = "outer"),[meta, split] )
    
    #meta, split, bio join
    ###############        
    meta_split_bio = \
        reduce(partial(pd.merge,
                       on = ["subject_id", "study_id"],
                       how = "left"), [meta_split, bio] )
    
    
    #meta, split, bio, mv join
    ###############            
    meta_split_bio_mv = \
        reduce(partial(pd.merge,
                       on = ["subject_id"],
                       how = "left"), [meta_split_bio, mv] )    
    
    
    #meta, split, bio, mv, imagePathDF join
    final = \
        reduce(partial(pd.merge,
                       on = ["dicom_id"],
                       how = "inner"), [meta_split_bio_mv, imagePathDF] )
    
    
    
    # final touch
    ###############
    final = final.replace(np.nan, 0)
    
    final["studyDatetime"] = (final.apply(lambda row : parseDateTime(row["StudyDate"],
                                                                     row["StudyTime"]),
                                         axis = 1))
    
    final["diag"] = (final.apply(lambda row : parseDiag(row["Atelectasis"],
                                                        row["Cardiomegaly"],
                                                        row["Consolidation"],
                                                        row["Edema"],
                                                        row["Enlarged Cardiomediastinum"],
                                                        row["Fracture"],
                                                        row["Lung Lesion"],
                                                        row["Lung Opacity"],
                                                        row["No Finding"],
                                                        row['Pleural Effusion'],
                                                        row['Pleural Other'],
                                                        row['Pneumonia'],
                                                        row['Pneumothorax'],
                                                        row["Support Devices"]),
                                         axis = 1))
    
    final = final[['subject_id',                   
                   "path", "studyDatetime", 'ViewPosition',                   
                   'diag','MVTime',"split"]]
    
    # group by subject_id and do some process
    ###############    
    XS = []
    for subject_id, df in final.groupby("subject_id") :
        
        Ds = df.to_dict('records')
                
        X     = compose(list, map(keyfilter(lambda x : x in ["path", "studyDatetime", "ViewPosition", "diag"])))(Ds)
        Y     = Ds[0]["MVTime"]
        SPLIT = Ds[0]["split"]
        
        XS.append({"X"     : X,
                   "Y"     : Y,
                   "SPLIT" : SPLIT})
                
    # save
    ###############
    if not os.path.exists(mvPath):
        os.makedirs(mvPath)
        
    f = open(joinedPath,"wb")
    pickle.dump(XS, f)
    f.close()
    
    
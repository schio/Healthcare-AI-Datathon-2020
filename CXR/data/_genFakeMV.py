"""
While the SQL team is extractig the MV information.
Using mimic-cxr-2.0.0-metadata.csv, I create a fake MV info
"""
import os, sys, csv, random

from datetime import datetime, date, time, timedelta

from glob import glob

from toolz import *
from toolz.curried import *

from operator import methodcaller


def readCsv (path):        
    with open(path, "r") as f :
        rows = list(csv.DictReader(f))        
    return list(rows)

def writeCsv (rows, to = "test.csv") -> None:        
    with open(to, 'w') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        for row in rows: writer.writerow(row)
            
def parseDateTime(date, time ) :
    
    def parseDate(val) :
        return datetime.strptime(val, "%Y%m%d").date()

    def parseTime(val) :
        val = val.split(".")[0].zfill(6)
        return datetime.strptime(val, "%H%M%S").time()
    
    return datetime.combine(parseDate(date),
                            parseTime(time))


def genMV (rows, L, W):
    
    maxRow = max(rows, key = get("StudyDateTime") )        
    hours = timedelta(hours = random.randrange(L,L+W))
    
    return {**maxRow, ** {"MVTime" : maxRow["StudyDateTime"] + hours}}
    

if __name__ == '__main__':
    
    
    metaDataPath = "./CXR_data/mimic-cxr-2.0.0-metadata.csv"
    
    # lead time, window size, MV proportion
    L, W, MVP = 24, 24, 0.3 
    
    pipe(metaDataPath, readCsv,
         
         # get "StudyDateTime"(datetime) by merging "StudyDate" & "StudyTime"
         map(lambda row : assoc(row, "StudyDateTime", parseDateTime(row["StudyDate"], row["StudyTime"]))),         
         
         # I dont need ["StudyDate", "StudyTime"]
         map(keyfilter(lambda key : key in ["subject_id", "StudyDateTime"])),         
         
         # create list of list where the len of the list is |patients|         
         groupby("subject_id"), methodcaller("values"),         
         
         # sample a patient if x~unif(0,1) < MVP = 0.3         
         filter(lambda rows : rows if operator.lt(random.random())(MVP)  else None),         
         
         # generate artificial MV
         map(partial(genMV, L = L, W = W)),         
         
         # I want only ["subject_id", "MVTime"]
         map(keyfilter(lambda key : key in ["subject_id", "MVTime"])), list,         
         
         # Donee~~~~~~
         partial(writeCsv, to = "test.csv")
        )

    
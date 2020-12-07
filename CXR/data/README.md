# DATA

This directory contains
- (genFakeMv) : To generate fake MV cases from the MIMIC-CXR Cohort
- (mergeCsvs)   : To merge a set of meta csv files provided into a pickle format (joined.pkl) 
- (dataGen)  : A Dataset class that generates input instances to our model
- (dataLoader) : A DataLoader that generate batch instances 



## Notes

* Before running a model,  please run **mergeCsvs.py** first.
* If you do not yet have the actual MV.csv file, run **genFakeMv.py** first.
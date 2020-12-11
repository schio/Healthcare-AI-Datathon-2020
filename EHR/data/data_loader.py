import pandas as pd
from utils import load_icustays, load_diagnoses, load_lab, split_train_test


feature_list = [load_icustays,
                load_diagnoses,
                load_lab
                ]

features = []
for feature in feature_list:
  features.append(feature())
df = pd.concat([features], axis=1)

train_x, train_y, test_x, test_y = split_train_test(df)

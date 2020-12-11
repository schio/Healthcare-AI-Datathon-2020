import pandas as pd
from utils import load_icustays, load_diagnoses, load_lab, split_data


class DataLoader():
  def __init__(self):
    feature_list = [load_icustays,
                    load_diagnoses,
                    load_lab
                    ]

    features = []
    for feature in feature_list:
      features.append(feature())
    df = pd.concat([features], axis=1)

    self.train_x, self_train_y, self.valid_x, self.valid_y, self.test_x, self.test_y = split_data()

  def get_train(self):
  return self.train_x, self.train_y

  def get_valid(self):
  return self.valid_x, self.valid_y

  def get_test(self):
  return self.test_x, self.test_y

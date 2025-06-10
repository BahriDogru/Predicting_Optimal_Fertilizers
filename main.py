import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.width',500)

# data
train_df = pd.read_csv('datasets/train.csv')
test_df = pd.read_csv('datasets/test.csv')

train_df.head()
test_df.head()
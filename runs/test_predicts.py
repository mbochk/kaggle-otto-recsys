import os
import pandas as pd
from read_data import get_data, do_eval

train_df, test_df, test_labels = get_data()

pp = "/home/mikhail.bochkarev/study/kaggle-otto-recsys/runs/"
for x in os.listdir(pp):
    if x.endswith('.csv'):
        print(x)
        pred = pd.read_csv(pp+x)
        print(do_eval(pred, test_labels))
        print()

import os
import pandas as pd
from pathlib import Path

from sklearn.metrics import log_loss

import random as r
import numpy as np


path = Path('/Users/macbook/UrFU/3 семестр/Автоматизаци mlops/tabular-playground-series-nov-2022 2/')



submission = pd.read_csv(path/'sample_submission.csv', index_col='id')
labels = pd.read_csv(path/'train_labels.csv', index_col='id')

# the ids of the submission rows (useful later)
sub_ids = submission.index

# the ids of the labeled rows (useful later)
gt_ids = labels.index

# list of files in the submission folder
subs = sorted(os.listdir(path/'submission_files'))

df_total = pd.DataFrame()
#Выбираем сто случайных чисел
r_list = r.sample(range(100), 100)  

for i in r_list:
    s0 = pd.read_csv(path / 'submission_files' / subs[i], index_col= 'id')
    df_total = pd.concat([df_total, s0], axis = 1)
    

df = df_total.sum(axis = 1)
df = df/100

score = log_loss(labels, df[:20000])

Logger.current_logger().report_scalar(title='first_test', series='log_loss', value=round(scores[1] * 100, 4), iteration=1)

df = df[20000:]

df = pd.DataFrame(df)
df.columns = ['pred']

df.to_csv('submissions.csv')

!kaggle competitions submit -c tabular-playground-series-nov-2022 -f submissions.csv -m "Submit"



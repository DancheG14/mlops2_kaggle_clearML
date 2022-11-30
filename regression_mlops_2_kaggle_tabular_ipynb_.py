# -*- coding: utf-8 -*-
""""Regression_mlops-2_kaggle-Tabular.ipynb"

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1EwL5se1Td_5el-fx-f0SYHQUfdxFlJlH
"""

import os
import pandas as pd
from pathlib import Path
from sklearn.metrics import log_loss
import random as r
import numpy as np
from google.colab import drive

!pip install clearml
from clearml import Task, Logger

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score as f1
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_regression

from matplotlib import pyplot
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import recall_score, make_scorer
from sklearn.model_selection import learning_curve

drive.mount('/content/drive')
!cp '/content/drive/MyDrive/' kaggle.json

!mkdir -p ~/.kaggle/ && cp /content/drive/MyDrive/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json


!kaggle competitions download -c tabular-playground-series-nov-2022
!unzip tabular-playground-series-nov-2022.zip

!clearml-init

path = Path('./')
task = Task.init(project_name="mlops2", task_name="Daniil")



submission = pd.read_csv(path/'sample_submission.csv', index_col='id')
labels = pd.read_csv(path/'train_labels.csv', index_col='id')

# list of files in the submission folder
subs = sorted(os.listdir(path/'submission_files'))

df_total = pd.DataFrame()
#Выбираем n случайных предсказаний моделей
n = 2000
r_list = r.sample(range(5000), n)  



for i in r_list:
    s0 = pd.read_csv(path / 'submission_files' / subs[i], index_col= 'id')
    df_total = pd.concat([df_total, s0], axis = 1)
    

df = df_total.sum(axis = 1)
df = df/n
df.to_csv('/content/drive/MyDrive/X_total.csv')

df = pd.read_csv('/content/drive/MyDrive/X_total.csv', index_col='id')
print(df)

#посмотрим распределение значений предсказаний моделей:
import seaborn as sns
sns.histplot(df, kde=True)

#ознакомимся с распределением исходных данных и меток
plt.scatter(df[:100], labels[:100])
plt.show()

df

labels.shape

#Разобьём на тестовую и тренировочную выборку и обучим модель линейки

X_train =  df[:20000].values.reshape(-1, 1)
X_test = df[20000:].values.reshape(-1, 1)

y_train = labels

X_train.shape

model = LogisticRegression(solver = 'liblinear', max_iter=2000)
model.fit(X_train, y_train)
# evaluate the model
y_pred = model.predict_proba(X_test)
# evaluate predictions
y_test = y_pred
y_pred

y_train

y_true = np.argmax(y_pred, axis=1)
score = log_loss(y_train, y_test)
print('log_loss metric: ', score)
y_test

mean_squared_error(X_test, y_true)

y_test[::, [1]]

y_true = np.argmax(y_pred, axis=1)
plt.scatter(X_train, y_train)
plt.scatter(X_test, y_true)
plt.show()

y_pred

y_true = np.argmax(y_pred, axis=1)
score = log_loss(y_train, y_true)
print('log_loss metric: ', score)

#Logger.current_logger().report_scalar(title='first_test', series='log_loss', value=round(score, 4), iteration=1)
#task.close()

y_pred = pd.DataFrame(data = y_pred[::, [1]])
y_pred.columns = ['pred']
y_pred.index = y_pred.index + 20000
y_pred
y_pred.to_csv('submissions.csv')


mean_squared_error(X_test, y_pred)


№!kaggle competitions submit -c tabular-playground-series-nov-2022 -f submissions.csv -m "Regression"

X_test.shape

y_pred.shape
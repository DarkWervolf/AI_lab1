import os
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn import svm


def read_data(path, filename):
    return pd.read_csv(os.path.join(path, filename))


def load_dataset(label_dict):
    train_X = read_data('', 'train.csv').values[:,:-2]
    train_y = read_data('', 'train.csv')['Activity']
    train_y = train_y.map(label_dict).values
    test_X = read_data('', 'test.csv').values[:,:-2]
    test_y = read_data('', 'test.csv')
    test_y = test_y['Activity'].map(label_dict).values
    return(train_X, train_y, test_X, test_y)


df = read_data('', 'train.csv')
df.head()

label_dict = {'WALKING':0, 'WALKING_UPSTAIRS':1, 'WALKING_DOWNSTAIRS':2, 'SITTING':3, 'STANDING':4, 'LAYING':5}
train_X, train_y, test_X, test_y = load_dataset(label_dict)

model = svm.SVC(kernel='linear')
#model = KNeighborsClassifier(n_neighbors=8)
model.fit(train_X, train_y)

yhat = model.predict(test_X)
yhat
target_names = ['Walking', 'Walking Upstairs', 'Walking Downstairs', 'Sitting', 'Standing', 'Laying']

print(model.score(train_X, train_y))
print(classification_report(test_y, yhat, target_names=target_names))
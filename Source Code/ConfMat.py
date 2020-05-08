import keras
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from keras import regularizers
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import to_categorical
from privacy.optimizers import dp_optimizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from keras.layers import Flatten
from keras.layers.core import Reshape
from random import uniform
from _thread import *
import numpy as np
import threading
import tensorflow as tf
import pandas as pd
import socket
import pickle
import time
import h5py

from keras.utils import plot_model


FLAGS = tf.flags.FLAGS

#tf.flags.DEFINE_float('learning_rate', 0.07, 'Learning rate for training')
tf.flags.DEFINE_float('noise_multiplier', 1.12, 'Ratio of the standard deviation to the clipping norm')
tf.flags.DEFINE_float('l2_norm_clip', 1.0, 'Clipping norm')
tf.flags.DEFINE_integer('batch_size', 256, 'Batch size')
tf.flags.DEFINE_integer('microbatches', 1, 'Number of microbatches ''(must evenly divide batch_size)')


def dummyEncode(df):
        columnsToEncode = list(df.select_dtypes(include=['category','object']))
        le = LabelEncoder()
        for feature in columnsToEncode:
            try:
                df[feature] = le.fit_transform(df[feature])
            except:
                print('Error encoding '+feature)
        return df

test_data = input("Enter the path to the filename or the filename\n")
test_model = input("Enter the filename of the model")
train_dataframe = pd.read_csv(test_data)
train_dataframe = dummyEncode(train_dataframe)

model = Sequential()
n_cols = 13

e = 'relu'
model.add(Dense(units=n_cols, activation=e,input_shape=(n_cols,)))
model.add(Dense(75, activation=e))
model.add(Dense(50,activation=e))
model.add(Dense (75, activation=e))
model.add(Dense(units=2, activation='softmax'))


model.load_weights(test_model)

#for i in range(test_Y.shape[0]):
#	print("Actual = %s, Predicted = %s " %(test_Y.iloc[i], y[i]))

m, accuracy, precision , total = [],[],[],[]
print('%5s %30s %5s %5s' %('total', 'confusion matrix', 'acc', 'pre'))
for i in range(10):
	td = train_dataframe.sample(frac = uniform(0.05,0.3))
	test_X = td.drop(columns=['change'])
	test_Y = td['change']
	y = model.predict_classes(test_X)
	total.append(test_X.shape[0])
	tn, fp ,fn, tp = confusion_matrix(test_Y, y).ravel()
	m.append([[tn,fp],[fn, tp]])
	accuracy.append(round((tn+tp)/(test_X.shape[0]),3))
	precision.append(round((tp)/(tp+fp),3))
	print('%5i %30s %5s %5s' %(total[i],str(m[i]),str(accuracy[i]),str(precision[i])))
	fpr, tpr, thresholds = roc_curve(test_Y, y)
	a = auc(fpr, tpr)

#ROC curve for the last confusion matrix
plt.plot([0,1],[0,1],'k--')
plt.plot(fpr,tpr, label = 'Area Under Curve = {:3f}'.format(a))
plt.xlabel('false positive rate')
plt.ylabel('true positive rate')
plt.legend(loc='best')
plt.show()

plt.title('ROC curve')
plt.savefig('ROC_model1.png')

#plot_model(model, to_file='model.png')




import keras
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras import regularizers
from keras.utils import to_categorical
from privacy.optimizers import dp_optimizer
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import compute_class_weight
from keras.layers import Flatten
from keras.layers.core import Reshape
from keras.utils import plot_model
import matplotlib.pyplot as plt
from _thread import *
import numpy as np
import threading
from random import uniform
import tensorflow as tf
import pandas as pd
import socket
import pickle
import time
import h5py

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_float('learning_rate', 0.7, 'Learning rate for training')
tf.flags.DEFINE_float('noise_multiplier', 1.12, 'Ratio of the standard deviation to the clipping norm')
tf.flags.DEFINE_float('l2_norm_clip', 1.0, 'Clipping norm')
tf.flags.DEFINE_integer('batch_size', 256, 'Batch size')
tf.flags.DEFINE_integer('microbatches', 1, 'Number of microbatches ''(must evenly divide batch_size)')
o = p = 0

def dummyEncode(df):
        columnsToEncode = list(df.select_dtypes(include=['category','object']))
        le = LabelEncoder()
        for feature in columnsToEncode:
            try:
                df[feature] = le.fit_transform(df[feature])
            except:
                print('Error encoding '+feature)
        return df


train_dataframe = pd.read_csv("dia5.csv")
train_dataframe = dummyEncode(train_dataframe)
train_X = train_dataframe.drop(columns=['change'])
train_Y = to_categorical(train_dataframe['change'])

model = Sequential()
n_cols = train_X.shape[1]

x=0.1
e = 'relu'
model.add(Dense(units=n_cols, activation=e,input_shape=(n_cols,)))
model.add(Dense(75, activation=e))
model.add(Dense(50,activation=e))
model.add(Dense (75, activation=e))
model.add(Dense(units=2, activation='softmax'))

model.compile(loss='categorical_hinge', optimizer=dp_optimizer.DPAdamGaussianOptimizer(l2_norm_clip=FLAGS.l2_norm_clip,
          noise_multiplier=FLAGS.noise_multiplier,
          num_microbatches=FLAGS.microbatches,
          ), metrics=['accuracy'])

csv_logger1 = keras.callbacks.CSVLogger('log1.csv', append = True, separator = ',')
model.load_weights('model1.h5')

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print('socket created')
host=socket.gethostname()
port = 9999
s.bind((host,port))
s.listen(1)
print('waiting for connection')
sc, address = s.accept()

print(" connected with " ,address)
print("  SENDING DATA TO CLIENT B ")


for i in range(3):
  print(str(i+1)+' LEVEL TRAINING')
  if o != 0:
    model.load_weights(filename)
  o = 1


  history=model.fit(train_X, train_Y, callbacks =[csv_logger1], validation_split=0.3,batch_size=1000, epochs=50, shuffle=True)
  filename = 'client1ts.pkl'
  model.save_weights(filename)



  with open(filename,"rb") as f:
    sc.send(b'BEGIN')
    l = f.read(1024)
    while(l):
      sc.send(l)
      l = f.read(1024)
    sc.send(b'ENDED')
  filename = "client1tr.pkl"

  print("DATA TRANSFERRED SUCCESSFULLY TO CLIENT B")
  
  with open(filename,"wb") as f:
    while True:
      data = sc.recv(1024)
      if(data == b'BEGIN'):
        continue
      elif data == b'ENDED':
        break
      else:
        f.write(data)

  print("DATA RECEIVED SUCCESSFULLY FROM CLIENT B")

  # Plot training & validation accuracy values
  plt.plot(history.history['acc'])
  plt.plot(history.history['val_acc'])
  plt.title('Model accuracy')
  plt.ylabel('Accuracy')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Test'], loc='upper left')
  plt.savefig('Model1ACC'+str(i)+'epoch.png')
  plt.clf()
  plt.cla()

  # Plot training & validation loss values
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('Model 1 loss')
  plt.ylabel('Loss')
  plt.xlabel('Epoch')
  plt.legend(['Train', 'Test'], loc='upper left')
  plt.savefig('Model1LOS'+str(i)+'epoch.png')
  plt.clf()
  plt.cla()
  model.save_weights('model1.h5')




model.save_weights('model1.h5')
model.save('model1t.h5')
s.close()

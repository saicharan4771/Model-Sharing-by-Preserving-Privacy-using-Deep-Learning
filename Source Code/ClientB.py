import keras
import time
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from keras.models import Sequential
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras import regularizers
from keras.models import load_model
from keras.layers import Dense
from keras.utils import to_categorical
from privacy.optimizers import dp_optimizer
from sklearn.preprocessing import LabelEncoder
from keras.layers.core import Reshape
from keras.layers import Flatten
from keras.utils import plot_model
import matplotlib.pyplot as plt
from _thread import *
import numpy as np
from random import uniform
import threading
import tensorflow as tf
import pandas as pd
import socket
import pickle
import random
import time
import h5py

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_float('learning_rate', 0.7, 'Learning rate for training')
tf.flags.DEFINE_float('noise_multiplier', 1.12, 'Ratio of the standard deviation to the clipping norm')
tf.flags.DEFINE_float('l2_norm_clip', 1.0, 'Clipping norm')
tf.flags.DEFINE_integer('batch_size', 256, 'Batch size')
tf.flags.DEFINE_integer('microbatches', 1, 'Number of microbatches ''(must evenly divide batch_size)')
print_lock = threading.Lock()

def dummyEncode(df):
        columnsToEncode = list(df.select_dtypes(include=['category','object']))
        le = LabelEncoder()
        for feature in columnsToEncode:
            try:
                df[feature] = le.fit_transform(df[feature])
            except:
                print('Error encoding '+feature)
        return df

train_dataframe = pd.read_csv("kid6.csv")
train_dataframe = dummyEncode(train_dataframe)
train_X = train_dataframe.drop(columns=['change'])
train_Y = to_categorical(train_dataframe['change'])

model = Sequential()
n_cols = train_X.shape[1]
print(n_cols)

x=0.1
e = 'relu'
model.add(Dense(units=n_cols, activation=e,input_shape=(n_cols,)))
model.add(Dense(75, activation=e))
model.add(Dense(50,activation=e))
model.add(Dense (75, activation=e))
model.add(Dense(units=2, activation='softmax'))

model.compile(loss='categorical_hinge', optimizer= dp_optimizer.DPAdamGaussianOptimizer(l2_norm_clip=FLAGS.l2_norm_clip,
          noise_multiplier=FLAGS.noise_multiplier,
          num_microbatches=FLAGS.microbatches,
          ), metrics=['accuracy'])


csv_logger2 = keras.callbacks.CSVLogger('log2.csv', append = True, separator = ',')

model.load_weights('model2.h5')
sc = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print('socket created')
port = 9999
host = socket.gethostname()
print("waiting for connection")
sc.connect((host, port))

print(" connected with ",host)
print(" RECEIVING  DATA FROM CLIENT A")
for i in range(3):
	print(str(i+1)+' LEVEL TRAINING')
	filename = "client2tr.pkl" 
	with open(filename,"wb") as f:
		while True:
			data = sc.recv(1024)
			if(data == b'BEGIN'):
				continue
			elif data == b'ENDED':
				break
			else:
				f.write(data)
        
	
        
	print("DATA RECEIVED SUCCESSFULLY FROM CLIENT A")

	model.load_weights(filename)
	history=model.fit(train_X, train_Y, callbacks =[csv_logger2], validation_split=0.3, batch_size=1000, epochs=50, shuffle=True)

	filename = "client2ts.pkl"
	model.save_weights(filename)
	
	with open(filename,"rb") as f:
		sc.send(b'BEGIN')
		l = f.read(1024)
		while(l):
			sc.send(l)
			l = f.read(1024)
		sc.send(b'ENDED')
        
	print("DATA TRANSFERRED SUCCESSFULLY TO CLIENT A")

	plt.plot(history.history['acc'])
	plt.plot(history.history['val_acc'])
	plt.title('Model 2 accuracy')
	plt.ylabel('Accuracy')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	plt.savefig('Model2ACC'+str(i)+'epoch.png')
	plt.clf()
	plt.cla()
	  

	# Plot training & validation loss values
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.title('Model 2 loss')
	plt.ylabel('Loss')
	plt.xlabel('Epoch')
	plt.legend(['Train', 'Test'], loc='upper left')
	plt.savefig('Model2LOS'+str(i)+'epoch.png')
	plt.clf()
	plt.cla()
	model.save_weights('model2.h5')
	


model.save_weights('model2.h5')
model.save('model2t.h5')
sc.close()

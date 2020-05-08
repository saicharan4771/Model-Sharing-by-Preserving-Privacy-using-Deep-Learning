from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import Reshape
from keras.utils import to_categorical
from privacy.optimizers import dp_optimizer
from sklearn.preprocessing import LabelEncoder
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import tensorflow as tf
import pandas as pd
import socket
import pickle
from IPython.display import SVG
from keras.utils.vis_utils import plot_model


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_float('learning_rate', 0.7, 'Learning rate for training')
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

dataset = input("Enter the filepath of the dataset\n")
model_name = input("Enter the name of the model with .h5 extension\n")
train_dataframe = pd.read_csv(dataset)
train_dataframe = dummyEncode(train_dataframe)
train_X = train_dataframe.drop(columns=['change'])
train_Y = to_categorical(train_dataframe['change'])

model = Sequential()
n_cols = train_X.shape[1]

model = Sequential()
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

model.save_weights(model_name)
print(model.summary())

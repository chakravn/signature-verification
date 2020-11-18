import os
import shutil
import random
import math
import numpy as np
import pandas as pd
import pickle
#import PIL
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow import keras 
import keras.backend as K
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Flatten, Dense, Concatenate, Dot, Lambda, Input
from tensorflow.keras.layers import Conv1D, Conv2D, Dropout, BatchNormalization, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from keras.regularizers import l2
import argparse


execute_eagerly = False # Change this to false to disable eager execution

if not execute_eagerly:
  tf.compat.v1.disable_eager_execution()

#print(tf.__version__)
#print(keras.__version__)

def initialize_weights(shape, dtype=None):
  return np.random.normal(loc = 0.0, scale = 1e-2, size = shape)

def initialize_bias(shape, dtype=None):
  return np.random.normal(loc = 0.5, scale = 1e-2, size = shape)

def siamese_nn(input_shape):
    """
        Model architecture
    """
    
    # Define the tensors for the two input images
    left_input = Input(input_shape)
    right_input = Input(input_shape)
    
    # Convolutional Neural Network
    model = Sequential()
    model.add(Conv2D(64, (10,10), activation='relu', input_shape=input_shape,
                   kernel_initializer=initialize_weights, 
                   bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (7,7), activation='relu',
                     kernel_initializer=initialize_weights,
                     bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (2,2), activation='relu', kernel_initializer=initialize_weights,
                     bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))
    model.add(MaxPooling2D())
    model.add(Conv2D(256, (2,2), activation='relu', kernel_initializer=initialize_weights,
                     bias_initializer=initialize_bias, kernel_regularizer=l2(2e-4)))
    model.add(Flatten())
    model.add(Dense(4096, activation='sigmoid',
                   kernel_regularizer=l2(1e-3),
                   kernel_initializer=initialize_weights,bias_initializer=initialize_bias))
    
    # Generate the encodings (feature vectors) for the two images
    encoded_l = model(left_input)
    encoded_r = model(right_input)
    
    # Add a customized layer to compute the absolute difference between the encodings
    L1_layer = Lambda(lambda tensors:K.abs(tensors[0] - tensors[1]))
    L1_distance = L1_layer([encoded_l, encoded_r])
    
    # Add a dense layer with a sigmoid unit to generate the similarity score
    prediction = Dense(1,activation='sigmoid',bias_initializer=initialize_bias)(L1_distance)
    
    # Connect the inputs with the outputs
    siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)
    
    # return the model
    return siamese_net
    
    

def read_inp(pkl):
    _temp_ = pickle.load(open(pkl, 'rb'))
    x_train = _temp_['x_train']
    x_valid = _temp_['x_valid']
    y_train = _temp_['y_train']
    y_valid = _temp_['y_valid']
    x_train_1 = x_train[:, 0].reshape(len(x_train[:, 0]), 64, 128,1)
    x_train_2 = x_train[:, 1].reshape(len(x_train[:, 1]), 64, 128,1)
    x_valid_1 = x_valid[:, 0].reshape(len(x_valid[:, 0]), 64, 128,1)
    x_valid_2 = x_valid[:, 1].reshape(len(x_valid[:, 1]), 64, 128,1)
    return x_train_1, x_train_2, x_valid_1, x_valid_2, y_train, y_valid

if __name__ == '__main__':
        
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--loss', type=str, default='binary_crossentropy')
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--training', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    #parser.add_argument('--validation', type=str, default=os.environ['SM_CHANNEL_VALIDATION'])

    args, _ = parser.parse_known_args()
    
    epochs     = args.epochs
    optim      = args.optim
    bs = args.batch_size
    loss       = args.loss
    model_dir  = args.model_dir
    training_dir   = args.training
    #validation_dir = args.validation
    
    #read input 
    pkl = [ os.path.join(training_dir, file) for file in os.listdir(training_dir) ][0]
    print(pkl)
    x_train_1, x_train_2, x_valid_1, x_valid_2, y_train, y_valid = read_inp(pkl)

    input_shape = (64, 128, 1)
    model = siamese_nn(input_shape)
    model.summary()
    optimizer = Adam(lr = 0.00005)
    model.compile(loss=loss,optimizer=optimizer, metrics=['accuracy'])
    model.fit([x_train_1, x_train_2], y_train, 
                    batch_size=16, epochs=32, 
                    validation_data=([x_valid_1, x_valid_2], y_valid)
                   )


    version = 'siamese/1'
    export_path = os.path.join(model_dir, str(version))
    if not os.path.isdir(export_path):
        os.makedirs(export_path)
   
    tf.keras.models.save_model(
        model,
        export_path,
        overwrite=True,
        include_optimizer=True,
        save_format=None,
        signatures=None,
        options=None
        )

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 14 17:45:18 2018

@author: kaku
The model is named by model_convnum_densenum
"""

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD

from keras.wrappers.scikit_learn import KerasRegressor

def model_2_1(input_shape):

    model = Sequential()

    model.add(Conv2D(16, (3, 3), activation='relu', input_shape= input_shape))
    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape= input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    # create model
    model.add(Dense(1))
#    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    # compile model
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    return model 

def model_2_2(input_shape):
    """
    Cannot converge when add a dense layer before output 
    """

    model = Sequential()

    model.add(Conv2D(16, (3, 3), activation='relu', input_shape= input_shape))
    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape= input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    # create model
    model.add(Dense(13, activation='relu'))
    model.add(Dense(1))
#    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    # compile model
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
    return model 

def model_3_1(input_shape):

    model = Sequential()

    model.add(Conv2D(10, (3, 3), activation='relu', input_shape= input_shape))
    model.add(Conv2D(10, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16, (3, 3), activation='relu', input_shape= input_shape))
    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape= input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    # create model
#    model.add(Dense(13, activation='relu'))
    model.add(Dense(1))
#    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    # compile model
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
#    model.compile(loss='mean_squared_error', optimizer='adam')

    return model 

def model_4_1(input_shape):

    model = Sequential()

    model.add(Conv2D(10, (3, 3), activation='relu', input_shape= input_shape))
    model.add(Conv2D(10, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(16, (3, 3), activation='relu', input_shape= input_shape))
    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape= input_shape))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape= input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    # create model
#    model.add(Dense(13, activation='relu'))
    model.add(Dense(1))
#    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    # compile model
#    model.compile(loss='mean_squared_error', optimizer='adam')
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    return model 

if __name__ == '__main__':
    print('Hello')
else:
    print('Different models can be loaded')
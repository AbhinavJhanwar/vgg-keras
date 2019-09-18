# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 12:33:54 2019

@author: abhinav.jhanwar
"""

from tensorflow.python.keras.applications import VGG16
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Dense, Flatten, Dropout, AveragePooling2D, BatchNormalization
from tensorflow.python.keras.optimizers import Adam
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

def vgg_16(pretrained_weights=None, input_size=(1024, 150, 3)):
    new_input = Input(shape=input_size)
    
    model = VGG16(include_top=False, input_tensor=new_input, weights=None)

    # Say not to train ResNet model layers as they are already trained
    #for layer in model.layers:
    #	layer.trainable = False
    
    # add custom model as per requirments
    x = model.output
    pool1 = AveragePooling2D(pool_size=(4, 4))(x)
    flat1 = Flatten()(pool1)
    dense1 = Dense(1024, activation="relu")(flat1)
    drop1 = Dropout(0.5)(dense1)
    dense2 = Dense(1024, activation="relu")(drop1)
    drop2 = Dropout(0.5)(dense2)
    dense3 = Dense(1, activation="linear")(drop2)
    
    model = Model(inputs=model.inputs, outputs=dense3)
    
    # Compile Our Transfer Learning Model
    model.compile(loss='mean_squared_error', optimizer=Adam(lr = 1e-4), metrics=['mse', 'mae', 'mape'])
    
    print(model.summary())

    if(pretrained_weights):
        print('Using pretrained weights:', pretrained_weights)
        model.load_weights(pretrained_weights)
        
    return model

def vgg_12(pretrained_weights=None, input_size=(256,256,1)):
    
    new_input = Input(shape=input_size)
    
    #model = VGG16(include_top=False, input_tensor=new_input, weights=None)
    model = VGG16(include_top=False, input_tensor=new_input)
    
    # Say not to train ResNet model layers as they are already trained
    #for layer in model.layers:
    #	layer.trainable = False
    
    # remove last 3 layers of the vgg16 model
    x = model.layers[-5].output
    pool1 = AveragePooling2D(pool_size=(4, 4))(x)
    flat1 = Flatten()(pool1)
    dense1 = Dense(1024, activation="relu")(flat1)
    drop1 = Dropout(0.5)(dense1)
    dense2 = Dense(1, activation="linear")(drop1)
    
    model = Model(inputs=model.inputs, outputs=dense2)
    
    # Compile Our Transfer Learning Model
    model.compile(loss='mean_squared_error', optimizer=Adam(lr = 1e-4), metrics=['mse', 'mae', 'mape'])
    
    print(model.summary())

    if(pretrained_weights):
        print('Using pretrained weights:', pretrained_weights)
        model.load_weights(pretrained_weights)
        
    return model


def vgg_12_bn(pretrained_weights=None, input_size=(256,256,1)):
    
    new_input = Input(shape=input_size)
    
    model = VGG16(include_top=False, input_tensor=new_input, weights=None)
    
    # Say not to train ResNet model layers as they are already trained
    #for layer in model.layers:
    #	layer.trainable = False
    
    # extract first convolutional layer of vgg & apply BN to first two conv layers
    conv1 = model.layers[1].output
    bn1 = BatchNormalization()(conv1)
    conv2 = model.layers[2](bn1)
    bn2 = BatchNormalization()(conv2)
    x = model.layers[3](bn2)
    
   # append other layers of vgg network
    for layer in model.layers[4:15]:
        x = layer(x)
    
    # append final dense layers to model
    pool1 = AveragePooling2D(pool_size=(4, 4))(x)
    flat1 = Flatten()(pool1)
    dense1 = Dense(1024, activation="relu")(flat1)
    drop1 = Dropout(0.5)(dense1)
    dense2 = Dense(1, activation="linear")(drop1)
    
    model = Model(inputs=model.inputs, outputs=dense2)
    
    # Compile Our Transfer Learning Model
    model.compile(loss='mean_squared_error', optimizer=Adam(lr = 1e-4), metrics=['mse', 'mae', 'mape'])
    
    print(model.summary())

    if(pretrained_weights):
        print('Using pretrained weights:', pretrained_weights)
        model.load_weights(pretrained_weights)
        
    return model


def vgg_9(pretrained_weights=None, input_size=(256,256,1)):
    
    new_input = Input(shape=input_size)
    
    model = VGG16(include_top=False, input_tensor=new_input, weights=None)
    
    # Say not to train ResNet model layers as they are already trained
    #for layer in model.layers:
    #	layer.trainable = False
    
    # remove last 3 layers of the vgg16 model
    x = model.layers[-9].output
    pool1 = AveragePooling2D(pool_size=(4, 4))(x)
    flat1 = Flatten()(pool1)
    dense1 = Dense(1024, activation="relu")(flat1)
    drop1 = Dropout(0.5)(dense1)
    dense2 = Dense(1, activation="linear")(drop1)
    
    model = Model(inputs=model.inputs, outputs=dense2)
    
    # Compile Our Transfer Learning Model
    model.compile(loss='mean_squared_error', optimizer=Adam(lr = 1e-4), metrics=['mse', 'mae', 'mape'])
    
    print(model.summary())

    if(pretrained_weights):
        print('Using pretrained weights:', pretrained_weights)
        model.load_weights(pretrained_weights)
        
    return model


def vgg_9_bn(pretrained_weights=None, input_size=(1024,128,3)):
    
    new_input = Input(shape=input_size)
    
    model = VGG16(include_top=False, input_tensor=new_input, weights=None)
    
    # Say not to train ResNet model layers as they are already trained
    #for layer in model.layers:
    #	layer.trainable = False
    
    # extract first convolutional layer of vgg & apply BN to first two conv layers
    conv1 = model.layers[1].output
    bn1 = BatchNormalization()(conv1)
    conv2 = model.layers[2](bn1)
    bn2 = BatchNormalization()(conv2)
    x = model.layers[3](bn2)
    
    # append other layers of vgg network
    for layer in model.layers[4:11]:
        x = layer(x)
        
    # append final dense layers to model
    pool1 = AveragePooling2D(pool_size=(4, 4))(x)
    flat1 = Flatten()(pool1)
    dense1 = Dense(1024, activation="relu")(flat1)
    drop1 = Dropout(0.5)(dense1)
    dense2 = Dense(1, activation="linear")(drop1)
    
    model = Model(inputs=model.inputs, outputs=dense2)
    
    # Compile Our Transfer Learning Model
    model.compile(loss='mean_squared_error', optimizer=Adam(lr = 1e-4), metrics=['mse', 'mae', 'mape'])
    
    print(model.summary())

    if(pretrained_weights):
        print('Using pretrained weights:', pretrained_weights)
        model.load_weights(pretrained_weights)
        
    return model

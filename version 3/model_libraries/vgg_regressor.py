# -*- coding: utf-8 -*-

from tensorflow.python.keras.applications import VGG16
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Input, Dense, Flatten, Dropout, AveragePooling2D, BatchNormalization
from tensorflow.python.keras.optimizers import Adam, SGD
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

def vgg_12(pretrained_weights='None', input_size=(256,256,1), initial_weights=True, lr=1e-4):
    
    new_input = Input(shape=input_size)
    
    if initial_weights:
        model = VGG16(include_top=False, input_tensor=new_input)
    else:
        model = VGG16(include_top=False, input_tensor=new_input, weights=None)
    
    # remove last 3 layers of the vgg16 model
    x = model.layers[-5].output
    pool1 = AveragePooling2D(pool_size=(4, 4))(x)
    flat1 = Flatten()(pool1)
    dense1 = Dense(1024, activation="relu")(flat1)
    drop1 = Dropout(0.5)(dense1)
    dense2 = Dense(1, activation="linear")(drop1)
    
    model = Model(inputs=model.inputs, outputs=dense2)
    
    # Compile Our Transfer Learning Model
    model.compile(loss='mean_squared_error', optimizer=Adam(lr=lr), metrics=['mse', 'mae', 'mape'])
    
    print(model.summary())

    if pretrained_weights!='None':
        print('Using pretrained weights:', pretrained_weights)
        model.load_weights(pretrained_weights)
        
    return model



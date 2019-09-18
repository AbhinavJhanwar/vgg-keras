# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 15:12:47 2019

@author: abhinav.jhanwar
"""

from keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np
from PIL import Image

def trainGenerator(batch_size, train_path, data_frame, aug_dict, image_color_mode="grayscale",
                    mask_color_mode="grayscale", flag_multi_class=False,
                    save_to_dir=None, target_size=(256,256), seed=1):
    '''
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    
    image_generator = image_datagen.flow_from_dataframe(
        data_frame,
        directory = train_path,
        x_col = 'field',
        y_col = 'lai',
        target_size = target_size,
        color_mode = image_color_mode,
        class_mode = 'other',
        batch_size = batch_size,
        seed = seed,
        save_to_dir = save_to_dir
        )
    return image_generator

def validationGenerator(batch_size, train_path, data_frame, aug_dict, image_color_mode="grayscale",
                    mask_color_mode="grayscale", flag_multi_class=False,
                    save_to_dir=None, target_size=(256,256), seed=1):
    '''
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    
    image_generator = image_datagen.flow_from_dataframe(
        data_frame,
        directory = train_path,
        x_col = 'field',
        y_col = 'lai',
        target_size = target_size,
        color_mode = image_color_mode,
        class_mode = 'other',
        batch_size = batch_size,
        seed = seed,
        save_to_dir = save_to_dir,
        shuffle=False
        )
    
    return image_generator


def testGenerator(batch_size, image_path, data_frame, aug_dict, x_col, y_col,
                  image_color_mode="grayscale", save_to_dir=None, target_size=(256,256), seed=1):
    
    image_datagen = ImageDataGenerator(**aug_dict)
    
    image_generator = image_datagen.flow_from_dataframe(
        data_frame,
        directory = image_path,
        x_col = x_col,
        y_col = y_col,
        target_size = target_size,
        color_mode = image_color_mode,
        class_mode = None,
        batch_size = batch_size,
        seed = seed,
        save_to_dir = save_to_dir,
        shuffle = False
        )
    
    return image_generator


def hsv_form(image):
    image = np.array(image)
    # convert to hsv
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    
    '''h,v = hsv_image.shape[:2]
    # filter for h value
    for x in range(h):
        for y in range(v):
            if hsv_image[x,y,0]<44 or hsv_image[x,y,0]>99:
                hsv_image[x,y,0]=0'''
    # rescale
    hsv_image = hsv_image/255
    #comb = np.concatenate([image, hsv_image], axis=2)
    return hsv_image
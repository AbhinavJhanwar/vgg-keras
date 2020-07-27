# -*- coding: utf-8 -*-

from keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np
from PIL import Image

def trainGenerator(batch_size, train_path, data_frame, x_col, y_col, aug_dict, image_color_mode="grayscale",
                    mask_color_mode="grayscale", flag_multi_class=False,
                    save_to_dir=None, target_size=(256,256), seed=1):
    
    image_datagen = ImageDataGenerator(**aug_dict)
    
    image_generator = image_datagen.flow_from_dataframe(
        data_frame,
        directory = train_path,
        x_col = x_col,
        y_col = y_col,
        target_size = target_size,
        color_mode = image_color_mode,
        class_mode = 'other',
        batch_size = batch_size,
        seed = seed,
        save_to_dir = save_to_dir
        )
    return image_generator


def validationGenerator(batch_size, train_path, data_frame, x_col, y_col, aug_dict, image_color_mode="grayscale",
                    mask_color_mode="grayscale", flag_multi_class=False,
                    save_to_dir=None, target_size=(256,256), seed=1):
    
    image_datagen = ImageDataGenerator(**aug_dict)
    
    image_generator = image_datagen.flow_from_dataframe(
        data_frame,
        directory = train_path,
        x_col = x_col,
        y_col = y_col,
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

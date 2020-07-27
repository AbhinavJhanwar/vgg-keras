import os
import shutil
from tqdm import tqdm
import cv2
import gc
import pandas as pd
gc.enable()
import numpy as np
from sklearn import metrics, preprocessing
import pandas as pd
import datetime
import logging
import glob

from model_libraries.vgg_regressor import vgg_12
from model_libraries.vgg_regressor import ModelCheckpoint, EarlyStopping
from model_libraries.dataFlatten import trainGenerator, validationGenerator


class trainVGG:
    def __init__(self, config):
        logging.info('model_libraries.trainVGG.parTrain.init started')
        try:
            # csv file path for actual lai and images names
            self.csv_file = config['input_paths']['csv_file']
            # images folder path
            self.images_dir = config['input_paths']['images_dir']
            
            # directory to save ouput
            self.output_dir = config['output_paths']['output_dir']
            # name of the model to be saved for vgg
            self.model_name = os.path.join(self.output_dir, config['output_paths']['vgg_model_name'])
            # images folder to save reshaped images as per vgg network requirments
            self.vgg_images = config['output_paths']['vgg_images']
            
            # training images directory
            self.train_images_dir = os.path.join(self.output_dir, config['vgg_train_paths']['train_images_dir'])
            # training csv file
            self.train_csv = os.path.join(self.output_dir, config['vgg_train_paths']['train_csv'])
            
            # validation images directory
            self.val_images_dir = os.path.join(self.output_dir, config['vgg_val_paths']['val_images_dir'])
            # validation csv file
            self.val_csv = os.path.join(self.output_dir, config['vgg_val_paths']['val_csv'])
            
            # x_col or column containing image names
            self.x_col = config['vgg_model_params']['x_col']
            # y_col or target column to train vgg model on
            self.y_col = config['vgg_model_params']['y_col']
            # learning rate to train model on
            self.lr = config['vgg_model_params']['lr']
            # training batch size
            self.train_batch = config['vgg_model_params']['train_batch']
            # validation bath size
            self.val_batch = config['vgg_model_params']['val_batch']
            # supported image formats
            self.img_format = eval(config['vgg_model_params']['img_format'])
            # target size of images
            self.target_size = eval(config['vgg_model_params']['target_size'])
            # image color format
            self.image_color_mode = config['vgg_model_params']['image_color_mode']
            # whether to use pretrained model, if yes then pretrained weights file else None
            self.pretrained_weights = config['vgg_model_params']['pretrained_weights']
            # whether to load initial weights or random weights
            self.initial_weights = config['vgg_model_params']['initial_weights']
            
            # log model configuration
            logging.info('Training configuration:')
            for tag in ['input_paths', 'output_paths', 'vgg_train_paths', 'vgg_val_paths', 'vgg_model_params']:
                for key, val in config[tag].items():
                    logging.info(key+' '+val)
            logging.info('model_libraries.trainVGG.parTrain.init completed')    
        except Exception as error:
            logging.exception('Error while reading configuration file')
            raise error
    
         
    def train(self):
        logging.info('model_libraries.trainVGG.parTrain.train started')
        try:
            data_gen_args = dict(rotation_range=0.2,
                                 width_shift_range=0.05,
                                 height_shift_range=0.05,
                                 shear_range=0.05,
                                 zoom_range=0.05,
                                 horizontal_flip=True,
                                 fill_mode='nearest',
                                 rescale=1./255
                                )

            # log data_gen_args values
            logging.info('data_gen_args:')
            for key, val in data_gen_args.items():
                logging.info(key+' '+str(val))

            # define training generator
            train_generator = trainGenerator(int(self.train_batch), 
                                             self.train_images_dir.strip('/')+'_'+self.vgg_images, 
                                             pd.read_csv(self.train_csv), 
                                             self.x_col, 
                                             self.y_col,
                                             data_gen_args, 
                                             save_to_dir=None, 
                                             image_color_mode=self.image_color_mode,
                                             target_size=self.target_size)

            # define validation generator
            validation_generator = validationGenerator(int(self.val_batch), 
                                                       self.val_images_dir.strip('/')+'_'+self.vgg_images,
                                                       pd.read_csv(self.val_csv), 
                                                       self.x_col, 
                                                       self.y_col,  
                                                       data_gen_args, 
                                                       save_to_dir=None, 
                                                       image_color_mode=self.image_color_mode,
                                                       target_size=self.target_size)

            # define model
            model = vgg_12(pretrained_weights=self.pretrained_weights, 
                           input_size=(self.target_size[0], self.target_size[1], 3), 
                           initial_weights=self.initial_weights, 
                           lr=float(self.lr))
            logging.info('model- ')
            logging.info(vgg_12)

            early_stopping = EarlyStopping(monitor='val_mean_squared_error', mode='min', patience=10, verbose=1)
            model_checkpoint = ModelCheckpoint(self.model_name, monitor='val_mean_squared_error', verbose=1, save_best_only=True)

            history = model.fit_generator(train_generator, 
                                          epochs=50,
                                          steps_per_epoch=4253,
                                          validation_data=validation_generator,
                                          validation_steps=1050,
                                          callbacks=[model_checkpoint, early_stopping],
                                          verbose=1)
            logging.info(self.model_name+' trained and saved successfully')
            logging.info('model_libraries.trainVGG.parTrain.train completed')
        except Exception as error:
            logging.exception('Error in training model')
            raise error
        

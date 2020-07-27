from model_libraries.dataFlatten import testGenerator
import numpy as np
from sklearn import metrics
from tensorflow.python.keras.models import load_model
import logging
import os
import pandas as pd

class generateEvaluationMetrics:
    def __init__(self, config):
        logging.info('model_libraries.generateEvaluationMetrics.generateEvaluationMetrics.init started')
        try:
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
            
            # test images directory
            self.test_images_dir = os.path.join(self.output_dir, config['vgg_test_paths']['test_images_dir'])
            # test csv file
            self.test_csv = os.path.join(self.output_dir, config['vgg_test_paths']['test_csv'])
            
            # x_col or column containing image names
            self.x_col = config['vgg_model_params']['x_col']
            # y_col or target column to train vgg model on
            self.y_col = config['vgg_model_params']['y_col']
            # training batch size
            self.train_batch = config['vgg_model_params']['train_batch']
            # test batch size
            self.test_batch = config['vgg_model_params']['test_batch']
            # target size of images
            self.target_size = eval(config['vgg_model_params']['target_size'])
            # image color format
            self.image_color_mode = config['vgg_model_params']['image_color_mode']
            
            # load trained model
            self.loadModel()
            
            logging.info('model_libraries.generateEvaluationMetrics.generateEvaluationMetrics.init completed')    
        except Exception as error:
            logging.exception('Error while reading configuration file')
            raise error
            
    
    def loadModel(self):
        logging.info('model_libraries.generateEvaluationMetrics.generateEvaluationMetrics.loadModel started')
        try:
            # load trained model
            self.model = load_model(self.model_name)
            logging.info('model_libraries.generateEvaluationMetrics.generateEvaluationMetrics.loadModel completed')
        except Exception as error:
            logging.exception(error)
            raise error
            
            
    def mean_absolute_percentage_error(self, y_true, y_pred): 
        try:
            y_true, y_pred = np.array(y_true), np.array(y_pred)
            return np.mean(np.abs((y_true - y_pred) / y_true))*100
        except Exception as error:
            logging.exception(error)
            raise error


    def generateMetrics(self):
        logging.info('model_libraries.generateEvaluationMetrics.generateEvaluationMetrics.generateMetrics started')
        try:
            # train data prediction
            # define test generator for preprocessing data as per train data
            data_gen_args = dict(
                                rescale=1./255
                    )                    
            test_generator1 = testGenerator(int(self.train_batch), 
                                            self.train_images_dir.strip('/')+'_'+self.vgg_images, 
                                            pd.read_csv(self.train_csv),
                                            data_gen_args, 
                                            x_col=self.x_col, 
                                            y_col='', 
                                            save_to_dir=None, 
                                            image_color_mode=self.image_color_mode,
                                            target_size=self.target_size)

            # load test data csv file
            train_data = pd.read_csv(self.train_csv)

            # make perdictions for the test data
            train_data[self.y_col+'_pred'] = self.model.predict_generator(test_generator1)

            # print various perfomance metrics for the test data
            logging.info("Train RMSE: "+str(np.sqrt(metrics.mean_squared_error(train_data[[self.y_col]], train_data[[self.y_col+'_pred']]))))
            logging.info("Train MSE: "+str(metrics.mean_squared_error(train_data[[self.y_col]], train_data[[self.y_col+'_pred']])))
            logging.info("Train MAE: "+str(metrics.mean_absolute_error(train_data[[self.y_col]], train_data[[self.y_col+'_pred']])))
            logging.info("Train MAPE: "+str(self.mean_absolute_percentage_error(train_data[[self.y_col]], train_data[[self.y_col+'_pred']])))

            # save train data predictions
            train_data.to_csv(os.path.join(self.output_dir, 'train_predictions.csv'), index=False)

            # test data prediction
            data_gen_args = dict(
                                rescale=1./255
                    )

            test_generator2 = testGenerator(int(self.test_batch), 
                                            self.test_images_dir.strip('/')+'_'+self.vgg_images, 
                                            pd.read_csv(self.test_csv),
                                            data_gen_args, 
                                            x_col=self.x_col, 
                                            y_col='',
                                            save_to_dir=None, 
                                            image_color_mode=self.image_color_mode,
                                            target_size=self.target_size)

            # load test data csv file
            test_data = pd.read_csv(self.test_csv)

            # make perdictions for the test data
            test_data[self.y_col+'_pred'] = self.model.predict_generator(test_generator2)

            logging.info("Test RMSE: "+str(np.sqrt(metrics.mean_squared_error(test_data[[self.y_col]], test_data[[self.y_col+'_pred']]))))
            logging.info("Test MSE: "+str(metrics.mean_squared_error(test_data[[self.y_col]], test_data[[self.y_col+'_pred']])))
            logging.info("Test MAE: "+str(metrics.mean_absolute_error(test_data[[self.y_col]], test_data[[self.y_col+'_pred']])))
            logging.info("Test MAPE: "+str(self.mean_absolute_percentage_error(test_data[[self.y_col]], test_data[[self.y_col+'_pred']])))
            
            # save test data predictions
            test_data.to_csv(os.path.join(self.output_dir, 'test_predictions.csv'), index=False)
            logging.info('model_libraries.generateEvaluationMetrics.generateEvaluationMetrics.generateMetrics completed')
        except Exception as error:
            logging.exception(error)
            raise error
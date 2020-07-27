import shutil
import os
from tqdm import tqdm
import cv2
import pandas as pd
import configparser
import imutils
import logging
import glob

class prepareData:
    def __init__(self, config):
        logging.info('data_prepare.prepareData.prepareData.init started')
        try:
            # csv file path for actual lai and images names
            self.csv_file = eval(config['input_paths']['csv_file'])
            # images folder path
            self.images_dir = eval(config['input_paths']['images_dir'])
            
            # directory to save ouput
            self.output_dir = config['output_paths']['output_dir']
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

            # test images directory
            self.test_images_dir = os.path.join(self.output_dir, config['vgg_test_paths']['test_images_dir'])
            # validation csv file
            self.test_csv = os.path.join(self.output_dir, config['vgg_test_paths']['test_csv'])

            # target size of images
            self.target_size = eval(config['vgg_model_params']['target_size'])
            # supported image formats
            self.img_format = eval(config['vgg_model_params']['img_format'])
            # x_col
            self.x_col = config['vgg_model_params']['x_col']
            # y_col
            self.y_col = config['vgg_model_params']['y_col']
            # whether to use pretrained model, if yes then pretrained weights file else None
            self.pretrained_weights = config['vgg_model_params']['pretrained_weights']
            # whether to load initial weights or random weights
            self.initial_weights = config['vgg_model_params']['initial_weights']
            logging.info('data_prepare.prepareData.prepareData.init completed')
        except Exception as error:
            logging.exception('Error while reading configuration file')
            raise error
            
    
    def validate_data(self):
        logging.info('data_prepare.prepareData.prepareData.validate_data started')
        #PENDING- checks masks and images have equal number
        logging.info('Validating data')
        # check images directory and number of images
        total_num_images=0
        for folder in self.images_dir:
            if os.path.exists(folder):
                num_images=0
                for img_format in self.img_format:
                    num_images+=len(glob.glob(os.path.join(folder, '*'+img_format)))
                total_num_images+=num_images
                if total_num_images<1:
                    logging.exception('No images in the Directory- '+str(folder))
                    raise Exception('No images in the Directory- '+str(folder))
                logging.info('Number of images in '+str(folder)+' -'+str(num_images))
            else:
                logging.exception('Directory doesn\'t exists- '+str(folder))
                raise Exception('Directory doesn\'t exists- '+str(folder))
        logging.info('Total number of images- '+str(total_num_images))
        
        # check csv files
        for csv_file in self.csv_file:
            try:
                csv = pd.read_csv(csv_file)
            except Exception as error:
                logging.exception('Error in reading %s file'%csv_file)
                raise error

            columns = csv.columns.values.tolist()
            if (self.x_col in columns) and (self.y_col in columns):
                logging.info('%s file validated'%csv_file)
            else:
                logging.exception(self.x_col+' and '+self.y_col+' columns not found in %s file'%csv_file)
                raise Exception(self.x_col+' and '+self.y_col+' columns not found in %s file'%csv_file)

        # check saved models
        if self.pretrained_weights!='None':
            if os.path.exists(self.pretrained_weights):
                logging.info('Model validated')
            else:
                logging.exception('Pretrained Model- '+self.pretrained_weights+' not found')
                raise Exception('Pretrained Model- '+self.pretrained_weights+' not found')
        else:
            logging.info('No pretrained weights specified')
        logging.info('data validated successfully')   
        logging.info('data_prepare.prepareData.prepareData.validate_data completed')
        
        
    def moveData(self, src='data/train/image/', dest='data/val/image/', percent=10):
        logging.info('data_prepare.prepareData.prepareData.moveData from {0} to {1} started'.format(src, dest))
        div = 100//percent
        i=0
        images = []
        for img_format in self.img_format:
            images+=glob.glob(os.path.join(src, '*'+img_format))
        for image in tqdm(images):
            if i%div==0:
                shutil.move(image, os.path.join(dest, image.split('/')[-1]))
            i+=1
            
        logging.info('data_prepare.prepareData.prepareData.moveData from {0} to {1} completed'.format(src, dest))
        

    # generate csv with all the images listed under field column
    def generate_csv(self, src, dest):
        logging.info('data_prepare.prepareData.prepareData.generate_csv for %s started'%dest)
        images = os.listdir(src)
        df = pd.DataFrame()
        target = []
        for image in images:
            target.append(self.data[self.y_col][self.data[self.x_col]==image].values[0]) 
        df[self.x_col] = images
        df[self.y_col] = target
        df.to_csv(dest, index=False)
        logging.info('data_prepare.prepareData.prepareData.generate_csv for %s completed'%dest)
    
    
    # generate csv for train and test data
    def generateCSV(self):
        logging.info('data_prepare.prepareData.prepareData.generateCSV started')
        try:
            self.data = pd.DataFrame()
            for csv_file in self.csv_file:
                self.data = pd.concat([self.data, pd.read_csv(csv_file)])
            if len(self.data)!=len(os.listdir(self.train_images_dir)+os.listdir(self.test_images_dir)+os.listdir(self.val_images_dir)):
                raise Exception('csv file length doesn\'t match with images directories')
            self.generate_csv(src=self.train_images_dir, dest=self.train_csv)
            self.generate_csv(src=self.val_images_dir, dest=self.val_csv)
            self.generate_csv(src=self.test_images_dir, dest=self.test_csv)
            logging.info('data_prepare.prepareData.prepareData.generateCSV completed')
        except Exception as error:
            logging.exception(error)
            raise error
        
    # split data to train and validation folders
    def splitData(self):
        logging.info('data_prepare.prepareData.prepareData.splitData started')
        # create train, validation and test folders
        try:
            for directory in [self.train_images_dir, self.val_images_dir, self.test_images_dir]:
                temp = ''
                for folder in directory.split('/'):
                    temp = temp+folder+'/'
                    if os.path.exists(temp):
                        pass
                    else:
                        os.mkdir(temp)
        except Exception as error:
            logging.exception(error)
            raise error
                    
        # split data into train, val and test folder
        for folder in self.images_dir:
            self.moveData(src=folder, dest=self.val_images_dir, percent=10)
            self.moveData(src=folder, dest=self.test_images_dir, percent=10)
            self.moveData(src=folder, dest=self.train_images_dir, percent=100)

        # split csv file to train, validation and test
        self.generateCSV()
        logging.info('data_prepare.prepareData.prepareData.splitData completed')
        

    # generates the processed images for training vgg model
    def reshapeImages(self):
        logging.info('data_prepare.prepareData.prepareData.reshapeImages started')
        # create vgg reshaped images folder
        try:
            for folder in [self.train_images_dir, self.val_images_dir, self.test_images_dir]:
                if os.path.exists(folder.strip('/')+'_'+self.vgg_images):
                    pass
                else:
                    os.mkdir(folder.strip('/')+'_'+self.vgg_images)
        except Exception as error:
            logging.exception('error while creating directory-'+os.path.join(folder, self.vgg_images))
            raise error
        
        # reshape images
        for folder in [self.train_images_dir, self.val_images_dir, self.test_images_dir]:
            self.reshape_images(folder)
        logging.info('data_prepare.prepareData.prepareData.reshapeImages completed')
        
            
    def reshape_images(self, folder):
        logging.info('data_prepare.prepareData.prepareData.reshape_images for %s started'%folder)
        # get all the images in images folder to be reshaped
        images=os.listdir(folder)
        
        try:
            # apply padding and rotation whereever required
            for image in tqdm(images):
                img = cv2.imread(os.path.join(folder, image))
                h, w = img.shape[:2]

                # rotate image if image height is less than image width
                if h<w:
                    img = imutils.rotate_bound(img, 90)

                h, w = img.shape[:2]
                # apply black padding in height if height is less than 1024
                if h<self.target_size[0]:
                    v_border = int((self.target_size[0]-h)//2)
                    img = cv2.copyMakeBorder(img, v_border, v_border, 0, 0, cv2.BORDER_CONSTANT, value=0)
                    h, w = img.shape[:2]

                # apply black padding in width if width is less than 256
                if w<self.target_size[1]:
                    h_border = int((self.target_size[1]-w)//2)
                    img = cv2.copyMakeBorder(img, 0, 0, h_border, h_border, cv2.BORDER_CONSTANT, value=0)
                    h, w = img.shape[:2]

                # save image
                cv2.imwrite(os.path.join(folder.strip('/')+'_'+self.vgg_images, image), img)
        except Exception as error:
            logging.exception(error)
            raise error
            
        logging.info('data_prepare.prepareData.prepareData.reshape_images for %s completed'%folder)
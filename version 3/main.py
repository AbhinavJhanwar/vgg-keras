import datetime
import logging
import absl.logging
logging.root.removeHandler(absl.logging._absl_handler)
absl.logging._warn_preinit_stderr = False

import warnings
warnings.filterwarnings('ignore')

import configparser
from model_libraries.trainVGG import trainVGG
from data_prepare.prepareData import prepareData
from model_libraries.generateEvaluationMetrics import generateEvaluationMetrics

logging.basicConfig(filename='training_logs.log',
                    level = logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    datefmt='%d-%b-%y %H:%M:%S')
logging.info(datetime.datetime.now())


try:
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    set_session(tf.Session(config=config))
except Exception as error:
    logging.exception('Unable to load gpu, using cpu instead')


if __name__=='__main__':
    # load configurations
    try:
        logging.info('Loading configuration file')
        config = configparser.ConfigParser()
        config.read('training_config.conf')
        logging.info('Configuration file loaded successfully')
    except Exception as error:
        logging.exception('Could not load configuration file')
        raise error
            
    # data preparation object
    dataPrepare = prepareData(config)
    # validate data
    dataPrepare.validate_data()
    # split data into train, test and validation sets
    dataPrepare.splitData()    
    # reshape images for training
    dataPrepare.reshapeImages()
    
    
    # vgg training object
    vggTraining = trainVGG(config)
    # train model
    vggTraining.train()
    
    # get scoring on train and test data
    scoring = generateEvaluationMetrics(config)
    scoring.generateMetrics()
    logging.info('Training Completed')
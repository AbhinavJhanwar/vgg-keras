# Running VGG Training Pipeline

Let's setup environment first. 
Install nvidia drivers, cuda 10.1 and cudnn 7.6.2 for using gpu.

## How to setup environment- LINUX
download anaconda using command wget https://repo.continuum.io/archive/anaconda_file.sh
``` 
wget https://repo.continuum.io/archive/Anaconda3-4.3.0-Linux-x86_64.sh
```

after downloading anaconda we need to install it. for installing anaconda run comman anaconda_file.sh
for detailed instructions to install anaconda check the following link - https://docs.anaconda.com/anaconda/install/linux/
``` 
bash Anaconda3-4.3.0-Linux-x86_64.sh
```

now once anaconda is installed we need to setup a conda environment. for setting this up run command-
conda create --name myenv
``` 
conda create --name lai-prediction python==3.6.8
```

after creating environment, activate it and install gpu requirments
```
source activate lai-prediction
conda install cudatoolkit
```

run requirment.txt file to install necessary python modules
```
pip install -r requirements.txt
```

## How to run training pipeline
modify configuration file **training_config.conf** for necessary paramenters of model and data
* specify paths of csv files separated by comma which contains the target variable in one column and the location/name of images in other column to train the model on in **csv_file** variable
* specify paths of images folders separated by comma to train the model on in **images_dir** variable
* specify various image formats to train model in **img_format** variable
* specify the name of column as in csv file dictating image file name in **x_col** variable
* specify the target column as in csv file in **y_col** variable to train model on
* specify path of any pretrained model to use in **pretrained_weights** otherwise leave as None
after updating configuration file run the python script **main.py**
```
python main.py
```

## Understanding other parameters-
<br/>**output_dir** - path of output directory where all the trained models, generated predictions and processed images will be saved
<br/>**vgg_images** - path of directory where all the processed images as required by vgg network will be saved
<br/>**vgg_model_name** - path of directory where best performing model will be saved
<br/>**train_images_dir**, **val_images_dir**, **test_images_dir** - paths of various directories created after splitting the data into train, test and validation sets
<br/>**train_csv**, **val_csv**, **test_csv** - paths of respective csv files for the splitted data
<br/>**lr** - learning rate of the vgg model
<br/>**train_batch** - batch size for training vgg model
<br/>**val_batch** - batch size for validating vgg model
<br/>**test_batch** - batch size for testing vgg model
<br/>**target_size** - size of input image to train vgg model
<br/>**image_color_mode** - color mode of image to train vgg model. example- 'rgb', 'grayscale'
<br/>**initial_weights** - set True/False based on whether to train model from scratch or load pretrained weights from keras. Note- only valid if **pretrained_weights** is set None

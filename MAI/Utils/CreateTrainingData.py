from pickle import dump
import tensorflow_addons as tfa
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.utils.version_utils import callbacks

import MAI.Utils.Params as params
from MAI.Utils.ReadData.ReadData import prepareDataset
from MAI.Utils.Functions.ASLFunction import AsymetricLossOptimized


def training_function():
    train_df, test_df, all_labels = prepareDataset()
    weight_path = "outputs/{}_weights.best.hdf5".format('xray_class')
    train_df['path'] = train_df['path'].astype(str)
    core_idg = ImageDataGenerator(
            horizontal_flip=True,
            vertical_flip=False,
            height_shift_range=0.1,
            width_shift_range=0.1,
            rotation_range=10,
            shear_range=0.1,
            fill_mode='reflect',
            zoom_range=0.2,
            validation_split=0.2)

    train_gen = core_idg.flow_from_dataframe(dataframe=train_df,
                                                 directory=None,
                                                 x_col='path',
                                                 y_col='newLabel',
                                                 class_mode='categorical',
                                                 classes=all_labels,
                                                 target_size=(params.IMG_SIZE, params.IMG_SIZE),
                                                 color_mode='rgb',
                                                 batch_size=8,
                                                 subset='training')

    valid_gen = core_idg.flow_from_dataframe(dataframe=train_df,
                                                 directory=None,
                                                 x_col='path',
                                                 y_col='newLabel',
                                                 class_mode='categorical',
                                                 classes=all_labels,
                                                 target_size=(params.IMG_SIZE, params.IMG_SIZE),
                                                 color_mode='rgb',
                                                 batch_size=8,
                                                 subset='validation')
    return train_df, test_df, all_labels

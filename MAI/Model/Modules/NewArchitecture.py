import tensorflow as tf
from tensorflow.keras import models, layers
import numpy as np
from tensorflow.python.keras.applications.efficientnet import EfficientNetB7
from tensorflow.python.keras.applications.inception_v3 import InceptionV3
from tensorflow.python.keras.applications.resnet_v2 import ResNet50V2, ResNet101V2, ResNet152V2
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.layers import Dense

from MAI.Model.Modules.help_fucntions import PrimaryCap, CapsuleLayer, Length, Mask
from MAI.Utils.Params import IMG_SIZE, BATCH_SIZE


def global_view(model):
    efficient = ResNet152V2(include_top=False)(model)
    efficient = GlobalAveragePooling2D()(efficient)
    efficient = Dense(256, activation='relu')(efficient)
    efficient = Dropout(0.5)(efficient)
    efficient = Dense(14, activation='sigmoid')(efficient)

    resnet_oneZeroOne = ResNet101V2(include_top=False)(model)
    resnet_oneZeroOne = GlobalAveragePooling2D()(resnet_oneZeroOne)
    resnet_oneZeroOne = Dense(256, activation='relu')(resnet_oneZeroOne)
    resnet_oneZeroOne = Dropout(0.5)(resnet_oneZeroOne)
    resnet_oneZeroOne = Dense(14, activation='sigmoid')(resnet_oneZeroOne)

    resnet_fifty = ResNet50V2(include_top=False)(model)
    resnet_fifty = GlobalAveragePooling2D()(resnet_fifty)
    resnet_fifty = Dense(256, activation='relu')(resnet_fifty)
    resnet_fifty = Dropout(0.5)(resnet_fifty)
    resnet_fifty = Dense(14, activation='sigmoid')(resnet_fifty)
    return efficient, resnet_fifty, resnet_oneZeroOne


def capsNet_view(input, routings):
    x = Conv2D(64, (5, 5), activation='relu')(input)
    x = BatchNormalization()(x)
    x = Conv2D(64, (5, 5), activation='relu')(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)

    primarycaps = PrimaryCap(x, dim_capsule=2, n_channels=8, kernel_size=7, strides=2, padding='valid')

    # Layer 3: Capsule layer. Routing algorithm works here.
    digitcaps = CapsuleLayer(num_capsule=16, dim_capsule=8, routings=routings, name='digitcaps')(primarycaps)
    digitcaps = CapsuleLayer(num_capsule=14, dim_capsule=4, routings=routings, name='digitcaps2')(digitcaps)

    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    # If using tensorflow, this will not be necessary. :)
    out_caps = Length(name='capsnet')(digitcaps)

    # Layer 5: Decoding layer of the capsule output
    decoded = Dense(384, activation='relu')(out_caps)
    decoded = Dense(768, activation='relu')(decoded)
    decoded = Dense(14, activation='sigmoid')(decoded)

    return decoded


def embedded_models(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                    n_class=14,
                    routings=2,
                    batch_size_o=BATCH_SIZE):
    input = Input(shape=input_shape, batch_size=batch_size_o)

    gv_efficient, gv_fifty, gv_one = global_view(input)
    cnv = capsNet_view(input, routings)

    fusion = concatenate([gv_efficient, gv_fifty, cnv])

    fusion = Dense(32)(fusion)
    fusion = Dropout(0.2)(fusion)
    fusion = Dense(14, activation="sigmoid")(fusion)

    train_Model = models.Model(input, fusion)

    return train_Model


def new_embedded_model(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                       n_class=14,
                       routings=2,
                       batch_size_o=BATCH_SIZE):
    input = layers.Input(shape=input_shape, batch_size=batch_size_o)

    # Layer 1: Just a conventional Conv2D layer
    x = Conv2D(64, (5, 5), activation='relu')(input)
    x = BatchNormalization()(x)
    x = Conv2D(64, (5, 5), activation='relu')(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = Conv2D(128, (3, 3), activation='relu')(x)

    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
    primarycaps = PrimaryCap(x, dim_capsule=2, n_channels=8, kernel_size=7, strides=2, padding='valid')

    # Layer 3: Capsule layer. Routing algorithm works here.
    digitcaps = CapsuleLayer(num_capsule=16, dim_capsule=8, routings=routings, name='digitcaps')(primarycaps)
    digitcaps = CapsuleLayer(num_capsule=14, dim_capsule=4, routings=routings, name='digitcaps2')(digitcaps)

    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    # If using tensorflow, this will not be necessary. :)
    out_caps = Length(name='capsnet')(digitcaps)

    model2 = ResNet50V2(include_top=False, weights="imagenet")(input)
    model2 = GlobalAveragePooling2D()(model2)
    model2 = Dense(256)(model2)
    model2 = Dropout(0.5)(model2)
    model2 = Dense(16)(model2)

    common = concatenate([out_caps, model2])
    #
    common = Dense(32)(common)
    common = Dropout(0.2)(common)
    common = Dense(14, activation="sigmoid")(common)

    train_model = models.Model(input, common)

    return train_model

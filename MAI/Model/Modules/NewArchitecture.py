import tensorflow as tf
from tensorflow.keras import models
from tensorflow.python.keras.applications.efficientnet import EfficientNetB1
from tensorflow.python.keras.applications.inception_v3 import InceptionV3
from tensorflow.python.keras.applications.resnet_v2 import ResNet50V2, ResNet101V2, ResNet152V2
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.layers import Dense

from MAI.Model.Modules.help_fucntions import PrimaryCap, CapsuleLayer, Length
from MAI.Utils.Params import IMG_SIZE, BATCH_SIZE


def global_view(model):
    efficient = EfficientNetB1(include_top=False)(model)
    efficient = GlobalMaxPooling2D()(efficient)
    efficient = Dense(128)(efficient)
    efficient = Dropout(0.5)(efficient)
    efficient = Dense(64)(efficient)

    inception = InceptionV3(include_top=False)(model)
    inception = GlobalMaxPooling2D()(inception)
    inception = Dense(128)(inception)
    inception = Dropout(0.5)(inception)
    inception = Dense(64)(inception)
    return concatenate([efficient, inception])


def capsNet_view(model, n_class, routings):
    res_net = ResNet152V2(include_top=False, weights='imagenet')(model)

    primaryCaps = PrimaryCap(res_net, dim_capsule=4, n_channels=8, kernel_size=9, strides=2, padding='valid')
    digitCaps = CapsuleLayer(num_capsule=n_class, dim_capsule=8, routings=routings, name='digitcaps')(primaryCaps)
    digitCaps = CapsuleLayer(num_capsule=n_class + 2, dim_capsule=16, routings=routings + 2, name='digitcaps2')(digitCaps)
    digitCaps = CapsuleLayer(num_capsule=n_class, dim_capsule=8, routings=routings, name='digitcaps3')(digitCaps)

    out_caps = Length(name='capsnet')(digitCaps)
    out_caps = Dense(128)(out_caps)
    out_caps = Dropout(0.5)(out_caps)
    out_caps = Dense(64)(out_caps)
    return out_caps


def embedded_models(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                    n_class=14,
                    routings=2,
                    batch_size_o=BATCH_SIZE):
    input = Input(shape=input_shape, batch_size=batch_size_o)

    model = Conv2D(64, (5, 5), activation='relu', padding='same', kernel_initializer='he_normal')(input)
    model = BatchNormalization()(model)
    model = Conv2D(64, (5, 5), activation='relu', padding='same')(model)
    model = Conv2D(128, (3, 3), activation='relu')(model)
    model = Conv2D(128, (3, 3), activation='relu')(model)

    gv = global_view(model)
    cnv = capsNet_view(model, n_class, routings)

    fusion = concatenate([gv, cnv])

    fusion = Flatten()(fusion)
    fusion = Dense(64)(fusion)
    fusion = Dense(32)(fusion)

    fusion = Dense(14, activation='sigmoid')(fusion)

    train_Model = models.Model(input, fusion)

    return train_Model

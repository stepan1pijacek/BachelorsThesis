import tensorflow as tf
from tensorflow.keras import models
from tensorflow.python.keras.applications.efficientnet import EfficientNetB7
from tensorflow.python.keras.applications.inception_v3 import InceptionV3
from tensorflow.python.keras.applications.resnet_v2 import ResNet50V2, ResNet101V2, ResNet152V2
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.layers import Dense

from MAI.Model.Modules.help_fucntions import PrimaryCap, CapsuleLayer, Length
from MAI.Utils.Params import IMG_SIZE, BATCH_SIZE


def global_view(model):
    efficient = ResNet152V2(include_top=False)(model)
    efficient = GlobalMaxPooling2D()(efficient)
    efficient = Dense(256)(efficient)
    efficient = Dropout(0.5)(efficient)
    efficient = Dense(16)(efficient)
    return efficient


def capsNet_view(input, routings):
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
    return out_caps


def embedded_models(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                    n_class=14,
                    routings=2,
                    batch_size_o=BATCH_SIZE):
    input = Input(shape=input_shape, batch_size=batch_size_o)

    gv_efficient = global_view(input)
    cnv = capsNet_view(input, routings)

    fusion = concatenate([gv_efficient, cnv])

    fusion = Dense(32)(fusion)
    fusion = Dropout(0.2)(fusion)
    fusion = Dense(14, activation="sigmoid")(fusion)

    train_Model = models.Model(input, fusion)

    return train_Model

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
    inception = InceptionV3(include_top=False)(model)
    inception = GlobalMaxPooling2D()(inception)
    inception = Dense(256)(inception)
    inception = Dropout(0.5)(inception)
    inception = Dense(16)(inception)

    efficient = ResNet152V2(include_top=False)(model)
    efficient = GlobalMaxPooling2D()(efficient)
    efficient = Dense(256)(efficient)
    efficient = Dropout(0.5)(efficient)
    efficient = Dense(16)(efficient)
    return efficient, inception


def capsNet_view(input, routings):

    model = Conv2D(64, (5, 5), activation='relu', padding='same')(input)
    model = BatchNormalization()(model)
    model = Conv2D(64, (5, 5), activation='relu')(model)
    model = Conv2D(128, (3, 3), activation='relu')(model)
    model = Conv2D(128, (3, 3), activation='relu')(model)

    primaryCaps = PrimaryCap(model, dim_capsule=4, n_channels=16, kernel_size=7, strides=2, padding='valid')

    # Layer 3: Capsule layer. Routing algorithm works here.
    digitCaps = CapsuleLayer(num_capsule=14, dim_capsule=8, routings=routings, name='digitcaps')(primaryCaps)
    digitCaps = CapsuleLayer(num_capsule=18, dim_capsule=16, routings=routings, name='digitcaps_inbetween_step')(digitCaps)
    digitCaps = CapsuleLayer(num_capsule=14, dim_capsule=4, routings=routings, name='digitcaps2')(digitCaps)

    out_caps = Length(name='capsnet')(digitCaps)
    out_caps = Dense(256)(out_caps)
    out_caps = Dropout(0.5)(out_caps)
    out_caps = Dense(16)(out_caps)

    return out_caps


def embedded_models(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                    n_class=14,
                    routings=2,
                    batch_size_o=BATCH_SIZE):
    input = Input(shape=input_shape, batch_size=batch_size_o)

    gv_efficient, gv_inception = global_view(input)
    cnv = capsNet_view(input, routings)

    fusion = concatenate([gv_efficient, gv_inception, cnv])

    fusion = Dense(32)(fusion)
    fusion = Dropout(0.2)(fusion)
    fusion = Dense(14, activation="sigmoid")(fusion)

    train_Model = models.Model(input, fusion)

    return train_Model

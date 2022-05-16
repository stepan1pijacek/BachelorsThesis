import tensorflow as tf
from tensorflow.keras import models
from tensorflow.python.keras.applications.resnet_v2 import ResNet50V2, ResNet101V2, ResNet152V2
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.layers import Dense

from MAI.Model.Modules.help_fucntions import PrimaryCap, CapsuleLayer, Length
from MAI.Utils.Params import IMG_SIZE, BATCH_SIZE


def module1(model):
    efficient_net = ResNet50V2(include_top=False)(model)
    # efficient_net = GlobalMaxPooling2D()(efficient_net)
    # efficient_net = Dense(128)(efficient_net)
    # efficient_net = Dropout(0.5)(efficient_net)
    # vracet spis hodnoty z efficient_net ne Dense net to same u dalsich modulu
    return efficient_net


def module2(model):
    inception_net = ResNet101V2(include_top=False)(model)
    # inception_net = GlobalMaxPooling2D()(inception_net)
    # inception_net = Dense(128)(inception_net)
    # inception_net = Dropout(0.5)(inception_net)
    return inception_net


def module3(model):
    res_net = ResNet152V2(include_top=False)(model)
    # res_net = GlobalMaxPooling2D()(res_net)
    # res_net = Dense(128)(res_net)
    # res_net = Dropout(0.5)(res_net)
    return res_net


def embedded_models(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                    n_class=14,
                    routings=2,
                    batch_size_o=BATCH_SIZE):
    input = Input(shape=input_shape, batch_size=batch_size_o)

    model = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(input)
    model = Activation('relu')(model)

    model = Conv2D(3, 1, strides=1, padding='same')(model)

    module1_out = module1(model)
    module2_out = module2(model)
    module3_out = module3(model)

    fusion = concatenate([module1_out, module2_out, module3_out])
    # input_caps_net = layers.Input(shape=softmax_out, batch_size=batch_size)
    #
    # input_caps_net = Input(shape=fusion, batch_size=4)
    # conv1 = layers.Conv2D(filters=256,
    # kernel_size=7, strides=1, padding='valid', activation='relu', name='conv1')(fusion)
    # print(tf.shape(conv1, name="conv1"))
    primaryCaps = PrimaryCap(fusion, dim_capsule=2, n_channels=16, kernel_size=9, strides=2, padding='valid')
    digitCaps = CapsuleLayer(num_capsule=n_class, dim_capsule=32, routings=routings, name='digitcaps')(primaryCaps)

    print(tf.shape(digitCaps))
    out_caps = Length(name='capsnet')(digitCaps)
    print(tf.shape(out_caps))
    out_caps = Dense(14, activation="sigmoid")(out_caps)
    train_Model = models.Model(input, out_caps)

    # plot_model(
    #    train_Model, to_file='Output/model.png', show_shapes=True, show_layer_names=True,
    #    rankdir='TB', expand_nested=False, dpi=96
    # )

    return train_Model

from tensorflow.keras import layers, models
from tensorflow.python.keras.applications.efficientnet import EfficientNetB4
from tensorflow.python.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.python.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.layers import Dense, GlobalMaxPooling2D, Dropout

from MAI.Model.Modules.help_fucntions import PrimaryCap, CapsuleLayer, Length
from MAI.Utils.Params import IMG_SIZE


def module1(model):
    efficient_net = EfficientNetB4(include_top=False)(model)
    efficient_net = GlobalMaxPooling2D()(efficient_net)
    efficient_net = Dense(128)(efficient_net)
    efficient_net = Dropout(0.5)(efficient_net)
    return Dense(64)(efficient_net)


def module2(model):
    inception_net = InceptionResNetV2(include_top=False)(model)
    inception_net = GlobalMaxPooling2D()(inception_net)
    inception_net = Dense(128)(inception_net)
    inception_net = Dropout(0.5)(inception_net)
    return Dense(64)(inception_net)


def module3(model):
    res_net = ResNet50V2(include_top=False)(model)
    res_net = GlobalMaxPooling2D()(res_net)
    res_net = Dense(128)(res_net)
    res_net = Dropout(0.5)(res_net)
    return Dense(64)(res_net)


def embedded_models():
    input = Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    model = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(input)
    model = Activation('relu')(model)

    model = Conv2D(3, 1, strides=1, padding='same')(model)

    module1_out = module1(model)
    module2_out = module2(model)
    module3_out = module3(model)

    fusion = concatenate([module1_out, module2_out, module3_out])
    fusion = layers.Reshape(target_shape=(IMG_SIZE, IMG_SIZE, 3))(fusion)
    conv1 = Conv2D(3, 1, strides=1, padding='same')(fusion)
    primaryCaps = PrimaryCap(conv1, dim_capsule=2, n_channels=8, kernel_size=9, strides=2, padding='valid')
    digitCaps = CapsuleLayer(num_capsule=14, dim_capsule=16, routings=2, name='digitcaps')(primaryCaps)

    out_caps = Length(name='capsnet')(digitCaps)
    out_caps = Dense(14, activation="relu")(out_caps)
    train_Model = models.Model(input, out_caps)

    return train_Model

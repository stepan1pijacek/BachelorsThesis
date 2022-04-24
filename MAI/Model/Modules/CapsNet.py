import tensorflow_addons as tfa
from tensorflow import optimizers
from tensorflow.keras import layers, models
from tensorflow.python.keras.applications.efficientnet import EfficientNetB1, EfficientNetB4
from tensorflow.python.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.python.keras.applications.resnet_v2 import ResNet152V2
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.layers import Dense, GlobalMaxPooling2D, Dropout
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.utils.vis_utils import plot_model
from tensorflow.python.framework.ops import disable_eager_execution

from MAI.Model.Modules.help_fucntions import PrimaryCap, CapsuleLayer, Length
from MAI.Utils.Params import IMG_SIZE, NUM_CLASSES, METRICS


def embedded_model():
    input = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    model = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(input)
    model = Activation('relu')(model)

    model = Conv2D(3, 1, strides=1, padding='same')(model)

    model1 = EfficientNetB1(include_top=False)(model)
    model1 = GlobalMaxPooling2D()(model1)
    model1 = Dense(128)(model1)
    model1 = Dropout(0.5)(model1)
    model1 = Dense(64)(model1)

    model2 = InceptionResNetV2(include_top=False)(input)
    model2 = GlobalMaxPooling2D()(model2)
    model2 = Dense(128)(model2)
    model2 = Dropout(0.5)(model2)
    model2 = Dense(64)(model2)

    model3 = ResNet152V2(include_top=False)(input)
    model3 = GlobalMaxPooling2D()(model3)
    model3 = Dense(128)(model3)
    model3 = Dropout(0.5)(model3)
    model3 = Dense(64)(model3)

    common = concatenate([model1, model2, model3])

    common = Dense(64)(common)
    common = Dense(32)(common)

    common = Dense(NUM_CLASSES, activation='sigmoid')(common)

    model = Model(inputs=input, outputs=common)
    model.compile(optimizer=optimizers.Adam(lr=0.0001), loss=tfa.losses.sigmoid_focal_crossentropy,
                  metrics=METRICS)
    model.summary()
    inputs = model.input
    plot_model(
        model, to_file='outputs/model.png', show_shapes=True, show_layer_names=True,
        rankdir='TB', expand_nested=False, dpi=96
    )

    return model


def CapsNet(input_shape=(IMG_SIZE, IMG_SIZE, 3),
            n_class=14,
            routings=2,
            batch_size=4):
    x = layers.Input(shape=input_shape, batch_size=batch_size)

    # conv1 = layers.Conv2D(filters=256, kernel_size=7, strides=1, padding='valid', activation='relu', name='conv1')(x)
    conv1 = EfficientNetB4(include_top=False)(x)
    # Layer 2: Conv2D layer with `squash` activation, then reshape to [None, num_capsule, dim_capsule]
    primarycaps = PrimaryCap(conv1, dim_capsule=2, n_channels=8, kernel_size=9, strides=2, padding='valid')

    # Layer 3: Capsule layer. Routing algorithm works here.
    digitcaps = CapsuleLayer(num_capsule=n_class, dim_capsule=16, routings=routings, name='digitcaps')(primarycaps)

    # Layer 4: This is an auxiliary layer to replace each capsule with its length. Just to match the true label's shape.
    # If using tensorflow, this will not be necessary. :)
    out_caps = Length(name='capsnet')(digitcaps)
    out_caps = Dense(14, activation="sigmoid")(out_caps)
    train_model = models.Model(x, out_caps)

    return train_model

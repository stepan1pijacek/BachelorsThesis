from pickle import dump
import tensorflow_addons as tfa
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.utils.version_utils import callbacks

import MAI.Utils.Params as params
from MAI.Utils.ReadData.ReadData import prepareDataset
from MAI.Utils.Functions.ASLFunction import AsymetricLossOptimized


class TrainingClass:
    @staticmethod
    def training_function(model):
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

        log = callbacks.CSVLogger('outputs/log.csv')
        checkpoint = callbacks.ModelCheckpoint(weight_path, monitor='val_auc', mode='max',
                                               save_best_only=True, save_weights_only=True, verbose=1)
        lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: 0.001 * (0.0001 ** epoch))

        # compile the model
        model.compile(optimizer=tfa.optimizers.AdamW(learning_rate=params.trans_learning_rate,
                                                     weight_decay=params.trans_weight_decay),
                      loss=AsymetricLossOptimized,
                      metrics=params.METRICS
                      )

        history = model.fit(
            train_gen,
            batch_size=8,
            epochs=100,
            validation_steps=valid_gen.samples // 16,
            steps_per_epoch=200,
            validation_data=valid_gen,

            callbacks=[log, checkpoint, lr_decay]
        )

        with open(f'outputs/history.txt',
                  'wb') as handle:  # saving the history of the model trained for another 50 Epochs
            dump(history.history, handle)

        model.save_weights("outputs/{}_weights.last.hdf5".format('xray_class'))

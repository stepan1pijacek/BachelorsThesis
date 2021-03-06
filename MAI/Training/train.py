import tensorflow as tf

from pickle import dump
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.utils.version_utils import callbacks

import MAI.Utils.Params as params
from MAI.Utils.ReadData.ReadData import main
from MAI.Utils.Functions.ASLFunction import AsymetricLossOptimized


class TrainingClass:
    @staticmethod
    def training_function(model):
        print("Training started \n")
        train_df, test_df, all_labels = main()
        weight_path = "Output/{}_weights.best.hdf5".format('xray_class')
        train_df['path'] = train_df['path'].astype(str).copy()
        core_idg = ImageDataGenerator(
            horizontal_flip=False,
            vertical_flip=False,
            height_shift_range=0.1,
            width_shift_range=0.1,
            rotation_range=10,
            shear_range=0.1,
            fill_mode='reflect',
            zoom_range=0.2,
            validation_split=0.2)
        # GCD 4 can be achieved with split factor of 0.35546

        train_gen = core_idg.flow_from_dataframe(dataframe=train_df,
                                                 directory=None,
                                                 x_col='path',
                                                 y_col='newLabel',
                                                 class_mode='categorical',
                                                 classes=all_labels,
                                                 target_size=(params.IMG_SIZE, params.IMG_SIZE),
                                                 color_mode='rgb',
                                                 batch_size=params.BATCH_SIZE,
                                                 subset='training')

        valid_gen = core_idg.flow_from_dataframe(dataframe=train_df,
                                                 directory=None,
                                                 x_col='path',
                                                 y_col='newLabel',
                                                 class_mode='categorical',
                                                 classes=all_labels,
                                                 target_size=(params.IMG_SIZE, params.IMG_SIZE),
                                                 color_mode='rgb',
                                                 batch_size=params.BATCH_SIZE,
                                                 subset='validation')

        log = callbacks.CSVLogger('Output/log.csv')
        checkpoint = callbacks.ModelCheckpoint(weight_path, monitor='val_auc', mode='max',
                                               save_best_only=True, save_weights_only=True, verbose=1)
        lr_decay = callbacks.LearningRateScheduler(schedule=lambda epoch: 0.001 * (0.0001 ** epoch))

        model.compile(
            optimizer=tf.optimizers.Adam(
                learning_rate=params.trans_learning_rate,
            ),
            loss=AsymetricLossOptimized,
            metrics=params.METRICS
        )

        print("Printing number of valid gen samples \n")
        print(valid_gen.samples)
        print("Printing number of training samples")
        print(train_gen.samples)
        history = model.fit(
            train_gen,
            batch_size=params.BATCH_SIZE,
            epochs=100,
            validation_steps=valid_gen.samples // 8,
            steps_per_epoch=500,
            validation_data=valid_gen,
            callbacks=[log, checkpoint, lr_decay]
        )

        with open(f'Output/history.txt',
                  'wb') as handle:  # saving the history of the model trained for another 50 Epochs
            dump(history.history, handle)

        model.save_weights("Output/{}_weights.last.hdf5".format('xray_class'))

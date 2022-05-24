import tensorflow as tf

import MAI.Training.train as train_idk
from MAI.Model.Modules.NewArchitecture import embedded_models
from MAI.Evaluation.Evaluate import evaluate


def main():
    try:
        with tf.device('/device:GPU:0'):
            train_idk.TrainingClass.training_function(embedded_models())
            evaluate(embedded_models())
    except RuntimeError as e:
        print(e)


if __name__ == '__main__':
    main()

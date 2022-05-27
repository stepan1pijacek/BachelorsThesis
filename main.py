import os
import tensorflow as tf

import MAI.Training.train as train_idk
from MAI.Model.Modules.NewArchitecture import embedded_models, new_embedded_model
from MAI.Evaluation.Evaluate import evaluate


def main():
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    train_idk.TrainingClass.training_function(new_embedded_model())
    evaluate(embedded_models())


if __name__ == '__main__':
    main()

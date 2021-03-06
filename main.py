import os

import MAI.Training.train as train_idk
from MAI.Model.Modules.NewArchitecture import embedded_models
from MAI.Evaluation.Evaluate import evaluate


def main():
    # Limit the programm to one GPU
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    train_idk.TrainingClass.training_function(embedded_models())
    evaluate(embedded_models())


if __name__ == '__main__':
    main()

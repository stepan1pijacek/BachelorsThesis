import MAI.Model.ReCa.CapsuleNetwork as Caps
import MAI.Training.CapsNetTraining as train
import MAI.Training.train as train_idk
from MAI.Model.Modules.CapsNet import CapsNet
from MAI.Utils.CreateTrainingData import training_function
import tensorflow as tf
import os
import json


def main():
    model = CapsNet()
    print("\n\n###############################################", flush=True)
    print("Output", flush=True)
    print("###############################################\n", flush=True)
    train_idk.TrainingClass.training_function(model)
    # Write log folder and arguments
    # if not os.path.exists("Output"):
    #    os.makedirs("Output")

    # train_df, test_ds, class_name = training_function()
    # Train capsule network
    # acc = train.train(train_df, test_ds, class_name)
    # with open("experiments/results.txt", 'a') as f:
    #    f.write("%s;%.5f\n" % ("Output", acc))


if __name__ == '__main__':
    main()

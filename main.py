import MAI.Model.ReCa.CapsuleNetwork as Caps
import MAI.Training.CapsNetTraining as train
from MAI.Utils.CreateTrainingData import training_function
import tensorflow as tf
import os
import json


def main():
    model = Caps.CapsNet()
    print("\n\n###############################################", flush=True)
    print("Output", flush=True)
    print("###############################################\n", flush=True)

    # Write log folder and arguments
    if not os.path.exists("Output"):
        os.makedirs("Output")

    with open("%s/args.txt" % "Output", "w") as file:
        file.write(json.dumps(vars("output_json")))

    # Train capsule network
    acc = train(training_function())
    with open("experiments/results.txt", 'a') as f:
        f.write("%s;%.5f\n" % ("Output", acc))


if __name__ == '__main__':
    main()

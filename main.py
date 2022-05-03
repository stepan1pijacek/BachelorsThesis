# import MAI.Model.ReCa.CapsuleNetwork as Caps
# import MAI.Training.CapsNetTraining as train
import MAI.Training.train as train_idk
from MAI.Model.Modules.NewArchitecture import embedded_models


def main():
    model = embedded_models()
    train_idk.TrainingClass.training_function(model)


if __name__ == '__main__':
    main()

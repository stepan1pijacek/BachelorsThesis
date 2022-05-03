# import MAI.Model.ReCa.CapsuleNetwork as Caps
# import MAI.Training.CapsNetTraining as train
import MAI.Training.train as train_idk
from MAI.Model.Modules.NewArchitecture import embedded_models


def main():
    train_idk.TrainingClass.training_function(embedded_models())


if __name__ == '__main__':
    main()

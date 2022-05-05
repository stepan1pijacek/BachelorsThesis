import MAI.Model.ReCa.CapsuleNetwork as Caps
import MAI.Training.CapsNetTraining as train
import MAI.Utils.CreateTrainingData as ctd
import MAI.Training.train as train_idk
from MAI.Model.Modules.NewArchitecture import embedded_models


def main():
    # model = Caps.CapsNet()
    # train.train(ctd.training_function())
    train_idk.TrainingClass.training_function(embedded_models())


if __name__ == '__main__':
    main()

import Model.ReCa.CapsuleNetwork as Caps
import Training.train as train


def main():
    model = Caps.CapsNet()
    train.TrainingClass.training_function(model)


if __name__ == '__main__':
    main()

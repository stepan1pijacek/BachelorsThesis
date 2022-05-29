import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn
from sklearn.metrics import roc_curve, roc_auc_score, \
    accuracy_score, precision_score, recall_score, f1_score, multilabel_confusion_matrix
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

from MAI.Utils.ReadData.ReadData import main
from MAI.Utils.Params import IMG_SIZE, BATCH_SIZE


def evaluate(model):
    train_df, test_df, all_labels = main()
    weight_path = "Output/{}_weights.best.hdf5".format('xray_class')
    test_df['path'] = test_df['path'].astype(str)
    test_core_idg = ImageDataGenerator(
    )

    test_X, test_Y = next(test_core_idg.flow_from_dataframe(
        dataframe=test_df,
        directory=None,
        x_col='path',
        y_col='newLabel',
        class_mode='categorical',
        classes=all_labels,
        target_size=(IMG_SIZE, IMG_SIZE),
        # target_size=(params.IMG_SIZE, params.IMG_SIZE),
        color_mode='rgb',
        batch_size=12880)
    )

    # load the best weights
    model.load_weights(weight_path)
    pred_Y = model.predict(test_X, batch_size=4, verbose=True)
    print(pred_Y)

    for c_label, p_count, t_count in zip(all_labels,
                                         100 * np.mean(pred_Y, 0),
                                         100 * np.mean(test_Y, 0)):
        print('%s: Dx: %2.2f%%, PDx: %2.2f%%' % (c_label, t_count, p_count))

    from statistics import mean
    auc_rocs, thresholds, sensitivity, specificity, accuracy, precision, recall, f1 = get_roc_curve(all_labels,
                                                                                                    pred_Y,
                                                                                                    test_Y)
    from tabulate import tabulate
    table = zip(all_labels, auc_rocs)
    print(f"Mean AUC : {mean(auc_rocs)}")
    print(tabulate(table, headers=['Pathology', 'AUC'], tablefmt='fancy_grid'))

    from tabulate import tabulate
    table = zip(all_labels, auc_rocs, thresholds, sensitivity, specificity, accuracy, precision, recall, f1)
    print(tabulate(table, headers=['Pathology', 'AUC', 'Threshold Value', 'Sensitivity', 'Specificity', 'Accuracy',
                                   'Precision', 'Recall', 'F1 Score'], tablefmt='fancy_grid'))


def get_roc_curve(labels, predicted_vals, test_Y):
    auc_roc_vals = []
    optimal_thresholds = []
    sensitivity = []
    specificity = []
    accuracy = []
    precision = []
    recall = []
    f1 = []
    for i in range(len(labels)):
        try:
            gt = test_Y[:, i]
            pred = predicted_vals[:, i]
            auc_roc = roc_auc_score(gt, pred)  # return
            auc_roc_vals.append(auc_roc)
            fpr, tpr, thresholds = roc_curve(gt, pred)
            optimal_idx = np.argmax(tpr - fpr)
            optimal_threshold = thresholds[optimal_idx]
            y_pred = pred > optimal_threshold
            acc = accuracy_score(gt, y_pred)
            prec = precision_score(gt, y_pred)
            rec = recall_score(gt, y_pred)
            f1_s = f1_score(gt, y_pred)
            accuracy.append(acc)
            precision.append(prec)
            recall.append(rec)
            f1.append(f1_s)
            optimal_thresholds.append(
                optimal_threshold)  # find optimal thresholds https://stats.stackexchange.com/questions/123124/how-to-determine-the-optimal-threshold-for-a-classifier-and-generate-roc-curve
            optimal_tpr = round(tpr[optimal_idx], 3)
            optimal_1_fpr = round(1 - fpr[optimal_idx], 3)
            #             print(f"Length of tpr tpr : {len(tpr)} \n Length of thresholds {len(thresholds)}")
            #             print(f"optimal index : {optimal_idx} \n Optimal 1 - fpr : {optimal_1_fpr}")
            sensitivity.append(optimal_tpr)
            specificity.append(1 - fpr[optimal_idx])
            plt.figure(1, figsize=(10, 10))
            plt.plot([0, 1], [0, 1], 'k--')  # black dash line
            plt.plot(fpr, tpr,
                     label=labels[i] + " (" + str(round(auc_roc, 3)) + ")")
            plt.xlabel('False positive rate')
            plt.ylabel('True positive rate')
            plt.title('ROC curve')
            plt.legend(loc='best')

            plt.savefig('Output/trained_net_2.png')

            cm = multilabel_confusion_matrix(test_Y, y_pred)
            cm_df = pd.DataFrame(cm)
            plt.figure(figsize=(12, 10))
            plt.title('Confusion Matrix')
            seaborn.heatmap(cm_df, annot=True, cmap='Blues', square=True)
        except:
            print(
                f"Error in generating ROC curve for {labels[i]}. "
                f"Dataset lacks enough examples."
            )
    plt.show()
    return auc_roc_vals, optimal_thresholds, sensitivity, specificity, accuracy, precision, recall, f1

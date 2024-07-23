

import numpy as np
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score, auc, multilabel_confusion_matrix, roc_curve, confusion_matrix, \
ConfusionMatrixDisplay , classification_report



def get_eval_para(true_value, prediction_value):
    cnf_matrix = confusion_matrix(true_value, prediction_value)
    # print(cnf_matrix)

    TN = cnf_matrix[0, 0]
    TP = cnf_matrix[1, 1]
    FN = cnf_matrix[1, 0]
    FP = cnf_matrix[0, 1]
    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)
    
    ACC = (TP+TN)/(TP+FN+TN+FP)
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = metrics.recall_score(true_value, prediction_value, average='macro')

    # Specificity or true negative rate
    TNR = np.mean(TN / (TN + FP))

    # Precision or positive predictive value
    F1_score = f1_score(true_value, prediction_value, average='macro')

    return ACC, TPR, TNR, F1_score, cnf_matrix

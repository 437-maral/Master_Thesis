import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import pandas
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score, auc, multilabel_confusion_matrix, roc_curve, confusion_matrix, \
    ConfusionMatrixDisplay
from sklearn.model_selection import KFold
from numpy import interp
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn

np.random.seed(1234)
random.seed(1234)

def pairwise_accuracy_score(Z, Y): 
    Z = np.asarray(Z, dtype=int)
    Y = np.asarray(Y, dtype=int)

    f1 = 1.0 * ((Z > 0) & (Y > 0)).sum(axis=1)  # numerator
    f2 = 1.0 * ((Z > 0) | (Y > 0)).sum(axis=1)  # denominator
    f1[f2 == 0] = 1.0
    f1[f2 > 0] /= f2[f2 > 0]

    return f1

def get_eval_para(true_value, prediction_value):
    cnf_matrix = multilabel_confusion_matrix(true_value, prediction_value)
    # print(cnf_matrix)

    TN = cnf_matrix[:, 0, 0]
    TP = cnf_matrix[:, 1, 1]
    FN = cnf_matrix[:, 1, 0]
    FP = cnf_matrix[:, 0, 1]
    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)

    ACC = accuracy_score(true_value, prediction_value)

    # Sensitivity, hit rate, recall, or true positive rate
    TPR = metrics.recall_score(true_value, prediction_value, average='macro')

    # Specificity or true negative rate
    TNR = np.mean(TN / (TN + FP))

    # Precision or positive predictive value
    F1_score = f1_score(true_value, prediction_value, average='macro')

    return ACC, TPR, TNR ,  F1_score ,cnf_matrix


uniform = lambda x,b: (np.abs(x/b) <= 1) and 1/2 or 0
triangle = lambda x,b: (np.abs(x/b) <= 1) and  (1 - np.abs(x/b)) or 0
gaussian = lambda x,b: (1.0/np.sqrt(2*np.pi))* np.exp(-.5*(x/b)**2) 
laplacian = lambda x,b: (1.0/(2*b))* np.exp(-np.abs(x/b)) 
epanechnikov = lambda x,b: (np.abs(x/b)<=1) and ((3/4)*(1-(x/b)**2)) or 0


###calculate distance
def pattern_layer(X,inp,kernel,sigma):
      k_values=[];
      for i,p in enumerate(X):
            edis = np.linalg.norm(p-inp); #find eucliden distance
            k = kernel(edis,sigma); ##use gaussion or other things 
            k_values.append(k)
      return k_values;

def summation_layer(k_values, Y_train, num_classes):
    summation_outputs = np.zeros(num_classes)
    
    if isinstance(Y_train, np.ndarray):
        Y_train = pd.DataFrame(Y_train)
        
    for cls in range(num_classes):
        # Get indices where the class label is 1
        class_indices = np.where(Y_train.iloc[:, cls] == 1)[0]
        
        if class_indices.size > 0:  # Ensure there are indices to prevent errors
            k_values = np.array(k_values);
            summation_outputs[cls] = np.sum(k_values[class_indices])
    return summation_outputs

def output_layer(avg_sum, num_classes):
    maxv = max(avg_sum)
    label = np.argmax(avg_sum)
    return label

def PNN_Train(X_train, Y_train, X_test, kernel, sigma, batch_size=256, epochs=2):
    # Number of classes
    num_classes = Y_train.shape[1]
    
    labels = []
    
    # If batch_size is not provided, use the entire dataset as a batch
    if batch_size is None:
        batch_size = X_train.shape[0]
    
    # Iterate over epochs
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        
        # Shuffle the training data at the beginning of each epoch
        indices = np.arange(X_train.shape[0])
        np.random.shuffle(indices)
        X_train_shuffled = X_train[indices]
        Y_train_shuffled = Y_train[indices]
        
        # Iterate over mini-batches
        for i in range(0, len(X_train_shuffled), batch_size):
            X_batch = X_train_shuffled[i:i+batch_size]
            Y_batch = Y_train_shuffled[i:i+batch_size]
            
            # Passing each sample observation in the mini-batch
            for s in X_test:
                k_values = pattern_layer(X_batch, s, kernel, sigma)
                summation_outputs = summation_layer(k_values, Y_batch, num_classes)
                
                # Calculate the average summation for each class
                class_counts = Y_batch.sum(axis=0)
                avg_sum = summation_outputs / class_counts
                
                label = output_layer(avg_sum, class_counts)
                labels.append(label)
    
    return np.array(labels)


def Kfold_PNN(X, Y, kernel, sigma):
    # K-Folds cross-validator
    kf = KFold(n_splits=10)
    label_num=Y.shape[1]
    # Metrics containers
    ACC = []
    SPE = []
    TPR = []
    F1_score = []
    ACC2 = []
    Pairwise_Acc = []
    Label_Acc = [[] for _ in range(Y.shape[1])]
    mean_Label_Acc = []
    Average_Label_Accuracy = []
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    fold = 1
    for train_index, test_index in kf.split(X, Y):
        print(f"{fold} fold:")
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        
        # Get prediction
        
        print("PNN  .........")
        pred_train = PNN_Train(X_train, Y_train, X_train, kernel, sigma)
        pred_test = PNN_Train(X_train, Y_train, X_test, kernel, sigma)
     
        # Train metrics
        print("Evalution .........")
        ACC_trn, TPR_trn, TNR_trn, F1_score_trn, cnf_matrix_trn = get_eval_para(Y_train, pred_train)
        print("ACC_train: " + str(ACC_trn))
        ACC2.append(ACC_trn)
        
        # Test metrics
        ACC_tst, TPR_tst, TNR_tst, F1_score_tst, cnf_matrix_tst = get_eval_para(Y_test, pred_test)
        print("ACC_test: " + str(ACC_tst))
        
        # Collect results
        ACC.append(ACC_tst)
        TPR.append(TPR_tst)
        F1_score.append(F1_score_tst)
        SPE.append(TNR_tst)

        test_avg_score = np.mean(pairwise_accuracy_score(Y_test, pred_test))
        Pairwise_Acc.append(test_avg_score)
        
        # Compute ROC curve and area the curve
        y_prob = pred_test  # You need actual probability predictions here
        fpr, tpr, thresholds = roc_curve(Y_test.ravel(), y_prob.ravel())
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        # Calculate Average_Label_Accuracy and Label_Acc
        each_label_value = []
        each_estimators_predict = []

        # Transpose to save each label true value in a list
        for i in range(len(Y_test[0])):  # Row num
            t = []
            for j in range(len(Y_test)):
                t.append(Y_test[j][i])
            each_label_value.append(t)

        # Transpose to save each label predict value in a list
        for i in range(len(pred_test[0])):  # Row num
            t = []
            for j in range(len(pred_test)):
                t.append(pred_test[j][i])
            each_estimators_predict.append(t)

        lbl_acc_sum = 0
        for i in range(label_num):
            lbl_acc = accuracy_score(each_label_value[i], each_estimators_predict[i])  # Each label accuracy in one fold
            Label_Acc[i].append(lbl_acc)
            lbl_acc_sum += lbl_acc

        Average_Label_Accuracy.append(lbl_acc_sum / label_num)
        fold += 1

    # Calculate mean auc
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)

    # K-fold mean value of label accuracy
    for i in range(label_num):
        mean_Label_Acc.append(np.mean(Label_Acc[i]))

    print("-------------- result ------------------")
    print("Train:" + str(np.mean(ACC2)))
    print("Test:" + str(np.mean(ACC)))
    print('ACC:', np.mean(ACC), 'TPR:', np.mean(TPR), 'SPE:', np.mean(SPE), 'F1_score:', np.mean(F1_score),
          'AUC:', mean_auc, 'Pairwise_Acc:', np.mean(Pairwise_Acc), 'Ave_Label_Acc:', np.mean(Average_Label_Accuracy))

    # Writing CSV Files
    columns_name = ['ACC', 'TPR', 'SPE', 'f1', 'auc', 'pairAcc', 'aveLabAcc'] + [f'label{i+1}' for i in range(Y.shape[1])]
    list_1 = [np.mean(ACC), np.mean(TPR), np.mean(SPE), np.mean(F1_score), mean_auc,
              np.mean(Pairwise_Acc), np.mean(Average_Label_Accuracy)] + mean_Label_Acc
    
    pd_data = pd.DataFrame(np.array(list_1).reshape(1, len(columns_name)), columns=columns_name)
    output_path = "./PNN_result.csv"
    pd_data.to_csv(output_path, index=False)

    # Plot ROC curve with matplotlib (only plot PNN)
    plt.figure("PNN")
    plt.plot((0, 1), (0, 1), c='#808080', lw=1, ls='--', alpha=0.7)  # Draw Diagonal Lines
    plt.plot(mean_fpr, mean_tpr, color='mediumblue', label=r'PNN (AUC = %0.4f)' % mean_auc, lw=1, alpha=.8)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.savefig("./PNN-roc-curve.pdf")
    plt.close("PNN")
    
    
    def main():
        dataset = pd.read_csv('kidney mlsmote result.csv')
        print("dataset.shape: " + str(dataset.shape))
        X = dataset.iloc[:, 8:]  # features
        Y = dataset.iloc[:, :8]  # labels
        X = (X - X.min()) / (X.max() - X.min())  # min-max normalization, X are mapped to the range 0 to 1.
        X = X.values
        Y= Y.values
        print("Running PNN ...")
        Kfold_PNN(X, Y, gaussian,0.05)
        
    if __name__ == '__main__':
        main()

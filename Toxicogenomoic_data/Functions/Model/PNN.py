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
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin

from Evalution_metrics import *

np.random.seed(1234)
random.seed(1234)

uniform = lambda x,b: (np.abs(x/b) <= 1) and 1/2 or 0
triangle = lambda x,b: (np.abs(x/b) <= 1) and  (1 - np.abs(x/b)) or 0
gaussian = lambda x,b: (1.0/np.sqrt(2*np.pi))* np.exp(-.5*(x/b)**2) 
laplacian = lambda x,b: (1.0/(2*b))* np.exp(-np.abs(x/b)) 
epanechnikov = lambda x,b: (np.abs(x/b)<=1) and ((3/4)*(1-(x/b)**2)) or 0

class PNN(BaseEstimator, ClassifierMixin):
    def __init__(self, sigma=0.1,kernel=gaussian):
        self.sigma = sigma
        self.kernel=kernel
    def fit(self, X, y):
        self.X_ = X
        self.y_ = y
        self.classes_ = np.arange(y.shape[1])  # Classes are assumed to be column indices for multi-label classification
        return self
    
    def _pdf(self, X):
        n_samples = X.shape[0]
        pdf_values = np.zeros((n_samples, self.X_.shape[0]))
        for i in range(n_samples):
            for j in range(self.X_.shape[0]):
                distance = np.linalg.norm(X[i] - self.X_[j])
                pdf_values[i, j] = self.kernel(distance, self.sigma)
        return pdf_values
    
    def predict(self, X):
        prob = np.zeros((X.shape[0], self.y_.shape[1]))
        pdf_values = self._pdf(X)
        for label in self.classes_:
            mask = (self.y_[:, label] == 1) #find respective label
            prob[:, label] = pdf_values[:, mask].sum(axis=1)
        return (prob > 0.75).astype(int)  # Threshold to get binary predictions
    
    
    def predict_probability(self, X):
        prob = np.zeros((X.shape[0], self.y_.shape[1]))
        pdf_values = self._pdf(X)
        epsilon = 1e-8
        for label in self.classes_:
            mask = (self.y_[:, label] == 1)
            prob[:, label] = pdf_values[:, mask].sum(axis=1)
        prob = np.where(prob < epsilon, epsilon, prob)  # Avoid zero probabilities
        prob /= prob.sum(axis=1, keepdims=True)  # Normalize to get probabilities
        return prob

def Kfold_PNN(X, Y,sigma=0.05):
    '''Run Model in 10 folds'''
    
    # K-Folds cross-validator
    kf = KFold(n_splits=10)
    
    Label_num=Y.shape[1]
    humming_Loss=[]
    Rank_Loss=[]
  
    ACC = []
    SPE = []
    TPR = []
    F1_score = []
    
    Pairwise_Acc = []
    
    scoring_fn = pairwise_accuracy_score

    Label_Acc = [[] for i in range(Y.shape[1])]  # expect to access the performance of EACH LABEL
    mean_Label_Acc = []  # k fold average
    Average_Label_Accuracy = []

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    fold = 1
    for train_index, test_index in kf.split(X, Y):
        print(str(fold)+" fold:")
        
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        # Initialize model to be used
        clf = PNN(sigma=0.05,kernel=gaussian)
        # train model
        # predict
        clf.fit(X_train, Y_train)
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_probability(X_test)

        # train set
        trn_pred = clf.predict(X_train)
        
        
        '''Different evalution metrcis are used'''
        #humming_Loss
        #Rank_Loss
        #accuracy_socre 
        #classification_report
        #confusion matrics 
        
        humming_Loss_trn=hamming_loss(Y_train, trn_pred)
        print("Humming_trn: " + str(humming_Loss_trn))
        
        humming_Loss_test=hamming_loss(Y_test,  y_pred)
        print("Humming_test: " + str(humming_Loss_test))
        
        Randking_loss_trn=ranking_loss(Y_train, trn_pred)
        print("Ranking_loss_trn: " + str(Randking_loss_trn))
        
        Randking_loss_tst=ranking_loss(Y_test,  y_pred)
        print("Ranking_loss_test: " + str(Randking_loss_tst))
        
        
        
        ###for each fold 
        #report 
        
        ACC_trn, TPR_trn,TNR_trn, F1_score_trn , cnf_matrix_trn = get_eval_para(Y_train, trn_pred)
        print("ACC_trn: " + str(ACC_trn))
        
        
        
        
        ACC_tst, TPR_tst,TNR_tst, F1_score_tst ,  cnf_matrix_tst= get_eval_para(Y_test, y_pred)
        print("ACC_test: " + str(ACC_tst))
        
        
        #save result 
        
        humming_Loss.append(humming_Loss_test)
        Rank_Loss.append(Randking_loss_tst)
        ACC.append(ACC_tst)
        TPR.append(TPR_tst)
        SPE.append(TNR_tst)
        F1_score.append(F1_score_tst)

        # calculate pairwise_accuracy_score
        test_pred = clf.predict(X_test)
        test_avg_score = np.mean(scoring_fn(Y_test, test_pred))
        print("Acc_Pair_test: " + str( test_avg_score))
        Pairwise_Acc.append(test_avg_score)
        
        
        
        # compute ROC curve and area the curve
        y_prob = y_prob
        fpr, tpr, thresholds = roc_curve(Y_test.ravel(), y_prob.ravel())
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        # calculate Average_Label_Accuracy and Label_Acc
        y_pred = y_pred
        each_label_value = []
        each_estimators_predict = []

        # transposition, save each label true value in a list
        for i in range(len(Y_test[0])):  # row num
            t = []
            for j in range(len(Y_test)):
                t.append(Y_test[j][i])
            each_label_value.append(t)

        # transposition, save each label predict value in a list
        for i in range(len(y_pred[0])):  # row num
            t = []
            for j in range(len(y_pred)):
                t.append(y_pred[j][i])
            each_estimators_predict.append(t)

        lbl_acc_sum = 0
        for i in range(0, Label_num, 1):
            lbl_acc = accuracy_score(each_label_value[i], each_estimators_predict[i])  # each label accuracy in one fold
            Label_Acc[i].append(lbl_acc)
            lbl_acc_sum = lbl_acc_sum + lbl_acc

        Average_Label_Accuracy.append(lbl_acc_sum / Label_num)

        fold = fold + 1

    # calculate mean auc
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)

    # k-fold mean value of label accuracy
    for i in range(0, Label_num, 1):
        mean_Label_Acc.append(np.mean(Label_Acc[i]))

    print("-------------- result ------------------")
    print("Test:" + str(np.mean(ACC)))
    print('ACC:', np.mean(ACC), 'Hamming_Loss:', np.mean(humming_Loss), 'Rank_Loss', np.mean(Rank_Loss),
          'AUC:', mean_auc, 'Pairwise_Acc:', np.mean(Pairwise_Acc), 'Ave_Label_Acc:', np.mean(Average_Label_Accuracy))

    # Writing CSV Files
    columns_name = [ 'ACC', 'TPR','spe', 'F1_score','AUC', 'Hamming_Loss', 'Rank_Loss', 'pairAcc', 'aveLabAcc',
                    'label1', 'label2', 'label3', 'label4', 'label5', 'label6', 'label7', 'label8']
    list_1 = [np.mean(ACC), np.mean(TPR), np.mean(SPE), np.mean(F1_score), mean_auc,np.mean(humming_Loss), np.mean(Rank_Loss),
              np.mean(Pairwise_Acc), np.mean(Average_Label_Accuracy)] + mean_Label_Acc
    
    pd_data = pd.DataFrame(np.array(list_1).reshape(1, 17), columns=columns_name)
    
    output_path = "./" + "PNN_result.csv"
    pd_data.to_csv(output_path, index=False)

    # Plot ROC curve with matplotlib (only plot arch001)
    plt.figure("PNN")
    plt.plot((0, 1), (0, 1), c='#808080', lw=1, ls='--', alpha=0.7)  # Draw Diagonal Lines
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color='mediumblue', label=r'PNN(Area=%0.4f)' % mean_auc, lw=1, alpha=.8)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.savefig("./" + "-PNN-roc-curve.pdf")
    # plt.show()
    plt.close("PNN")

    
if __name__ == "__main__":
    
    # Load the iris dataset
    dataset = pd.read_csv('kidney mlsmote result.csv')
    print("dataset.shape: " + str(dataset.shape))
    # split the features-X and class labels-y
    X = dataset.iloc[:, 8:]  # features
    Y = dataset.iloc[:, :8]  # labels
    # Normalise the data
    X = (X - X.min()) / (X.max() - X.min())  # min-max normalization, X are mapped to the range 0 to 1.
    X = X.values
    Y = Y.values
    Kfold_PNN(X, Y)

import numpy as np
import matplotlib.pyplot as plt
import random
import sys
import pandas as pd
np.float = float 

###metrics
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score, auc, multilabel_confusion_matrix, roc_curve, confusion_matrix, \
    ConfusionMatrixDisplay
from sklearn.model_selection import KFold

from Evalution_metrics import *

##model 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf , keras
##optimizer
from keras.optimizers import Nadam, Adam, Optimizer, Adadelta
from numpy import interp

from tensorflow.keras.regularizers import L1,L2,L1L2




np.random.seed(1234)
random.seed(1234)



        
'''This simpelest model for prediction of multi-label classification'''

def DNN_tns(dim_no,class_no,l2w=1e-5):
    if l2w is None:
            regularizer = None
    else:
        regularizer = L2(l2w)
    # create simple mlp
    model = Sequential()
    model.add(Dense(128, input_dim=dim_no, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(class_no, activation='sigmoid'))
    
    
    return model
class GeneDNN:
    """
    Genomic Deep Learning Model For Multi Label Classification 

    Parameters
    ----------
    n_features: int
    n_labels: int
    scoring_fn:
    b: int, optional, default=3
    nb_epochs: int
        number of epochs to train, to save time, we just train 100 epochs
        train the training data n times
    batch_size: int, optional, default=256

    Attributes
    ----------
    model : keras.models.Model instance
    """
    def __init__(self, n_features, n_labels, l2w,learning_rate=0.001, decay=0., b=3, batch_size=256, nb_epochs=2, optimizer='Adam'):
        self.batch_size = batch_size
        self.b = b
        self.nb_epochs = nb_epochs
        self.l2w = l2w
        self.n_labels = n_labels
        self.n_features = n_features
        self.model = DNN_tns(n_features, n_labels)
        self.model.summary()
        
        if optimizer == 'Adam':
            self.optimizer = Adam(learning_rate=learning_rate, decay=decay)
        elif optimizer == 'Adadelta':
            self.optimizer = Adadelta(learning_rate=learning_rate, decay=decay)
        elif optimizer is None:
            self.optimizer = Nadam()
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")

        self.model.compile(optimizer=optimizer, loss='binary_crossentropy')


    def train(self, X, Y):
        self.model.fit(X, Y, epochs=self.nb_epochs, batch_size=self.batch_size)
        
    def evaluate(self, X, Y):
        return self.model.evaluate(X, Y)

    def predict(self, X):
        return (self.model.predict(X) > 0.75).astype(float)

    def predict_probability(self, X):
        return self.model.predict(X)
    
    
def Kfold_GNN(X, Y):
    '''Run Model in 10 folds'''
    
    # K-Folds cross-validator
    kf = KFold(n_splits=10)

    # l2 parameter
    param = 10 ** -5

    feat_num = X.shape[1]
    label_num = Y.shape[1]
    
    
    ###label names 
    label_names=['Necrosis', 'Dilatation', 'Dilatation, cystic', 'Cast,hyaline',
       'Regeneration', 'Cyst', 'Cellular infiltration, lymphocyte',
       'Change, basophilic']


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
        clf = GeneDNN(n_features=feat_num, n_labels=label_num,
                         b=3,nb_epochs=200,l2w=param,batch_size=64)
        # train model
        clf.train(X_train, Y_train)
        # predict
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
        save_classification_report(Y_test,  y_pred,label_names,fold,output_dir='./Classification_Report')
        #-----------Confusion_Matrix
        cnf_plot(Y_test,  y_pred,label_names,fold,output_dir='./Confusion_Matrix')

     
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
        for i in range(0, label_num, 1):
            lbl_acc = accuracy_score(each_label_value[i], each_estimators_predict[i])  # each label accuracy in one fold
            Label_Acc[i].append(lbl_acc)
            lbl_acc_sum = lbl_acc_sum + lbl_acc

        Average_Label_Accuracy.append(lbl_acc_sum / label_num)

        fold = fold + 1

    # calculate mean auc
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)

    # k-fold mean value of label accuracy
    for i in range(0, label_num, 1):
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
    
    output_path = "./" + "DNN_result.csv"
    pd_data.to_csv(output_path, index=False)

    # Plot ROC curve with matplotlib (only plot arch001)
    plt.figure("DNN")
    plt.plot((0, 1), (0, 1), c='#808080', lw=1, ls='--', alpha=0.7)  # Draw Diagonal Lines
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    plt.plot(mean_fpr, mean_tpr, color='mediumblue', label=r'DNN(Area=%0.4f)' % mean_auc, lw=1, alpha=.8)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc='lower right')
    plt.savefig("./" + "-DNN-roc-curve.pdf")
    # plt.show()
    plt.close("DNN")

    def main():
        dataset = pd.read_csv('kidney mlsmote result.csv')
        print("dataset.shape: " + str(dataset.shape))
        X = dataset.iloc[:, 8:]  # features
        Y = dataset.iloc[:, :8]  # labels
        X = (X - X.min()) / (X.max() - X.min())  # min-max normalization, X are mapped to the range 0 to 1.
        X = X.values
        Y = Y.values
        print("Running DNN ...")
        Kfold_GNN(X, Y)
        
    if __name__ == '__main__':
        main()

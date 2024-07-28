import numpy as np
import matplotlib.pyplot as plt
import random
import sys
import scipy.sparse as sp
import pandas as pd
np.float = float 


from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score, auc, multilabel_confusion_matrix, roc_curve, confusion_matrix, \
    ConfusionMatrixDisplay
from sklearn.model_selection import KFold

##model 
import torch
from torch import nn

##optimizer
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from numpy import interp

np.random.seed(1234)
random.seed(1234)

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

def pairwise_accuracy_score(Z, Y): 
    Z = np.asarray(Z, dtype=int)
    Y = np.asarray(Y, dtype=int)

    f1 = 1.0 * ((Z > 0) & (Y > 0)).sum(axis=1)  # numerator
    f2 = 1.0 * ((Z > 0) | (Y > 0)).sum(axis=1)  # denominator
    f1[f2 == 0] = 1.0
    f1[f2 > 0] /= f2[f2 > 0]

    return f1

class GeneDataset(Dataset):
    def __init__(self, features, labels=None):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32) if labels is not None else None

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.labels is not None:
            return self.features[idx], self.labels[idx]
        else:
            return self.features[idx]
        
        
        
'''This simpelest model for prediction of multi-label classification'''

class DNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DNN, self).__init__()
        # Define the layers
        self.fc1 = nn.Linear(input_size, 128)  
        self.fc2 = nn.Linear(128, 128)              # Dropout layer with 10% dropout rate       
        self.fc3 = nn.Linear(128, output_size) # Output layer

    def forward(self, x):
        # Forward pass through the network
        x = torch.relu(self.fc1(x))         
        x = torch.relu(self.fc2(x))                          
        x = torch.sigmoid
        
class GeneDNN:
    def __init__(self, n_features: int, n_labels: int, l2w: float,
                 learning_rate=0.001, decay=0., b: int = 3, batch_size: int = 256,
                 nb_epochs: int = 2, optimizer='Adam'):
        self.batch_size = batch_size
        self.b = b
        self.nb_epochs = nb_epochs
        self.l2w = l2w
        self.n_labels = n_labels
        self.n_features = n_features
        self.model = DNN(n_features, n_labels)
        #self.patience = patience

        if optimizer is None:
            raise ValueError('You should provide an argument for the optimizer')
        elif optimizer == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=decay)
        elif optimizer == 'Adadelta':
            self.optimizer = optim.Adadelta(self.model.parameters(), lr=learning_rate, weight_decay=decay)
        elif optimizer == 'Adagrad':
            self.optimizer = optim.Adagrad(self.model.parameters(), lr=learning_rate, weight_decay=decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer}")

    def train(self, X, Y, val_X=None, val_Y=None):
        nb_epochs = self.nb_epochs
        optimizer = self.optimizer
        model = self.model
        self.criterion = nn.BCELoss()
        
        dataset = GeneDataset(X, Y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)


        for epoch in range(nb_epochs):
            model.train()
            running_loss = 0.0
            for inputs, labels in dataloader:
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
            print(f'Epoch {epoch + 1}, Loss: {running_loss / len(dataloader)}')
        
        print('Finished Training')

    def predict(self, X):
        model = self.model
        model.eval()
        predictions = []
        dataloader = DataLoader(GeneDataset(X), batch_size=self.batch_size, shuffle=False)
        with torch.no_grad():
            for inputs in dataloader:
                outputs = model(inputs)
                predicted = (outputs > 0.75).float()
                predictions.append(predicted)
        return torch.cat(predictions, dim=0)
    
    def predict_probability(self, X):
        model = self.model
        model.eval()
        probabilities = []
        dataloader = DataLoader(GeneDataset(X), batch_size=self.batch_size, shuffle=False)
        with torch.no_grad():
            for inputs in dataloader:
                outputs = model(inputs)
                #don't change to specefic format 
                probabilities.append(outputs)
        return torch.cat(probabilities, dim=0)
    
    
    
    def Kfold_GNN(X, Y):
        # K-Folds cross-validator
    kf = KFold(n_splits=10)

    # l2 parameter
    param = 10 ** -5

    feat_num = X.shape[1]
    label_num = Y.shape[1]

    ACC = []
    SPE = []
    TPR = []
    F1_score = []
    ACC2 = []
    
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

        # Initialize the classifier to be use
        clf = GeneDNN(n_features=feat_num, n_labels=label_num, nb_epochs=200,
                         l2w=param,b=3)
        # train model
        clf.train(X_train, Y_train)
        # predict
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_probability(X_test)

        # train set
        trn_pred = clf.predict(X_train)
     
        ACC_trn, TPR_trn,TNR_trn, F1_score_trn , cnf_matrix_trn = get_eval_para(Y_train, trn_pred)
        print("ACC_trn: " + str(ACC_trn))
        ACC2.append(ACC_trn)
        ACC_tst, TPR_tst,TNR_tst, F1_score_tst ,  cnf_matrix_tst= get_eval_para(Y_test, y_pred)
        print("ACC_tst: " + str(ACC_tst))
        
        ###COLLECT RESULT
        ACC.append(ACC_tst)
        TPR.append(TPR_tst)
        F1_score.append(F1_score_tst)
        SPE.append(TNR_tst)

        # calculate pairwise_accuracy_score
        test_pred = clf.predict(X_test)
        test_avg_score = np.mean(scoring_fn(Y_test, test_pred))
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
    print("train:" + str(np.mean(ACC2)))
    print("test:" + str(np.mean(ACC)))
    print('ACC:', np.mean(ACC), 'TPR:', np.mean(TPR), 'SPE', np.mean(SPE), 'F1_score:', np.mean(F1_score),
          'AUC:', mean_auc, 'Pairwise_Acc:', np.mean(Pairwise_Acc), 'Ave_Label_Acc:', np.mean(Average_Label_Accuracy))

    # Writing CSV Files
    columns_name = [ 'ACC', 'TPR', 'spe', 'f1', 'auc', 'pairAcc', 'aveLabAcc',
                    'label1', 'label2', 'label3', 'label4', 'label5', 'label6', 'label7', 'label8']
    list_1 = [np.mean(ACC), np.mean(TPR), np.mean(SPE), np.mean(F1_score), mean_auc,
              np.mean(Pairwise_Acc), np.mean(Average_Label_Accuracy)] + mean_Label_Acc
    
    pd_data = pd.DataFrame(np.array(list_1).reshape(1, 15), columns=columns_name)
    
    output_path = "./" + "DNN_result.csv"
    pd_data.to_csv(output_path, index=False)

 
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
import numpy as np 
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import accuracy_score, f1_score, auc, multilabel_confusion_matrix, roc_curve, confusion_matrix, \
    ConfusionMatrixDisplay , classification_report


def hamming_loss(y_true, y_pred):
    """
    Calculate the Humming Loss for multi-label classification.

    Parameters:
    y_true (tf.Tensor): A tensor of true labels of shape (batch_size, num_labels).
    y_pred (tf.Tensor): A tensor of predicted scores of shape (batch_size, num_labels).

    Returns:
    tf.Tensor: The ranking loss.
    """
    num_labels = y_true.shape[1]
    num_samples = y_true.shape[0]
    incorrect_labels = np.sum(y_true != y_pred)
    # Calculate Hamming Loss
    result = incorrect_labels / (num_labels * num_samples)
    return result




def ranking_loss(y_true, y_pred):
    """
    Calculate the ranking loss for multi-label classification.

    Parameters:
    y_true (np.ndarray): An array of true labels of shape (batch_size, num_labels).
    y_pred (np.ndarray): An array of predicted scores of shape (batch_size, num_labels).

    Returns:
    float: The ranking loss.
    """
    y_true = y_true.astype(np.float32)
    differences = y_pred[:, None, :] - y_pred[:, :, None]
    partial_losses = np.maximum(0.0, 1 - differences)
    true_mask = y_true[:, None, :] * (1 - y_true[:, :, None])
    loss = partial_losses * true_mask
    ranking_loss_value = np.sum(loss) / np.sum(true_mask)
    
    return ranking_loss_value



def pairwise_accuracy_score(Z, Y): 
    Z = np.asarray(Z, dtype=int)
    Y = np.asarray(Y, dtype=int)

    f1 = 1.0 * ((Z > 0) & (Y > 0)).sum(axis=1)  # numerator
    f2 = 1.0 * ((Z > 0) | (Y > 0)).sum(axis=1)  # denominator
    f1[f2 == 0] = 1.0
    f1[f2 > 0] /= f2[f2 > 0]

    return f1



def save_classification_report(y_true, y_pred, target_names, fold, filename='classification_report.csv', output_dir=None):
    """
    Generate and save the classification report to a CSV file.

    Parameters:
    y_true (numpy array): A 2D array of true labels.
    y_pred (numpy array): A 2D array of predicted labels.
    target_names (list): List of target names for the labels/for each pathological finding.
    fold (int): Fold number for identification in filename.
    filename (str): Name of the CSV file to save the report.
    output_dir (str): Directory to save the CSV file.

    Returns:
    None
    """
    # Generate the classification report
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    # Determine the file path
    if output_dir is not None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        file_path = os.path.join(output_dir, f"{filename.replace('.csv', '')}_fold{fold}.csv")
    else:
        file_path = f"{filename.replace('.csv', '')}_fold{fold}.csv"

    # Save the DataFrame to a CSV file
    report_df.to_csv(file_path, index=True)
    print(f"Classification report saved to '{file_path}'")



def cnf_plot(Y_test, y_pred, name_labels, fold, output_dir=None):
    """
    Create confusion matrix for each pathological finding.

    Parameters:
    Y_test (numpy array): A 2D array of true labels.
    y_pred (numpy array): A 2D array of predicted labels.
    name_labels (list of str): The names of pathological findings.
    fold (int): Fold number for identification in filename.
    output_dir (str): Directory to save the plots.

    Returns:
    None
    """
    fig, axes = plt.subplots(2, 4, figsize=(25, 10))
    axes = axes.ravel()
    
    font_size = {'size': '24'}  # Adjust to fit
    tick_size = 21

    for i in range(len(name_labels)):
        cm = confusion_matrix(Y_test[:, i], y_pred[:, i])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
        disp.plot(ax=axes[i], values_format='.4g', cmap='YlGn')
        disp.ax_.set_title(name_labels[i], fontdict=font_size)
        disp.ax_.tick_params(axis='both', labelsize=tick_size)
        disp.ax_.set_xlabel("Predicted label", fontdict=font_size)
        disp.ax_.set_ylabel("True label", fontdict=font_size)
        
        if i < 4:
            disp.ax_.set_xlabel('')
        if i % 4 != 0:
            disp.ax_.set_ylabel('')
        disp.im_.colorbar.remove()

    plt.subplots_adjust(wspace=0.1, hspace=0.3)
    cbar1 = fig.colorbar(disp.im_, ax=axes)
    cbar1.ax.tick_params(labelsize=tick_size)
        
    if output_dir is not None:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        file_path = os.path.join(output_dir, f"confusion_matrix_fold{fold}.pdf")
        plt.savefig(file_path)
    else:
        plt.show()

    plt.close(fig)  # Close the figure to release memory

               
        
        
def get_eval_para(y_true,y_pred):
    """
    Parameters:
    y_true (numpy array): A 2D array of true labels.
    y_pred (numpy array): A 2D array of predicted labels.
    Returns:
    evalution metrcis
    """
    cnf_matrix = multilabel_confusion_matrix(y_true,y_pred)
    # print(cnf_matrix)

    TN = cnf_matrix[:, 0, 0]
    TP = cnf_matrix[:, 1, 1]
    FN = cnf_matrix[:, 1, 0]
    FP = cnf_matrix[:, 0, 1]
    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)


    ACC = accuracy_score(y_true,y_pred)

    # Sensitivity, hit rate, recall, or true positive rate
    TPR= metrics.recall_score(y_true,y_pred ,average='macro')

    # Specificity or true negative rate
    TNR = np.mean(TN / (TN + FP))

    # Precision or positive predictive value
    F1_score = f1_score(y_true,y_pred ,average='macro')

    return ACC, TPR, TNR ,  F1_score ,cnf_matrix


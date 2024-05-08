import os
import re
from collections import Counter
from pathlib import Path
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import interp
from imblearn.over_sampling import SMOTE 
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from Fingerprints import Fingerprints




def preparing_train_data(data,method):
    
    
    Y_train= data['Nephro'].values

    #features_train
    smiles_train=data['smiles'].tolist() 
    
    #modified the format of chemical structure 
    
    fingerprints=Fingerprints(data['smiles'])
    
    smiles_train= fingerprints.clean(smiles_train)
    #change to fingerprint
    simles_train_list= fingerprints.smiles_to_fingerprint(smiles_train,method=method)
    
    #dataframe
    smiles_train_to_dataframe= pd.DataFrame(simles_train_list)
    #preparing features
    smiles_train_to_dataframe=smiles_train_to_dataframe.iloc[:,0:].values

    
    return smiles_train_to_dataframe, Y_train


def preparing_test_data(data,method):
    
    #Label_test
    Y_test= data['Nephro'].values
    #features test
    smiles_test=data['smiles'].tolist()
    fingerprints=Fingerprints(data['smiles'])
    smiles_test= fingerprints.clean(smiles_test)
    
    simles_test_list= fingerprints.smiles_to_fingerprint(smiles_test,method=method)
    smiles_test_to_dataframe= pd.DataFrame(simles_test_list)
    smiles_test_to_dataframe=smiles_test_to_dataframe.iloc[:,0:].values
    
    return smiles_test_to_dataframe , Y_test


def sampling_techniques(chemical,fingerprint,method):
    
    X, Y = preparing_train_data(chemical,fingerprint)
    
    if method == 'smote':
        smote = SMOTE()
        x_sampled, y_sampled = smote.fit_resample(X, Y)
    elif method == 'undersampling':
        under = RandomUnderSampler()
        x_sampled, y_sampled = under.fit_resample(X, Y)
    elif method == 'oversampling':
        over = RandomOverSampler()
        x_sampled, y_sampled = over.fit_resample(X, Y)
    else:
        raise ValueError("Unsupported sampling method: {}".format(method))
    
    return x_sampled, y_sampled
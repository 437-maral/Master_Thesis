#author@Masoumeh
import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem
import rdkit.Chem.MACCSkeys as mk
import os
import numpy
import pickle
import warnings
import pandas as pd
from  Similarity_based_appraoch import *
import tensorflow as tf , keras


def predict_withdrawn_drug(chemo_smiles):
    '''This function is designed to give us the prediction score for unknown data'''
    # Preparing SMILES for machine learning model
    compounds = chemo_smiles['compounds'].tolist()
    smiles = chemo_smiles['smiles'].tolist()
    chemo_fingerprints_bits, chemo_fingerprints = preparing_data(chemo_smiles, 'maccs')
    
    chemo_predict = []
    chemo_score = []
    compound_status = []
    
    current_dir = os.getcwd()
    
    # Load the Random Forest model
    with open(os.path.join(current_dir, 'final_RF_best_model.pkl'), 'rb') as file:
        loaded_rf = pickle.load(file)
        print("Random Forest loaded successfully.")
    
    # Iterate over each compound's fingerprint
    for idx in range(chemo_fingerprints_bits.shape[0]):
        each_compound_fingerprint =  chemo_fingerprints_bits[idx, :].reshape(1, -1)
        
        # Prediction
        y_ch = loaded_rf.predict(each_compound_fingerprint)
        confidence_scores = loaded_rf.predict_proba(each_compound_fingerprint)
        
        if y_ch[0] == 0:
            status = 'Inactive_Compound'
            confidence_score_ch = confidence_scores[0][0]
        elif y_ch[0] == 1:
            status = 'Active_Compound'
            confidence_score_ch = confidence_scores[0][1]
        else:
            status = 'Unknown'
            confidence_score_ch = None
        
        # Append results
        chemo_predict.append(y_ch[0])
        chemo_score.append(confidence_score_ch)
        compound_status.append(status)
    
    # Create DataFrame
    result = pd.DataFrame({
        'Compounds': compounds,
        'Smiles': smiles,
        'Predict_Score': chemo_predict,
        'Confidence_Score': chemo_score,
        'Status': compound_status
    })
    
    # Save to CSV
    result.to_csv(os.path.join(current_dir, 'Chemical_Score.csv'), index=False)

                
    
    
    
    
    
    
    
    #
        
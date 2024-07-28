import pandas as pd
import numpy as np

#delete unuseful columns

def remove_unuseful_columns(data, pattern):
    '''This function removes columns based on a specified pattern and renames columns by modifying their names.'''
    for column in data.columns[2:]:
        parts = column.split()
        if len(parts) > 4:
        
            
            if parts[4] == pattern:
                data = data.drop(columns=column)
            
        
            # Change the column name
            new_column_name = "".join(parts[:2]) + " " + " ".join(parts[2:])
            data.rename(columns={column: new_column_name}, inplace=True)
            
                
        if len(parts) > 3 and parts[3] == pattern:
            data = data.drop(columns=column)
    return data 


def Remove_incomplete_drugs(data):
    '''remove drugs whose dose is not complete'''
    #go to each drugs if the drug deosn't have 3 levels it's needed to be excluded
    
    included_drugs = []

    processed_drugs = {}
    
    for col_name in data.columns[2:]:
        drug= col_name.split()[0]
        dose= col_name.split()[1]
        
        if drug not in processed_drugs:
            processed_drugs[drug] = []
            
        processed_drugs[drug].append(dose)
        
        if set(processed_drugs[drug]) == {'L', 'M', 'H'}:
            
            included_drugs.append(drug)
      
    return included_drugs


def omitNA(data,save=False):
    '''save argument is needed to be sure whether we should save excluded genes or not'''
    # Step 1: Filter out rows where the second column is NaN
    index = [i for i in range(len(data)) if not pd.isna(data.iloc[i, 1])]
    result1 = data.iloc[index, :]
    
    
    mark = []
    NA_rows=[]
    for row in range(len(result1)):
        NA = False  
        for col in range(2, result1.shape[1]):  
            if pd.isna(result1.iloc[row, col]):
                NA = True
                NA_rows.append(row)
                excluded_genes=result1.iloc[NA_rows,:]
                break
        if NA == False:
            mark.append(row)
                
    result2 = result1.iloc[mark, :]
    
    if save:
        excluded_genes.to_csv('Normalized_Genes_With_NA.csv')

   
    
    return result2 ,excluded_genes



        


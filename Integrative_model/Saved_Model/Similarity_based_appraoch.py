import pandas as pd
import numpy as np
import pubchempy as pcp
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import AllChem , MACCSkeys

def get_smiles_from_name(compounds):
    '''This function aims to find smiles/chemical strctures based on compounds'''
    smiles_list = []
    for com in compounds:
        result = pcp.get_compounds(com, 'name')
        count = 1  # Initialize count for each compound
        for res in result:
            smiles_list.append({
                'compounds': com,
                'smiles': res.isomeric_smiles,
                'count': count
            })
            count += 1  # Increment count for each SMILES found
    return smiles_list


def clean(elements_list):
    '''if there is a space between compounds'''
    cleaned_elements = []
    for element in elements_list:
        # Remove spaces and connect broken parts
        cleaned_element = ''.join(element.split())
        cleaned_elements.append(cleaned_element)
    return cleaned_elements


def smiles_to_fingerprint(smiles_list, method, n_bits=2048):
    '''Using different fingerprints converts chemical structures to binary vectors'''
    fingerprints = []
   
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            if method == 'maccs':
                # MACCS keys have a fixed length of 167 bits
                fp = MACCSkeys.GenMACCSKeys(mol)
                fingerprints.append(fp)
            elif method == 'morgan':
                fpg = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
                fp = fpg.GetFingerprint(mol)
                fingerprints.append(fp)
            elif method == 'atom_pairs':
                apgen = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=n_bits)
                fp = apgen.GetFingerprint(mol)
                fingerprints.append(fp)
            elif method == 'rdkit':
                rdkgen = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=n_bits)
                fp = rdkgen.GetFingerprint(mol)
                fingerprints.append(fp)
            elif method == 'topological_torsions':
                ttgen = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=n_bits)
                fp = ttgen.GetFingerprint(mol)
                fingerprints.append(fp)
            else:
                raise ValueError(f"Unsupported method: {method}")
        else:
            print(f"Warning: Could not parse SMILES string '{smiles}'")

    return fingerprints


def preparing_data(data,method):
    '''Prepared data fo similarity comparison'''
    #features_train
    smiles_train=data['smiles'].tolist() 
    #modified the format of chemical structure 
    smiles_train= clean(smiles_train)
    #change to fingerprint
    simles_train_list= smiles_to_fingerprint(smiles_train,method=method)
    Fingerprints_gen_dic = {
    'compound': data['compounds'],
    'fingerprint': simles_train_list}
    smiles_train_to_dataframe= pd.DataFrame(np.array(simles_train_list))
    #This is ultimate features
    smiles_train_to_dataframe=smiles_train_to_dataframe.iloc[:,0:].values
    
    
    
    return smiles_train_to_dataframe,Fingerprints_gen_dic






def get_highest_similarity_pairs(fingerprints_genomic_data, fingerprints_chemical_data):
    '''Select compounds based on similarity from chemical data to give insights about the toxicity of compounds.'''
    unique_compounds = set()
    high_similarity_pairs = []
    
    n_query = len(fingerprints_chemical_data['fingerprint'])
    n_reference = len(fingerprints_genomic_data['fingerprint'])
    # To store similarity scores
    similarity_matrix = np.zeros((n_query, n_reference))
    compound_max_similarity = {}
    
    compound_names = fingerprints_genomic_data['compound']

    for i, query_chem in enumerate(fingerprints_chemical_data['fingerprint']):
        # Iterate over reference fingerprints
        for j, compare_fingerprint in enumerate(fingerprints_genomic_data['fingerprint']):
            similarity = DataStructs.TanimotoSimilarity(query_chem, compare_fingerprint)
            similarity_matrix[i, j] = similarity

            if similarity > 0.6:
                compound_name = compound_names[j]
                unique_compounds.add(compound_name)

                if compound_name not in compound_max_similarity or similarity > compound_max_similarity[compound_name]:
                    compound_max_similarity[compound_name] = similarity

    for compound_name, max_similarity in compound_max_similarity.items():
        high_similarity_pairs.append((max_similarity, compound_name))

    return high_similarity_pairs




def combined_with_genomic_data(genomic_data,high_similarity_pairs):
    
    compounds_list=[item[1] for item in high_similarity_pairs]
    
    selected_columns = [col for col in genomic_data.columns[2:]  if any(col.split()[0] in compound and col.split()[1] in ['L', 'H']for compound in compounds_list)]
    
    selected_genomic_data= list(genomic_data.columns[:2]) + selected_columns
    filtered_genomic_data= genomic_data[selected_genomic_data]
    
    filtered_toxicology_transpose=filtered_genomic_data.T
    filtered_toxicology_transpose.columns = filtered_toxicology_transpose.iloc[1]
    filtered_toxicology_transpose= filtered_toxicology_transpose.iloc[2:]

    return filtered_toxicology_transpose








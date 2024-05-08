from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import AllChem , MACCSkeys
import numpy as np


class Fingerprints(object):
    def __init__(self, smiles):
        
        self.smiles=smiles
        
        
    def clean(self,smiles):
        cleaned_elements = []
        for element in smiles:
            # Remove spaces and connect broken parts
            cleaned_element = ''.join(element.split())
            cleaned_elements.append(cleaned_element)
            
        return cleaned_elements
    
    
    
    def smiles_to_fingerprint(self,smiles, method, n_bits=2048):
        
        fingerprints = []
        invalid_smiles = []
        for smiles in smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                if method == 'maccs':
                    fp = MACCSkeys.GenMACCSKeys(mol) ##it has different length
                elif method == 'morgan':
                    fpg = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=n_bits)
                    fp = fpg.GetFingerprint(mol)
                elif method == 'Atom pairs':
                    apgen = rdFingerprintGenerator.GetAtomPairGenerator(fpSize=n_bits)
                    fp = apgen.GetFingerprint(mol)
                elif method == 'RDKit':
                    rdkgen = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=n_bits)
                    fp = rdkgen.GetFingerprint(mol)
                elif method == 'Topological torsions':
                    ttgen = rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=n_bits)
                    fp = ttgen.GetFingerprint(mol)
                else:
                    raise ValueError("Unsupported fingerprint method. Use 'morgan', 'maccs', 'Atom pairs', 'RDKit', or 'Topological torsions'.")
                fingerprints.append(np.array(fp))
                
        return fingerprints
        
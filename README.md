# Master_Thesis


Abstract:
This master's thesis explores the development of a computational model for predicting nephrotoxicity induced by small molecules. 

Initially, various machine learning algorithms were employed to investigate whether chemical structures could effectively discern nephrotoxic compounds. Among these algorithms, the Random Forest model demonstrated superior predictive performance. To address the challenge of imbalanced datasets, sampling techniques such as SMOTE, oversampling, and undersampling were applied. SMOTE and oversampling techniques proved successful in mitigating dataset imbalances, yielding promising results when combined with the Random Forest model during cross-validation and evaluation on unseen data.

In the subsequent phase of the study, pathological findings associated with the compounds were examined using gene expression profiles. Toxicogenomic data from TG-Gates, featuring rats exposed to 41 drugs over a 24-hour period, was collected. Preprocessing steps were implemented to address missing data and incomplete drug records. Gene selection was performed utilizing dose-response curves to capture biological information across different dosage levels. Genes failing to exhibit typical dose-response curves were excluded from further analysis. Finally, a Deep Learning model was deployed to predict potential pathological findings based on gene expression data.

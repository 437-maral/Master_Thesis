import os
import sys
from pathlib import Path
import random
import matplotlib.pyplot as plt
import pandas as pd
import copy
import shap



class ShapValues(object):
    def __init__(self, dataset, model):
        ''':param dataset: chemical dataset
           :param model: random forest model '''
           
        self.model = model
        self.dataset = dataset
        self.shap_values=None
        
    def computeRFShap(self, test_dataset):
        model = self.model
        explainer = shap.Explainer(model)
        self.shap_values = explainer.shap_values(test_dataset)
        return explainer, self.shap_values
        
  
        
    def saved_result_inplot(self, X_test, save=False, output_dir=None, summary_plot=False, dependence_plot=False):
        
        if self.shap_values is None:
            print("Error: shap_values is not yet computed.")
            return
        
        
        shap_values = copy.deepcopy(self.shap_values)
        
        if summary_plot:
            shap.summary_plot(shap_values, X_test)
        if dependence_plot:
            shap.dependence_plot(shap_values, X_test) 
        if save:
            if output_dir is not None:
                output_path = os.path.join(output_dir, 'shap_feature_explanation_plot.png')
            else:
                output_path = 'shap_feature_explanation_file.png'
            plt.tight_layout()
            plt.savefig(output_path)
            plt.close()
        else:
            plt.show()

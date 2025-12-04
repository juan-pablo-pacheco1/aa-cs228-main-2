"""
OCD Bayesian Network Model
Loads data, learns structure and parameters, performs inference
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import json
import pickle

from algorithms.bayesian_network import (
    Variable, statistics, bayesian_score,
    K2Search, LocalDirectedGraphSearch, estimate_parameters, sub2ind
)
from algorithms.em_learning import em_learning


class OCDBayesianNetwork:
    """Wrapper class for OCD treatment Bayesian network"""
    
    def __init__(self):
        self.variables = None
        self.graph = None
        self.cpds = None
        self.var_names = None
        
    def load_data(self, data_dir='data/processed'):
        """Load preprocessed data"""
        train_df = pd.read_csv(f'{data_dir}/train_discrete.csv')
        val_df = pd.read_csv(f'{data_dir}/val_discrete.csv')
        test_df = pd.read_csv(f'{data_dir}/test_discrete.csv')
        
        with open(f'{data_dir}/variable_definitions.json', 'r') as f:
            var_defs = json.load(f)
        
        self.variables = [Variable(name, r) for name, r in var_defs.items()]
        self.var_names = [v.name for v in self.variables]
        
        return train_df, val_df, test_df
    
    def learn_structure(self, train_df, method='k2'):
        """Learn Bayesian network structure"""
        D_train = train_df.values.T.astype(float)
        
        # Fill missing values for structure learning
        D_complete = D_train.copy()
        for i in range(D_complete.shape[0]):
            missing_mask = np.isnan(D_complete[i, :])
            if np.any(missing_mask):
                valid_values = D_complete[i, ~missing_mask]
                if len(valid_values) > 0:
                    # Find most common value (values are 1-indexed)
                    unique, counts = np.unique(valid_values.astype(int), return_counts=True)
                    mode_val = unique[counts.argmax()]
                    D_complete[i, missing_mask] = mode_val
        
        if method == 'k2':
            # Define causal ordering
            ordering_names = [
                'Age_Group', 'Gender', 'Ethnicity', 'Family_History', 'Duration',
                'Depression', 'Anxiety', 'Obsession_Type', 'Compulsion_Type',
                'YBOCS_Severity_Baseline', 'Treatment', 'Adherence', 'Side_Effects',
                'YBOCS_Severity_Week12', 'Response'
            ]
            ordering = [self.var_names.index(name) for name in ordering_names]
            
            k2 = K2Search(ordering)
            self.graph = k2.fit(self.variables, D_complete)
            print(f"K2 learned {self.graph.number_of_edges()} edges")
            
        elif method == 'local':
            G_init = nx.DiGraph()
            G_init.add_nodes_from(range(len(self.variables)))
            
            local_search = LocalDirectedGraphSearch(G_init, k_max=1000)
            self.graph = local_search.fit(self.variables, D_complete)
            print(f"Local search learned {self.graph.number_of_edges()} edges")
        
        score = bayesian_score(self.variables, self.graph, D_complete)
        print(f"Bayesian score: {score:.2f}")
        
        return self.graph
    
    def learn_parameters(self, train_df, use_em=True):
        """Learn CPD parameters"""
        D_train = train_df.values.T.astype(float)
        
        if use_em and np.isnan(D_train).any():
            print("Learning parameters with EM...")
            self.cpds, ll_history = em_learning(self.variables, self.graph, D_train, 
                                                max_iter=50, tol=1e-4)
            return ll_history
        else:
            print("Learning parameters with MLE...")
            # Need complete data
            D_complete = D_train.copy()
            for i in range(D_complete.shape[0]):
                missing_mask = np.isnan(D_complete[i, :])
                if np.any(missing_mask):
                    valid_values = D_complete[i, ~missing_mask]
                    if len(valid_values) > 0:
                        unique, counts = np.unique(valid_values.astype(int), return_counts=True)
                        mode_val = unique[counts.argmax()]
                        D_complete[i, missing_mask] = mode_val
            
            self.cpds = estimate_parameters(self.variables, self.graph, D_complete)
            return None
    
    def predict_response(self, patient_data):
        """Predict treatment response for a patient"""
        response_idx = self.var_names.index('Response')
        response_cpd = self.cpds[response_idx]
        
        parents = list(self.graph.predecessors(response_idx))
        
        if len(parents) == 0:
            probs = response_cpd[0, :]
        else:
            # Handle missing parent values
            parent_vals = []
            for p in parents:
                val = patient_data[p]
                if np.isnan(val):
                    # Use most common value as default
                    val = 2  # Middle category as fallback
                parent_vals.append(int(val))
            
            parent_r = [self.variables[p].r for p in parents]
            parent_idx = sub2ind(parent_r, parent_vals) - 1
            probs = response_cpd[parent_idx, :]
        
        pred_category = np.argmax(probs) + 1
        return pred_category, probs
    
    def evaluate(self, test_df):
        """Evaluate on test set"""
        D_test = test_df.values.T.astype(float)
        response_idx = self.var_names.index('Response')
        
        predictions = []
        true_responses = []
        
        for col_idx in range(D_test.shape[1]):
            patient = D_test[:, col_idx]
            true_response = int(patient[response_idx])
            pred_response, _ = self.predict_response(patient)
            
            predictions.append(pred_response)
            true_responses.append(true_response)
        
        predictions = np.array(predictions)
        true_responses = np.array(true_responses)
        
        accuracy = np.mean(predictions == true_responses)
        return accuracy, predictions, true_responses
    
    def visualize_structure(self, save_path=None):
        """Visualize network structure"""
        plt.figure(figsize=(14, 10))
        
        pos = nx.spring_layout(self.graph, k=2, iterations=50, seed=42)
        labels = {i: self.variables[i].name for i in range(len(self.variables))}
        
        nx.draw(self.graph, pos, labels=labels, with_labels=True,
                node_color='lightblue', node_size=2000,
                font_size=8, font_weight='bold',
                arrows=True, arrowsize=20, arrowstyle='->',
                edge_color='gray', width=1.5)
        
        plt.title("Bayesian Network Structure", fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        return plt.gcf()
    
    def save_model(self, path):
        """Save learned model"""
        model_data = {
            'variables': self.variables,
            'graph': self.graph,
            'cpds': self.cpds
        }
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load saved model"""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        self.variables = model_data['variables']
        self.graph = model_data['graph']
        self.cpds = model_data['cpds']
        self.var_names = [v.name for v in self.variables]
        print(f"Model loaded from {path}")


def main():
    """Train and evaluate Bayesian network"""
    print("="*60)
    print("OCD Treatment Bayesian Network")
    print("="*60)
    
    # Initialize
    bn = OCDBayesianNetwork()
    
    # Load data
    print("\n1. Loading data...")
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(base_dir, 'data/processed')
    train_df, val_df, test_df = bn.load_data(data_dir)
    print(f"   Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Learn structure
    print("\n2. Learning structure with K2...")
    bn.learn_structure(train_df, method='k2')
    
    # Learn parameters
    print("\n3. Learning parameters with EM...")
    ll_history = bn.learn_parameters(train_df, use_em=True)
    if ll_history:
        print(f"   Final log-likelihood: {ll_history[-1]:.2f}")
    
    # Visualize
    print("\n4. Visualizing structure...")
    results_dir = os.path.join(base_dir, 'results')
    os.makedirs(results_dir, exist_ok=True)
    bn.visualize_structure(os.path.join(results_dir, 'bn_structure.png'))
    print(f"   Saved to results/bn_structure.png")
    
    # Evaluate
    print("\n5. Evaluating on test set...")
    accuracy, preds, trues = bn.evaluate(test_df)
    print(f"   Accuracy: {accuracy*100:.2f}%")
    
    # Save model
    print("\n6. Saving model...")
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    bn.save_model(os.path.join(models_dir, 'bayesian_network.pkl'))
    
    print("\n" + "="*60)
    print("Bayesian network training complete!")
    print("="*60)
    
    return bn


if __name__ == "__main__":
    bn = main()


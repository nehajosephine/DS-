import numpy as np  # Importing NumPy for handling arrays
import pandas as pd  # Importing pandas for structured data operations
from sklearn.model_selection import train_test_split  # Function to split datasets for ML models
from sklearn.metrics import mean_squared_error  # Function to compute error in predictions
from scipy.cluster.hierarchy import linkage, dendrogram  # Functions for hierarchical clustering
import matplotlib.pyplot as plt  # Matplotlib for plotting data visualizations
from mlxtend.frequent_patterns import apriori, association_rules  # Importing Apriori functions for pattern mining

# Sample transaction data for market basket analysis
purchases = [
    ['eggs', 'cheese', 'yogurt', 'banana'], 
    ['eggs', 'cheese', 'yogurt'],  
    ['eggs', 'cheese'], 
    ['eggs', 'cheese', 'banana'],  
    ['cheese', 'yogurt', 'banana'], 
    ['cheese', 'banana'],  
    ['eggs', 'cheese', 'yogurt', 'banana'],  
    ['eggs', 'banana']  
]

# Extracting unique items from the dataset
unique_items = sorted(set(product for order in purchases for product in order))

# Creating a DataFrame with binary encoding (1 if the item is bought, else 0)
order_data = pd.DataFrame([{product: (product in order) for product in unique_items} for order in purchases])

# Running Apriori algorithm to identify frequently occurring itemsets
common_patterns = apriori(order_data, min_support=0.3, use_colnames=True)

# Generating association rules from identified patterns
associations = association_rules(common_patterns, metric="lift", min_threshold=1.0) 

# Displaying the frequent itemsets found
print("Frequent Itemsets:")
print(common_patterns)

# Displaying the association rules derived
print("\nAssociation Rules:")
print(associations)

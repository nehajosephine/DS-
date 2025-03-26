import numpy as np  # For numerical operations
import pandas as pd  # For handling datasets
import matplotlib.pyplot as plt  # For visualization
from sklearn.model_selection import train_test_split  # To split dataset
from sklearn.metrics import mean_squared_error  # To evaluate model performance
from scipy.cluster.hierarchy import linkage, dendrogram  # For hierarchical clustering

# Load the dataset
dataset_path = "setadv.csv"  # Path to dataset
dataset = pd.read_csv(dataset_path)  # Read CSV file into a pandas DataFrame

# Extract features for clustering
features = dataset[['TV', 'radio', 'newspaper']].values  # Convert selected columns into a NumPy array

# Perform hierarchical clustering using complete linkage
linkage_result = linkage(features, method='complete')

# Visualizing the clustering with a dendrogram
plt.figure(figsize=(10, 5))  # Set the figure size
plt.title("Hierarchical Clustering Dendrogram (Complete Linkage)")  # Add title to the plot
plt.xlabel("Data Points")  # Label for x-axis
plt.ylabel("Distance")  # Label for y-axis
dendrogram(linkage_result)  # Generate dendrogram
plt.show()  # Display the dendrogram
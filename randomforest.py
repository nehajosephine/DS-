import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load dataset
data_path = "setadv.csv"
data = pd.read_csv(data_path)

# Define feature matrix and target variable
features = data[['TV', 'radio', 'newspaper']].values  # Selecting predictor variables
target = data['sales'].values  # Selecting target variable

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Number of models to use for ensembling
num_models = 5
model_predictions = []

# Train multiple Random Forest models and collect their predictions
for seed in range(num_models):
    rf_model = RandomForestRegressor(n_estimators=10, random_state=seed)  # Initialize model with 10 trees
    rf_model.fit(X_train, y_train)  # Train model on training data
    predictions = rf_model.predict(X_test)  # Make predictions on test data
    model_predictions.append(predictions)  # Store predictions from this model

# Compute final predictions as the mean of individual predictions
ensemble_predictions = np.mean(model_predictions, axis=0)

# Calculate model performance using Mean Squared Error
mse_value = mean_squared_error(y_test, ensemble_predictions)
print(f"Mean Squared Error: {mse_value}")  # Print MSE value
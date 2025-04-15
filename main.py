# smart_waste_predictor.py

# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# For reproducibility
np.random.seed(42)

# ------------------------------
# Step 1: Create a Synthetic Dataset
# ------------------------------
# We'll generate an imaginary dataset with 1000 records.

# Define the number of samples
num_samples = 1000

# Features:
# - fill_percentage: current bin fill level from 0 to 100%
# - temperature: current ambient temperature in °C (e.g., 10 to 40)
# - precipitation: amount of rain in mm in the last hour (0 to 20)
# - traffic_index: a number representing traffic congestion (1: light, 10: heavy)
# - population_density: persons per km² near the bin (100 to 10000)
# - hour_of_day: hour in 24-hour format (0 to 23)

fill_percentage = np.random.uniform(0, 100, num_samples)
temperature = np.random.uniform(10, 40, num_samples)
precipitation = np.random.uniform(0, 20, num_samples)
traffic_index = np.random.randint(1, 11, num_samples)
population_density = np.random.uniform(100, 10000, num_samples)
hour_of_day = np.random.randint(0, 24, num_samples)

# The target variable: needs_pickup
# For simplicity, let's say if the fill percentage is above 80% or (80% is not reached, but high population density and heavy traffic)
# then the bin is likely to be full (needs pickup).
needs_pickup = np.where((fill_percentage > 80) | 
                        ((fill_percentage > 60) & (population_density > 5000) & (traffic_index > 7)),
                        1, 0)

# Create a DataFrame from the synthetic data
data = pd.DataFrame({
    'fill_percentage': fill_percentage,
    'temperature': temperature,
    'precipitation': precipitation,
    'traffic_index': traffic_index,
    'population_density': population_density,
    'hour_of_day': hour_of_day,
    'needs_pickup': needs_pickup
})

# ------------------------------
# Step 2: Prepare the Data for Modeling
# ------------------------------
# Separate the features (X) and the target (y)
X = data.drop('needs_pickup', axis=1)
y = data['needs_pickup']

# Split data into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ------------------------------
# Step 3: Train a Machine Learning Model
# ------------------------------
# We will use a RandomForestClassifier as our model.
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# ------------------------------
# Step 4: Evaluate the Model
# ------------------------------
# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate the accuracy and generate a classification report
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Smart Waste Collection Predictor")
print("===============================")
print("Model Accuracy:", round(accuracy, 2))
print("\nClassification Report:\n", report)

# ------------------------------
# Step 5: Save the Model (Optional)
# ------------------------------
# If you'd like to save the model for later use, you can use joblib (uncomment the below lines):
# import joblib
# joblib.dump(model, 'smart_waste_predictor_model.pkl')

import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import xgboost as xgb

# Load penguins dataset
data = sns.load_dataset('penguins').dropna()

# Feature and target split
X = data.drop('species', axis=1)
y = data['species']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit XGBoost model
model = xgb.XGBClassifier()
model.fit(X_train, y_train)

# Print confirmation
print("Model has been trained successfully.")


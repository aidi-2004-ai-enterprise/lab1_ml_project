import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
import xgboost as xgb

def load_penguins_data():
    """Load and clean the penguins dataset."""
    data = sns.load_dataset('penguins').dropna()
    return data

def split_dataset(data):
    """Split dataset into features and target, then into train and test sets."""
    X = data.drop('species', axis=1)
    y = data['species']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def create_xgboost_model():
    """Create a default XGBoost classifier."""
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    return model

def fit_model(model, X_train, y_train):
    """Fit the XGBoost model to the training data."""
    model.fit(X_train, y_train)
    return model

if __name__ == "__main__":
    # Load data
    penguins_data = load_penguins_data()
    
    # Person A: Split dataset
    X_train, X_test, y_train, y_test = split_dataset(penguins_data)
    print("Training set shape:", X_train.shape)
    print("Test set shape:", X_test.shape)
    
    # Person B: Create model
    model = create_xgboost_model()
    print("Default XGBoost model created:")
    print(model)
    
    # Person C: Fit model
    model = fit_model(model, X_train, y_train)
    print("Model has been trained successfully.")
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

def load_penguins_data():
    # Load penguins dataset from seaborn
    data = sns.load_dataset('penguins')
    return data.dropna()  # Drop missing values

if __name__ == "__main__":
    penguins_data = load_penguins_data()
    print(penguins_data.head())
    
    # Feature and target split
    X = penguins_data.drop('species', axis=1)
    y = penguins_data['species']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Print the shapes
    print("Training set shape:", X_train.shape)
    print("Test set shape:", X_test.shape)
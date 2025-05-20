import pandas as pd
import seaborn as sns

def load_penguins_data():
    # Load penguins dataset from seaborn
    data = sns.load_dataset('penguins')
    return data

if __name__ == "__main__":
    penguins_data = load_penguins_data()
    print(penguins_data.head())

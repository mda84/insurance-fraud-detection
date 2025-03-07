import pandas as pd

def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads the dataset from a CSV file.
    """
    data = pd.read_csv(file_path)
    return data

def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the dataset by handling missing values and stripping extra whitespace.
    """
    data = data.copy()
    # Strip whitespace from all object columns.
    for col in data.select_dtypes(include='object').columns:
        data[col] = data[col].str.strip()
    return data

if __name__ == "__main__":
    df = load_data("../data/fraud_oracle.csv")
    df = clean_data(df)
    print(df.head())

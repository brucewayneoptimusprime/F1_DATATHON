import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from train_xgboost import train_and_evaluate  # Importing the training function

# Define file paths
file_paths = {
    'sample_submission': r"C:\ML F1 DATATHON\data\sample_submission.csv",
    'test': r"C:\ML F1 DATATHON\data\test.csv",
    'train': r"C:\ML F1 DATATHON\data\train.csv",
    'validation': r"C:\ML F1 DATATHON\data\validation.csv"
}

# Define specific columns with mixed dtype errors to be cast as int
int_columns = {
    'test': [13, 16],
    'train': [4, 13, 14, 16, 47],
    'validation': [13, 16, 36]
}

# Function to load data and fill missing values with mode
def load_data(file_path, int_columns=None):
    """
    Load data from CSV file and return DataFrame.
    """
    dtype = None
    converters = None

    if int_columns:
        dtype = {col: 'int' for col in int_columns}
        converters = {col: lambda x: int(float(x)) if isinstance(x, str) and x != '\\N' and '.' not in x else np.nan for col in int_columns}

    df = pd.read_csv(file_path, dtype=dtype, converters=converters, low_memory=False)

    # Fill missing values with mode for columns with many missing values
    mode_fill_columns = ['fastestLap', 'max_speed']  # Example columns with many missing values
    for col in mode_fill_columns:
        if col in df.columns:
            mode_value = df[col].mode().iloc[0]
            df[col].fillna(mode_value, inplace=True)

    return df

# Function to describe dataset
def describe_dataset(df):
    """
    Print dataset information, numerical columns description,
    missing values before and after imputation, unique values count,
    and columns with mixed data types.
    """
    print(df.info())
    print("\nNumerical Columns Description:")
    print(df.describe())

    categorical_cols = df.select_dtypes(include=['object']).columns
    if not categorical_cols.empty:
        print("\nCategorical Columns Description:")
        print(df[categorical_cols].describe())

    print("\nMissing Values (Before Imputation):")
    print(df.isnull().sum())

    # Fill missing values with mode for all columns (optional)
    df.fillna(df.mode().iloc[0], inplace=True)

    print("\nMissing Values (After Mode Imputation):")
    print(df.isnull().sum())

    print("\nUnique Values Count:")
    for col in df.columns:
        print(f"{col}: {df[col].nunique()} unique values")

    print("\nColumns with Mixed Data Types:")
    mixed_dtype_cols = df.select_dtypes(include=['object']).columns
    print(mixed_dtype_cols)

    return df

# Function to detect outliers and create boxplot
def detect_outliers_and_boxplot(df, column_name):
    """
    Detect outliers in a specific column using IQR method and create a boxplot of filtered data.
    """
    # Calculate percentiles
    percentile_25 = df[column_name].quantile(0.25)
    percentile_75 = df[column_name].quantile(0.75)

    # Calculate IQR (Interquartile Range)
    IQR = percentile_75 - percentile_25

    # Calculate upper and lower limits for outliers
    upper_limit = percentile_75 + 1.5 * IQR
    lower_limit = percentile_25 - 1.5 * IQR

    # Filter data within upper and lower limits
    filtered_data = df[(df[column_name] >= lower_limit) & (df[column_name] <= upper_limit)]

    # Create a boxplot of the filtered data
    plt.figure(figsize=(8, 6))
    plt.boxplot(filtered_data[column_name], vert=False)
    plt.xlabel(column_name)
    plt.ylabel('Distribution')
    plt.title(f'Boxplot of {column_name} with adjusted whiskers')
    plt.show()

    return filtered_data

# Main function to run the script
def main():
    # Load each dataset and describe it
    for key, path in file_paths.items():
        print(f"\nLoading and describing dataset: {key}")
        df = load_data(path, int_columns.get(key))
        df = describe_dataset(df)

        # Example usage to detect outliers and create boxplot for 'position' column
        if 'position' in df.columns:
            filtered_data = detect_outliers_and_boxplot(df, 'position')

        # Example usage to train model (replace with actual data handling)
        # For demonstration, assume X_train, y_train, X_val, y_val are derived from df
        # Adjust as per your actual data and requirements
        target_columns = ['position', 'result_driver_standing']  # Replace with your actual target column names
        if all(col in df.columns for col in target_columns):
            X_train = df.drop(columns=target_columns)
            y_train = df[target_columns[0]]  # Assuming position is the first target column

            # Example: Train and evaluate using XGBoost
            train_and_evaluate(X_train, y_train)


if __name__ == "__main__":
    main()

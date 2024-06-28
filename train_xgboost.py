import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Function to load data
def load_data(file_path):
    return pd.read_csv(file_path)

# Function to preprocess data (replace with your preprocessing steps)
def preprocess_data(df):
    # Drop non-numeric columns or encode them appropriately
    df_numeric = df.select_dtypes(include=['int', 'float'])  # Select only numeric columns

    # Assuming 'position' and 'result_driver_standing' are your target columns
    X = df_numeric.drop(columns=['position', 'result_driver_standing'])
    y = df_numeric[['position', 'result_driver_standing']]  # Selecting both target columns
    return X, y

# Function to train and evaluate XGBoost model
def train_and_evaluate(X_train, y_train, X_test, result_driver_standing_test):
    # XGBoost parameters (replace with your preferred parameters)
    params = {
        'max_depth': 6,
        'eta': 0.3,
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse'
    }

    # Split data into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Convert data to DMatrix format for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train['position'])
    dval = xgb.DMatrix(X_val, label=y_val['position'])
    dtest = xgb.DMatrix(X_test)

    # Train the model
    model = xgb.train(params, dtrain, num_boost_round=10, evals=[(dtrain, 'train'), (dval, 'validation')])

    # Make predictions on test set
    position_predictions = model.predict(dtest)

    # Create a DataFrame with the required columns for submission
    predictions_df = pd.DataFrame({
        'result_driver_standing': result_driver_standing_test,
        'position': position_predictions
    })

    # Save both 'position' and 'result_driver_standing' columns in the same file
    predictions_combined_path = r"C:\ML F1 DATATHON\Submit here\predictions_combined.csv"
    predictions_df.to_csv(predictions_combined_path, index=False)

    print(f"Predicted columns saved successfully to:\n{predictions_combined_path}")

def main():
    # Load and preprocess train data
    train_data_path = r"C:\ML F1 DATATHON\data\train.csv"
    df_train = load_data(train_data_path)
    X_train, y_train = preprocess_data(df_train)

    # Load test data
    test_data_path = r"C:\ML F1 DATATHON\data\test.csv"
    df_test = load_data(test_data_path)

    # Extract 'result_driver_standing' for the test dataset
    result_driver_standing_test = df_test['result_driver_standing']

    # Ensure the test dataset has the same columns as the train dataset
    X_test = df_test[X_train.columns]

    # Train and evaluate XGBoost model
    train_and_evaluate(X_train, y_train, X_test, result_driver_standing_test)

if __name__ == "__main__":
    main()

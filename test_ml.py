import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import (
    train_model,
    compute_model_metrics,
    inference,
    save_model,
    load_model
)

# Load raw data
def rawdata():
    return pd.read_csv("data/census.csv")

# Sample data and split
def test_data():
    df = pd.read_csv("data/census.csv")
    df_sampled = df.sample(100)
    X = df_sampled.drop('salary', axis=1)  # Assuming 'salary' is the label column
    y = df_sampled['salary']
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Test 1: Test model predictions
def test_one():
    """Test model type"""

    X_train, _, y_train, _ = test_data()

    # Process the data
    # Make sure 'categorical_features' is correctly passed as a keyword argument
    X_train_processed, y_train_processed, encoder, lb = process_data(
        X_train, y_train, categorical_features=['workclass'], label='salary'
    )

    # Train the model
    model = train_model(X_train_processed, y_train_processed)

    # Make predictions
    preds = inference(model, X_train_processed)

    # Assertions
    assert isinstance(preds, np.ndarray), f"Expected predictions to be numpy array, but got {type(preds)}"
    assert preds.shape[0] == len(y_train_processed), f"Expected number of predictions ({len(y_train_processed)}) to match number of labels ({preds.shape[0]})"

# Test 2: Test the model type
def test_two():
    """Test the type of the model returned by the training function."""

    X_train, _, y_train, _ = test_data()

    # Process data (train set)
    X_train_processed, y_train_processed, encoder, lb = process_data(
        X_train, y_train, categorical_features=['workclass'], label='salary', training=True
    )
    
    # Train the model
    model = train_model(X_train_processed, y_train_processed)
    
    # Test if the model is of expected type (e.g., LogisticRegression)
    assert isinstance(model, LogisticRegression), f"Expected model of type LogisticRegression, got {type(model)}"

# Test 3: Test data processing
def test_three():
    """Test that the data is processed correctly"""

    X_train, X_test, y_train, y_test = test_data()

    # Test that X_train and y_train are pandas DataFrames/Series
    assert isinstance(X_train, pd.DataFrame), f"Expected X_train to be a DataFrame, got {type(X_train)}"
    assert isinstance(y_train, pd.Series), f"Expected y_train to be a Series, got {type(y_train)}"
    
    # Test that X_test and y_test are pandas DataFrames/Series
    assert isinstance(X_test, pd.DataFrame), f"Expected X_test to be a DataFrame, got {type(X_test)}"
    assert isinstance(y_test, pd.Series), f"Expected y_test to be a Series, got {type(y_test)}"
    
    # Test the size of the train and test datasets
    assert len(X_train) == 80, f"Expected X_train to have 80 rows, but got {len(X_train)}"
    assert len(X_test) == 20, f"Expected X_test to have 20 rows, but got {len(X_test)}"

if __name__ == '__main__':
    pytest.main()



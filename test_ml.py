import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
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
    """Test that the shape and types of the data are correct"""
    
    X_train, X_test, y_train, y_test = test_data()
    
    # Test that X_train and X_test have the correct number of columns
    assert X_train.shape[1] == 10, f"Expected X_train to have 10 features, but got {X_train.shape[1]}"
    assert X_test.shape[1] == 10, f"Expected X_test to have 10 features, but got {X_test.shape[1]}"
    
    # Test that y_train and y_test are one-dimensional
    assert y_train.ndim == 1, f"Expected y_train to be one-dimensional, but got {y_train.ndim} dimensions"
    assert y_test.ndim == 1, f"Expected y_test to be one-dimensional, but got {y_test.ndim} dimensions"
    
    # Test the type of the labels (y_train and y_test) to be integers or floats
    assert pd.api.types.is_numeric_dtype(y_train), f"Expected y_train to have numeric dtype, got {y_train.dtype}"
    assert pd.api.types.is_numeric_dtype(y_test), f"Expected y_test to have numeric dtype, got {y_test.dtype}"

# Test 2: Test the model type
def test_two():
    """Test the type of the model returned by the training function."""

    X_train, _, y_train, _ = test_data()

    # Process data (train set)
    X_train_processed, y_train_processed, encoder, lb = process_data(
        X_train, y_train, categorical_features=['workclass', 'education', 'marital-status'], label='salary', training=True
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



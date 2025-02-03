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

# Pytest fixture to handle setup and preprocessing for training
@pytest.fixture(scope="module")
def setup_data():
    """Fixture to set up and preprocess train data."""
    X_train, _, y_train, _ = test_data()  # Get data
    
    # Combine X_train and y_train into a single DataFrame
    train = pd.concat([X_train, y_train], axis=1)
    
    # Process the data
    X_train_processed, y_train_processed, encoder, lb = process_data(
        train, categorical_features=['workclass', 'education', 'marital-status'], label='salary'
    )
    
    return X_train_processed, y_train_processed, encoder, lb

# Pytest fixture to handle setup and preprocessing for testing
@pytest.fixture(scope="module")
def setup_test_data(setup_data):
    """Fixture to process test data using the trained encoder and label binarizer."""
    # Get processed train data from setup_data fixture
    X_train_processed, y_train_processed, encoder, lb = setup_data
    
    # Get test data
    X_test, _, y_test, _ = test_data()  # Replace this with actual test data loading
    test = pd.concat([X_test, y_test], axis=1)
    
    # Process the test data using the encoder and label binarizer from the train data
    X_test_processed, y_test_processed, _, _ = process_data(
        test, categorical_features=['workclass', 'education', 'marital-status'], label='salary', encoder=encoder, lb=lb, training=False
    )
    
    return X_test_processed, y_test_processed, encoder, lb

# Test 1: Test model predictions
def test_one(setup_data):
    """Test model type"""

    X_train_processed, y_train_processed, encoder, lb = setup_data

    # Train the model
    model = train_model(X_train_processed, y_train_processed)

    # Make predictions
    preds = inference(model, X_train_processed)

    # Assertions
    assert isinstance(preds, np.ndarray), f"Expected predictions to be numpy array, but got {type(preds)}"
    assert preds.shape[0] == len(y_train_processed), f"Expected number of predictions ({len(y_train_processed)}) to match number of labels ({preds.shape[0]})"

# Test 2: Test the model type
def test_two(setup_data):
    """Test the type of the model returned by the training function."""

    X_train_processed, y_train_processed, encoder, lb = setup_data
    
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


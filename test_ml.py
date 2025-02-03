import pytest
# TODO: add necessary import
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, label_binarize
from ml.data import process_data
from ml.model import (
    train_model,
    compute_model_metrics,
    inference,
    save_model,
    load_model
)
#load raw data
def rawdata():
    return pd.read_csv("data/census.csv")

#sample data
def test_data():
    df = pd.read_csv("data/census.csv")
    df.sample(100)
    return train_test_split

# TODO: implement the first test. Change the function name and input as needed
def test_one():
    """Test model type"""

    X_train, _, y_train, _ = test_data()

    X_train_processed, y_train_processed, encoder, lb = process_data(
        X_train, y_train, categorical_features=['workclass', 'education', 'marital-status'], label='salary'
    )

    model = train_model(X_train_processed, y_train_processed)

    preds = inference(model, X_train_processed)

    assert isinstance(preds, np.ndarray)

    assert preds.shape == y_train_processed

if __name__ == '__main':
    pytest.main()

    pass


# TODO: implement the second test. Change the function name and input as needed
def test_two():
    def test_train_model():
    """Test the type of the model returned by the training function."""
    # Load data and split into train/test
    X_train, _, y_train, _ = test_data()

    # Process data (train set)
    X_train_processed, y_train_processed, encoder, lb = process_data(
        X_train, y_train, categorical_features=['workclass', 'education', 'marital-status'], label='salary', training=True
    )
    
    # Train the model
    model = train_model(X_train_processed, y_train_processed)
    
    # Test if the model is of expected type (e.g., LogisticRegression)
    assert isinstance(model, LogisticRegression), f"Expected model of type LogisticRegression, got {type(model)}"



# TODO: implement the third test. Change the function name and input as needed
def test_three():
    def test_data_processing():
    # Load data and split into train/test
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


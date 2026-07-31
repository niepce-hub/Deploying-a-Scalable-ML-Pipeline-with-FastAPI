import pytest
import pandas as pd
from ml.data import process_data
from ml.model import train_model, inference

# TODO: implement the first test. Change the function name and input as needed
def test_process_data_shapes():
    """
    Ensure process_data returns matching X and y shapes."""
    data = pd.DataFrame({
        "workclass": ["Private"],
        "education": ["HS-grad"],
        "marital-status": ["Married-civ-spouse"],
        "occupation": ["Prof-specialty"],
        "relationship": ["Husband"],
        "race": ["White"],
        "sex": ["Male"],
        "native-country": ["United-States"],
        "age": [37],
        "fnlgt": [178356],
        "education-num": [10],
        "capital-gain": [0],
        "capital-loss": [0],
        "hours-per-week": [40],
        "salary": [">50K"]
    })
    
    cat_features = [
        "workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"
    ]

    X, y, encoder, lb = process_data(
        data,
        categorical_features=cat_features,
        label="salary",
        training=True
    )

    assert X.shape[0] == y.shape[0]
    assert X.shape[1] > 0 

# TODO: implement the second test. Change the function name and input as needed
def test_train_model_returns_fitted_model():
    """
    # Verify train_model returns a model with a predict method, """
    X = pd.DataFrame([[0, 1, 2]])
    y = [1]
    model = train_model(X, y)
    assert hasattr(model, "predict")


# TODO: implement the third test. Change the function name and input as needed
def test_inference_output_length():
    """Confirm inference returns predictions of correct length. """
    X = pd.DataFrame([[0, 1, 2]])
    y = [1]
    model = train_model(X, y)
    preds = inference(model, X)
    assert len(preds) == len(y)

from fastapi.testclient import TestClient
from main import app

# Create a reusable client instance
client = TestClient(app)

# Test the root endpoint


def test_get_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "message": "Welcome to the Census Income Prediction API!"
    }

# Test the predict endpoint


def test_post_predict():

    # Sample input matching your Data model fields
    sample_input = {
        "age": 37,
        "workclass": "Private",
        "fnlgt": 178356,
        "education": "HS-grad",
        "education-num": 10,
        "marital-status": "Married-civ-spouse",
        "occupation": "Prof-specialty",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    }

    response = client.post("/predict/", json=sample_input)
    assert response.status_code == 200
    assert "result" in response.json()

# Test that prediction returns a valid label


def test_prediction_label_valid():
    sample_input = {
        "age": 37,
        "workclass": "Private",
        "fnlgt": 178356,
        "education": "HS-grad",
        "education-num": 10,
        "marital-status": "Married-civ-spouse",
        "occupation": "Prof-specialty",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    }

    response = client.post("/predict/", json=sample_input)
    assert response.status_code == 200

    result = response.json()["result"]
    assert any(x in result for x in ["<=50K", ">50K", "<50K", ">50K"])

# Test multiple predictions in a loop


def test_multiple_predictions():
    inputs = [
        {
            "age": 25,
            "workclass": "Private",
            "fnlgt": 20000,
            "education": "Bachelors",
            "education-num": 13,
            "marital-status": "Never-married",
            "occupation": "Sales",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Female",
            "capital-gain": 0,
            "capital-loss": 0,
            "hours-per-week": 40,
            "native-country": "United-States"
        },
        {
            "age": 52,
            "workclass": "Self-emp-inc",
            "fnlgt": 150000,
            "education": "Masters",
            "education-num": 14,
            "marital-status": "Married-civ-spouse",
            "occupation": "Exec-managerial",
            "relationship": "Husband",
            "race": "White",
            "sex": "Male",
            "capital-gain": 5000,
            "capital-loss": 0,
            "hours-per-week": 60,
            "native-country": "United-States"
        }
    ]

    for item in inputs:
        response = client.post("/predict/", json=item)
        assert response.status_code == 200
        assert response.json()["result"] in ["<=50K", ">50K", "<50K", ">50K"]

# Test missing required field


def test_missing_field():
    bad_input = {
        "age": 37,
        # "workclass" is missing
        "fnlgt": 178356,
        "education": "HS-grad",
        "education-num": 10,
        "marital-status": "Married-civ-spouse",
        "occupation": "Prof-specialty",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital-gain": 0,
        "capital-loss": 0,
        "hours-per-week": 40,
        "native-country": "United-States"
    }

    response = client.post("/predict/", json=bad_input)
    assert response.status_code == 422

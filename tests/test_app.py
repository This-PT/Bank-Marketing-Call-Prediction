from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_valid_result():
    customer = {
        "age": 30,
        "job": "student",
        "marital": "single",
        "education": "tertiary",
        "default": "no",
        "balance": 1500,
        "housing": "no",
        "loan": "no",
        "pdays": -1,
        "previous": 0,
        "poutcome": "unknown",
    }

    response = client.post("/predict", json=customer)
    result = response.json()

    assert response.status_code == 200
    assert result["prediction"] in ["yes", "no"]
    assert 0 <= result["yes_probability"] <= 1



def test_predict_rejects_missing_fields():
    incomplete_customer = {
        "age": 30
    }

    response = client.post(
        "/predict",
        json=incomplete_customer,
    )

    assert response.status_code == 422

def test_predict_rejects_invalid_poutcome():
    customer = {
            "age": 30,
            "job": "student",
            "marital": "single",
            "education": "tertiary",
            "default": "no",
            "balance": 1500,
            "housing": "no",
            "loan": "no",
            "pdays": -1,
            "previous": 0,
            "poutcome": "maybe",
        }

    response = client.post("/predict", json=customer)

    assert response.status_code == 422


def test_predict_rejects_invalid_housing():
    customer = {
            "age": 30,
            "job": "student",
            "marital": "single",
            "education": "tertiary",
            "default": "no",
            "balance": 1500,
            "housing": "sometimes",
            "loan": "no",
            "pdays": -1,
            "previous": 0,
            "poutcome": "unknown",
        }

    response = client.post("/predict", json=customer)

    assert response.status_code == 422
from fastapi import FastAPI
from pydantic import BaseModel
from predict import load_model_bundle, predict_customer
from typing import Literal

class CustomerInput(BaseModel):
    age:int
    job: Literal[
    "admin.",
    "blue-collar",
    "entrepreneur",
    "housemaid",
    "management",
    "retired",
    "self-employed",
    "services",
    "student",
    "technician",
    "unemployed",
    "unknown",
    ]
    marital:Literal["divorced", "married", "single"]
    education:Literal["primary", "secondary", "tertiary", "unknown"]
    default:Literal["yes", "no"]
    balance:float
    housing:Literal["yes", "no"]
    loan:Literal["yes", "no"]
    pdays:int
    previous:int
    poutcome: Literal["failure", "other", "success", "unknown"]



app = FastAPI()
model_bundle = load_model_bundle()


@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(customer: CustomerInput):
    customer_data = customer.model_dump()
    result = predict_customer(customer_data, model_bundle)
    return result
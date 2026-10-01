# Bank Marketing: Who Should We Call?

This is my traditional machine learning project using the [UCI Bank Marketing dataset](https://archive.ics.uci.edu/dataset/222/bank%2Bmarketing). I wanted to build more than a model that prints an accuracy number. The goal is to understand the full workflow: define when a prediction happens, prepare the data, compare models fairly, choose a decision threshold, and inspect mistakes. I also built a FastAPI API, tested its inputs and predictions, and packaged it with Docker so someone else can run it.

## The question

**Before deciding which customers to call, can we identify customers more likely to subscribe to a term deposit?**

The dataset has 45,211 customer records. About 11.7% have `y = yes`, so the classes are imbalanced. A model that always predicts `no` gets about 88.3% accuracy but finds no subscribers. That is why I focus on precision, recall, F1, balanced accuracy, and the confusion matrix instead of accuracy alone.

## Features available at prediction time

The model uses 11 features. I removed `y` because it is the answer. I also removed `duration`, `contact`, `day`, `month`, and `campaign` from the inputs for this project's **before-calling** decision point. They describe the current contact or campaign and would not necessarily be known when choosing whom to call. This timing assumption matters: a different prediction point could allow a different feature set.

I kept features such as `age`, `balance`, `housing`, `loan`, `pdays`, `previous`, and `poutcome`. The last three describe *previous* campaign history, so they can be available before the current campaign. A previous `success` is a clue, not the answer to whether a customer subscribes this time.

## Method

1. Split the data into 80% development data (`X_train`) and 20% held-out test data, stratified by `y`.
2. Put numeric imputation and scaling, plus categorical imputation and one-hot encoding, inside each model pipeline. This lets every cross-validation fold learn preprocessing only from its training rows.
3. Compare a dummy model, balanced logistic regression, and a balanced decision tree.
4. Use five stratified folds within `X_train` to obtain out-of-fold predictions. Every development row is predicted by a model that did not train on that row.
5. Compare thresholds `0.30`, `0.40`, `0.50`, `0.60`, and `0.70` using the `yes` class F1 score. Refit the selected model on all of `X_train`, then evaluate it on the test data.

I also tried a small tree grid in `tune_tree.py`: depths 4, 6, and 8, with minimum leaf sizes 50 and 100. The depth-6, leaf-50 tree had the highest out-of-fold F1 among these tested combinations. The differences from some other settings were small, so I do not treat it as a guaranteed optimum.

## Results

| Model | Selected threshold | Out-of-fold yes F1 | Out-of-fold balanced accuracy |
| --- | ---: | ---: | ---: |
| Dummy (always `no`) | — | 0.000 | 0.500 |
| Logistic regression | 0.60 | 0.354 | 0.642 |
| Decision tree (`max_depth=6`, `min_samples_leaf=50`) | 0.70 | **0.392** | **0.655** |

The decision tree won among the tested models by out-of-fold `yes` F1. Threshold `0.70` was only slightly ahead of `0.60` (`0.392` versus `0.390`), so I see them as practically close rather than claiming that `0.70` is universally best.

For the selected tree on the test data:

| Metric | Value |
| --- | ---: |
| Accuracy | 0.860 |
| Balanced accuracy | 0.656 |
| `yes` precision | 0.400 |
| `yes` recall | 0.390 |
| `yes` F1 | 0.395 |

| Actual / predicted | Predicted `no` | Predicted `yes` |
| --- | ---: | ---: |
| Actual `no` | 7,366 | 619 |
| Actual `yes` | 645 | 413 |

The tree identified 413 of the 1,058 subscribers in the test set. Of the 1,032 customers it predicted as `yes`, 413 subscribed. This is a prioritization signal, not proof that calling these customers would be profitable; the dataset does not provide call costs or subscription value.

## What the errors taught me

`poutcome` means the outcome of a **previous** marketing campaign:

- `success`: 306 test customers had previous success. The final tree predicted `yes` for all 306; 199 subscribed this time and 107 did not.
- `unknown`: 7,370 test customers had no known previous outcome. The tree missed 548 of the 657 actual subscribers in this group.

The final tree's split-based feature importance was highest for `poutcome_success` (0.462), followed by `housing_no` (0.183), `age` (0.137), `balance` (0.097), and `pdays` (0.084). These numbers describe how much the tree used each feature for splits. They do **not** show a causal effect or a percentage change in subscription probability.

## How to run

For local development, use Python 3.11. Run these commands in Windows Command Prompt from the project folder:

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

With the dataset file `bank-full.csv` present, train and evaluate the models:

```bat
python baseline.py
```

`baseline.py` prints the cross-validation comparison, selected model and threshold, test metrics, confusion matrix, errors by previous campaign outcome, and tree feature importance.

To repeat the separate tree-setting comparison:

```bat
python tune_tree.py
```

## Saved model and new predictions

Running `baseline.py` saves the selected fitted pipeline, threshold, model name, and required feature names to:

```text
artifacts/bank_marketing_model.joblib
```

Run a prediction for the example customer without retraining:

```bat
python predict.py
```

`predict.py` loads the saved bundle, checks that all required customer features are present, produces the probability of `yes`, and applies the selected threshold. The example customer currently produces:

```text
yes_probability=0.493
threshold=0.70
prediction=no
```

The saved artifact should only be loaded from a trusted source and should use compatible Python and scikit-learn versions.

## Run the API locally

With the virtual environment active and `artifacts/bank_marketing_model.joblib` present:

```bat
python -m uvicorn app:app --reload
```

Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs). FastAPI creates this page so I can see the request format and try the endpoints from the browser.

| Endpoint | What it does |
| --- | --- |
| `GET /health` | Returns `{"status": "ok"}` |
| `POST /predict` | Receives one customer's features and returns a model prediction |

In `/docs`, open `POST /predict`, click **Try it out**, paste this JSON, and click **Execute**:

```json
{
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
  "poutcome": "unknown"
}
```

The response contains:

| Field | Meaning |
| --- | --- |
| `prediction` | `yes` or `no`, based on the selected threshold |
| `yes_probability` | The model's estimated probability of `yes` |
| `threshold` | The cutoff applied to that score |
| `model_name` | The model used for the prediction |

I use a Pydantic `CustomerInput` schema to check the required fields and their types. Categorical fields use `Literal` to accept the values defined in `app.py`. Missing fields or invalid categories, such as `poutcome="maybe"` or `housing="sometimes"`, return HTTP `422` before prediction runs.

The saved pipeline loads when the API starts. Each request then goes through the saved preprocessing, model, and threshold without retraining.

## Run with Docker

Install Docker Desktop and make sure its engine is running. From the project folder, build the image:

```bat
docker build -t bank-marketing-api .
```

This packages Python 3.11, the dependencies, API code, and the saved model. The model file must be present in `artifacts/` before building. The training dataset is not needed to serve predictions.

Start a container from that image:

```bat
docker run --rm -p 8000:8000 bank-marketing-api
```

Open [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health) to check the API, or [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) to try the same prediction request shown above.

`-p 8000:8000` connects port 8000 on my computer to port 8000 inside the container. Uvicorn listens on `0.0.0.0` inside the container; I use `127.0.0.1` in the browser to reach it from my computer. If another local API is already using port 8000, stop it first.

Keep the terminal open while testing. Press `Ctrl+C` to stop the container. `--rm` removes the stopped container, while the image stays available for the next run. Rebuild the image after changing the code, dependencies, or model.

Docker runs the API environment on a machine with Docker installed. These commands run it locally; they do not publish it to an online server.

## API tests

With the local virtual environment active and the saved model present:

```bat
python -m pytest tests/test_app.py -v
```

The tests use FastAPI's `TestClient`, so I do not need to start Uvicorn or Docker to run them. The valid prediction test uses the real saved model.

| Test | What it checks |
| --- | --- |
| Health | HTTP `200` and the expected status JSON |
| Valid customer | HTTP `200`, a valid prediction label, and a score between 0 and 1 |
| Missing fields | HTTP `422` |
| Invalid `poutcome` | HTTP `422` with only that field invalid |
| Invalid `housing` | HTTP `422` with only that field invalid |

All five tests passed during local development.

## Limitations and next steps

This is a learning experiment. I inspected the test results during earlier development before the final cross-validation version, so the test set is **not a pristine one-time final holdout**. I report its score as an exploratory result, not an unbiased final performance claim. For a stronger future evaluation, I would set aside a fresh untouched test set before making further modeling decisions. I would also check how performance changes over time and whether predicted scores need calibration before using them as probabilities in a real decision process.


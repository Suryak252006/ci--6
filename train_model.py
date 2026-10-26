
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def create_dataset():
    rng = np.random.default_rng(42)
    n = 300

    data = pd.DataFrame({
        "attendance": rng.integers(50, 101, n),
        "internal_marks": rng.integers(20, 101, n),
        "assignment_marks": rng.integers(30, 101, n),
        "previous_score": rng.integers(30, 101, n)
    })

    score = (
        0.25 * data["attendance"]
        + 0.35 * data["internal_marks"]
        + 0.20 * data["assignment_marks"]
        + 0.20 * data["previous_score"]
    )

    data["result"] = (score >= 60).astype(int)
    return data


def main():
    data = create_dataset()
    data.to_csv("student_results.csv", index=False)

    features = [
        "attendance",
        "internal_marks",
        "assignment_marks",
        "previous_score"
    ]

    X = data[features]
    y = data["result"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            max_iter=1000,
            random_state=42
        ))
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    joblib.dump(model, "student_result_model.pkl")

    metrics = {
        "accuracy": float(accuracy),
        "training_records": len(X_train),
        "testing_records": len(X_test)
    }

    with open("metrics.json", "w") as file:
        json.dump(metrics, file, indent=4)

    print("Model trained successfully")
    print("Accuracy:", round(accuracy, 4))
    print("Model and metrics saved")


if __name__ == "__main__":
    main()

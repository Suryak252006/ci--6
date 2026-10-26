
import json
import os
import unittest

import joblib
import pandas as pd


class TestMLPipeline(unittest.TestCase):

    def test_dataset_exists(self):
        self.assertTrue(os.path.exists("student_results.csv"))

    def test_model_exists(self):
        self.assertTrue(os.path.exists("student_result_model.pkl"))

    def test_metrics_exist(self):
        self.assertTrue(os.path.exists("metrics.json"))

    def test_accuracy(self):
        with open("metrics.json") as file:
            metrics = json.load(file)

        self.assertGreaterEqual(metrics["accuracy"], 0.85)

    def test_model_predicts(self):
        model = joblib.load("student_result_model.pkl")

        sample = pd.DataFrame([{
            "attendance": 90,
            "internal_marks": 85,
            "assignment_marks": 88,
            "previous_score": 80
        }])

        prediction = model.predict(sample)[0]
        self.assertEqual(int(prediction), 1)


if __name__ == "__main__":
    unittest.main()

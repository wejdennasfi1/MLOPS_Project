import unittest
from model_pipeline import train_model, prepare_data

class TestModelTraining(unittest.TestCase):
    def test_train_model(self):
        X_train, _, y_train, _, scaler = prepare_data()
        model = train_model(X_train, y_train)
        self.assertIsNotNone(model, "Model should not be None")
        self.assertTrue(hasattr(model, "predict"), "Model should have a predict method")

if __name__ == '__main__':
    unittest.main()

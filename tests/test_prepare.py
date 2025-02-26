import unittest
from model_pipeline import prepare_data

class TestPrepareData(unittest.TestCase):
    def test_prepare_data(self):
        X_train, X_test, y_train, y_test, scaler = prepare_data()
        self.assertGreater(len(X_train), 0, "X_train should not be empty")
        self.assertGreater(len(X_test), 0, "X_test should not be empty")
        self.assertEqual(len(X_train.shape), 2, "X_train should be 2D")

if __name__ == '__main__':
    unittest.main()

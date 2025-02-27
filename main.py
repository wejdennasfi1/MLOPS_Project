import argparse
from model_pipeline import ml_pipeline


def main():
    parser = argparse.ArgumentParser(description="Pipeline Machine Learning")
    parser.add_argument("--prepare", action="store_true", help="Prepare data")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate the model")
    parser.add_argument("--predict", action="store_true", help="Predict the model")
    args = parser.parse_args()
    ml_pipeline(prepare=args.prepare, train=args.train, evaluate=args.evaluate)


if __name__ == "__main__":
    main()

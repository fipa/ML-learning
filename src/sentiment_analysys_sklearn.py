import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from typing import Tuple, List
from joblib import Memory
import os
import argparse

from .models import get_estimators, get_transformer
from .input_data import get_input_data

CACHE_DIR = "./.cache"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)
memory = Memory(CACHE_DIR, verbose=0)

def load_and_clean_data(filepath: str) -> Tuple[pd.Series, pd.Series]:
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Data loaded successfully. Shape: {df.shape}")
    print("Sample data:\n", df.head())

    if 'text' not in df.columns or 'sentiment' not in df.columns:
        raise ValueError("CSV must contain 'text' and 'sentiment' columns.")

    if df.isnull().to_numpy().any():
        print("Warning: Data contains missing values. Handling them by dropping rows...")
        df = df.dropna()
        print(f"Shape after dropping NaNs: {df.shape}")

    x = df['text']
    y = df['sentiment']
    return x, y

def split_data(x: pd.Series, y: pd.Series, test_size: float, random_state: int) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    print("\nSplitting data into training and testing sets...")
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Training set size: {len(x_train)}")
    print(f"Test set size: {len(x_test)}")
    return x_train, x_test, y_train, y_test

def build_and_train_pipeline(x_train: pd.Series, y_train: pd.Series, transformer: CountVectorizer, estimator: BaseEstimator) -> Pipeline:
    print(f"\nDefining and training the model pipeline with {type(estimator).__name__}...")
    pipeline = Pipeline([
        ('transformer', transformer),
        ('classifier', estimator)
    ], memory=memory)
    pipeline.fit(x_train, y_train)
    print("Model training complete.")
    return pipeline

def evaluate_model(pipeline: Pipeline, x_test: pd.Series, y_test: pd.Series, model_name: str):
    print(f"\n--- Evaluating: {model_name} ---")
    y_pred = pipeline.predict(x_test)
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, zero_division=0))

def predict_new_data(pipeline: Pipeline, new_examples: List[str], model_name: str):
    print(f"\n--- Testing {model_name} with new examples ---")
    predictions = pipeline.predict(new_examples)
    for feedback, sentiment in zip(new_examples, predictions):
        print(f"'{feedback}' -> Predicted Sentiment: {sentiment}")

def main():

    parser = argparse.ArgumentParser(description="Train and evaluate sentiment analysis models.")
    parser.add_argument("--data-file", type=str, required=True,
                        help="Path to the CSV file containing training data (text, sentiment columns).")
    parser.add_argument("--examples-file", type=str, required=True,
                        help="Path to the CSV file containing example sentences (one per line).")
    args = parser.parse_args()

    data_filepath = args.data_file
    examples_filepath = args.examples_file


    print(f"\nLoading examples from {examples_filepath}...")
    try:
        input_data = get_input_data(examples_filepath)
    except FileNotFoundError:
        print(f"Error: Examples file not found at {examples_filepath}")
        print("Exiting due to missing examples file specified via command line.")
        return

    print(f"\nLoading training data from {data_filepath}...")
    try:
        x, y = load_and_clean_data(data_filepath)
    except FileNotFoundError:
        print(f"Error: Training data file not found at {data_filepath}")
        print("Exiting due to missing training file specified via command line.")
        return

    x_train, x_test, y_train, y_test = split_data(x, y, 0.3, 42)

    estimators = get_estimators()
    [model_name, transformer] = get_transformer()
    print(f"\nUsing transformer: {model_name}")

    for model_name, estimator in estimators:
        print(f"\n{'='*20} Training and Evaluating: {model_name} {'='*20}")
        trained_pipeline = build_and_train_pipeline(x_train, y_train, transformer, estimator)
        evaluate_model(trained_pipeline, x_test, y_test, model_name)
        predict_new_data(trained_pipeline, input_data, model_name)

    print("\n--- Script Finished ---")

if __name__ == "__main__":
    main()

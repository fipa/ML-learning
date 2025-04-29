import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from typing import Tuple, List
from joblib import Memory
import os

# Define a cache directory
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

def split_data(x: pd.Series, y: pd.Series, test_size: float = 0.3, random_state: int = 42) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    print("\nSplitting data into training and testing sets...")
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Training set size: {len(x_train)}")
    print(f"Test set size: {len(x_test)}")
    return x_train, x_test, y_train, y_test

def build_and_train_pipeline(x_train: pd.Series, y_train: pd.Series) -> Pipeline:
    print("\nDefining and training the model pipeline...")
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('clf', LogisticRegression(random_state=42, solver='liblinear'))
    ], memory=memory)
    pipeline.fit(x_train, y_train)
    print("Model training complete.")
    return pipeline

def evaluate_model(pipeline: Pipeline, x_test: pd.Series, y_test: pd.Series):
    print("\nEvaluating the model on the test set...")
    y_pred = pipeline.predict(x_test)
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

def predict_new_data(pipeline: Pipeline, new_examples: List[str]):
    print("\n--- Testing with new examples ---")
    predictions = pipeline.predict(new_examples)
    for feedback, sentiment in zip(new_examples, predictions):
        print(f"'{feedback}' -> Predicted Sentiment: {sentiment}")

def main():
    data_filepath = 'data/feedback.csv'
    new_feedback_examples = [
        "This is fantastic!",
        "What a waste of money.",
        "It's just average.",
        "Doesn't work as expected",
        "I'm happy but I would be happier if I hadn't buy the product",
    ]

    x, y = load_and_clean_data(data_filepath)
    x_train, x_test, y_train, y_test = split_data(x, y)
    trained_pipeline = build_and_train_pipeline(x_train, y_train)
    evaluate_model(trained_pipeline, x_test, y_test)
    predict_new_data(trained_pipeline, new_feedback_examples)

if __name__ == "__main__":
    main()

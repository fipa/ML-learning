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
    """Loads data from a CSV file, cleans it, and returns features (x) and target (y)."""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Data loaded successfully. Shape: {df.shape}")
    print("Sample data:\n", df.head())

    # Check if data is loaded correctly and has expected columns
    if 'text' not in df.columns or 'sentiment' not in df.columns:
        raise ValueError("CSV must contain 'text' and 'sentiment' columns.")

    # Handle missing values
    if df.isnull().to_numpy().any():
        print("Warning: Data contains missing values. Handling them by dropping rows...")
        df = df.dropna()
        print(f"Shape after dropping NaNs: {df.shape}")

    x = df['text']
    y = df['sentiment']
    return x, y

def split_data(x: pd.Series, y: pd.Series, test_size: float = 0.3, random_state: int = 42) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Splits the data into training and testing sets."""
    print("\nSplitting data into training and testing sets...")
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Training set size: {len(x_train)}")
    print(f"Test set size: {len(x_test)}")
    return x_train, x_test, y_train, y_test

def build_and_train_pipeline(x_train: pd.Series, y_train: pd.Series) -> Pipeline:
    """Builds the TF-IDF + Logistic Regression pipeline and trains it."""
    print("\nDefining and training the model pipeline...")
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('clf', LogisticRegression(random_state=42, solver='liblinear'))
    ], memory=memory)
    pipeline.fit(x_train, y_train)
    print("Model training complete.")
    return pipeline

def evaluate_model(pipeline: Pipeline, x_test: pd.Series, y_test: pd.Series):
    """Evaluates the model on the test set and prints the classification report."""
    print("\nEvaluating the model on the test set...")
    y_pred = pipeline.predict(x_test)
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

def predict_new_data(pipeline: Pipeline, new_examples: List[str]):
    """Predicts sentiment for new, unseen text examples."""
    print("\n--- Testing with new examples ---")
    predictions = pipeline.predict(new_examples)
    for feedback, sentiment in zip(new_examples, predictions):
        print(f"'{feedback}' -> Predicted Sentiment: {sentiment}")

def main():
    """Main function to run the sentiment analysis workflow."""
    # Define file path and new examples
    data_filepath = 'data/feedback.csv'
    new_feedback_examples = [
        "This is fantastic!",
        "What a waste of money.",
        "It's just average.",
        "Doesn't work as expected",
        "I'm happy but I would be happier if I hadn't buy the product",
    ]

    # 1. Load and Clean Data
    x, y = load_and_clean_data(data_filepath)

    # 2. Split Data
    x_train, x_test, y_train, y_test = split_data(x, y)

    # 3. Build and Train Pipeline
    trained_pipeline = build_and_train_pipeline(x_train, y_train)

    # 4. Evaluate Model
    evaluate_model(trained_pipeline, x_test, y_test)

    # 5. Predict on New Data
    predict_new_data(trained_pipeline, new_feedback_examples)

    print("\n--- Script Finished ---")

if __name__ == "__main__":
    main()

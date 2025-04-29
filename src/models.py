from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, BaseEstimator
from typing import List, Tuple

def get_transformer() -> Tuple[str, CountVectorizer]:
    transformer = ("TfidfVectorizer", TfidfVectorizer(stop_words='english'))
    return transformer
    
def get_estimators() -> List[Tuple[str, BaseEstimator]]:
    estimators = [
        ("Logistic Regression", LogisticRegression(random_state=42, solver='liblinear', max_iter=1000)),
        ("Linear SVC", LinearSVC(random_state=42, dual=True, max_iter=10000)),
        ("Multinomial Naive Bayes", MultinomialNB())
    ]
    return estimators

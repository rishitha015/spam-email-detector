from sklearn.model_selection import cross_val_score
import pandas as pd
import joblib
import re

from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


# Load dataset (address unchanged)
df = pd.read_csv(
    "C:/Users/mutte/OneDrive/Desktop/ML/data/spam.csv",
    encoding="latin-1"
)

df = df[['text', 'spam']]
df.columns = ['message', 'label']


# Clean function
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


df['message'] = df['message'].apply(clean_text)


# Build pipeline
model = Pipeline([
    ('tfidf', TfidfVectorizer(
        stop_words='english',
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9
    )),
    ('svc', LinearSVC(class_weight="balanced"))
])


# Cross Validation (check overall performance)
scores = cross_val_score(model, df['message'], df['label'], cv=5)
print("Cross Validation Accuracy:", scores.mean())


# Train-test split (final accuracy check)
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.2, random_state=42
)

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
print("Test Accuracy:", accuracy_score(y_test, y_pred))


# Save model
joblib.dump(model, "../model/spam_model.pkl")
print("Model saved successfully!")

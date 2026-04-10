import pandas as pd
import nltk
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv(
    "C:/Users/mutte/OneDrive/Desktop/ML/data/spam.csv", encoding="latin-1")

df = df[['text', 'spam']]
df.columns = ['message', 'label']

# Only if needed
# df['label'] = df['label'].map({'ham': 0, 'spam': 1})
# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.2, random_state=42
)

# Build pipeline (VERY IMPORTANT)
model = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('nb', MultinomialNB())
])

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
print("Accuracy:", accuracy_score(y_test, y_pred))

# Save model
joblib.dump(model, "../model/spam_model.pkl")

print("Model saved successfully!")

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
import joblib

# Load CSV dataset (NOT Excel)
df = pd.read_csv("keyphrase_training_large.csv")

df = df.dropna(subset=["phrase","label"])

phrases = df["phrase"].astype(str).tolist()
labels = df["label"].astype(int).tolist()

X_train, X_test, y_train, y_test = train_test_split(
    phrases, labels, test_size=0.2, random_state=42
)

model = Pipeline([
("tfidf",TfidfVectorizer(ngram_range=(1,2),stop_words="english")),
("clf",LogisticRegression(max_iter=1000))
])

model.fit(X_train,y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test,y_pred)
precision = precision_score(y_test,y_pred)
recall = recall_score(y_test,y_pred)
f1 = f1_score(y_test,y_pred)
cm = confusion_matrix(y_test,y_pred)

print("======================================")
print(" Model Performance Metrics")
print("======================================")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(y_test,y_pred))

joblib.dump(model,"keyphrase_model.joblib")
print("\nModel trained and saved as 'keyphrase_model.joblib'")

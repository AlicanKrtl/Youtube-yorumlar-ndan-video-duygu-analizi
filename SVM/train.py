import torch
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd

# Step 1: Load and preprocess data
data = pd.read_csv("/home/alican/Documents/Studies/begüm_proje/merged_data.csv")

comments = data["comments"].apply(lambda x: " ".join(eval(x)) if len(x)>0 else "").values
emotion = data["emotion"].values


# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(comments, emotion, test_size=0.2, random_state=42)

# Feature extraction using TF-IDF
tfidf_vectorizer = TfidfVectorizer()
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Training SVM model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_tfidf, y_train)

# Evaluating the model
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return accuracy, report

accuracy, report = evaluate_model(svm_model, X_test_tfidf, y_test)

# Saving the model and results to a text file
with open("svm_model_results.txt", "w") as f:
    f.write("Best SVM model:\n")
    f.write(str(svm_model))
    f.write("\n\nEvaluation Results:\n")
    f.write("Accuracy: {}\n".format(accuracy))
    f.write("Classification Report:\n{}\n".format(report))
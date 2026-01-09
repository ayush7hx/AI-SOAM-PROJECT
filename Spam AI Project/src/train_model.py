import os
import sys
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# Add src to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_preprocessing import load_data, preprocess_data

def train_model():
    """Train the email spam detection model"""
    print("🔄 Loading and preprocessing email data...")

    # Load and preprocess data
    df = load_data()
    df = preprocess_data(df)

    print(f"✅ Email data loaded: {len(df)} messages")
    print(f"   - Spam emails: {df['label'].sum()}")
    print(f"   - Ham emails: {len(df) - df['label'].sum()}")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_message'], df['label'], test_size=0.2, random_state=42
    )

    print("🔄 Training model...")

    # Vectorize text
    vectorizer = CountVectorizer(stop_words='english', max_features=5000)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train model
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    # Evaluate
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"✅ Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

    # Save model and vectorizer
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(model, os.path.join(models_dir, 'spam_model.pkl'))
    joblib.dump(vectorizer, os.path.join(models_dir, 'vectorizer.pkl'))

    print("✅ Model saved to models/ directory")

    return model, vectorizer

if __name__ == "__main__":
    print("="*50)
    print("🧠 TRAINING EMAIL SPAM DETECTOR AI")
    print("Created by: Ayush7hx")
    print("="*50)
    train_model()
    print("="*50)
    print("✅ Training completed!")
    print("="*50)
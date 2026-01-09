# src/train_model.py
"""
Main training script for spam classifier
Run this file to train and save the model
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Import our preprocessing module
from src.data_preprocessing import load_and_prepare_data

def train_spam_classifier():
    """
    Train and save spam classification model
    """
    print("="*60)
    print("SPAM DETECTOR AI - TRAINING MODULE")
    print("="*60)
    
    # 1. Load and prepare data
    print("\n[1/4] Loading data...")
    messages, labels = load_and_prepare_data('data/spam.csv')
    print(f"   Loaded {len(messages)} messages")
    print(f"   Spam: {sum(labels)}, Ham: {len(labels)-sum(labels)}")
    
    # 2. Vectorize text
    print("\n[2/4] Vectorizing text...")
    vectorizer = CountVectorizer(max_features=3000)
    X = vectorizer.fit_transform(messages).toarray()
    y = labels
    print(f"   Created {X.shape[1]} features")
    
    # 3. Split data
    print("\n[3/4] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   Training: {X_train.shape[0]} samples")
    print(f"   Testing: {X_test.shape[0]} samples")
    
    # 4. Train model
    print("\n[4/4] Training model...")
    model = MultinomialNB()
    model.fit(X_train, y_train)
    
    # 5. Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n✅ Model trained successfully!")
    print(f"   Accuracy: {accuracy:.4f}")
    
    # 6. Save model
    print("\n💾 Saving model...")
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/spam_model.pkl')
    joblib.dump(vectorizer, 'models/vectorizer.pkl')
    print("   Model saved to 'models/' directory")
    
    # 7. Show sample predictions
    print("\n📋 Sample Predictions:")
    print("-"*40)
    
    test_samples = [
        ("Free entry to win £1000 cash!", "SPAM"),
        ("Hey, let's meet for coffee tomorrow", "HAM"),
        ("URGENT! Your account has been compromised", "SPAM"),
        ("Mom, what's for dinner tonight?", "HAM")
    ]
    
    for text, expected in test_samples:
        vector = vectorizer.transform([text]).toarray()
        pred = model.predict(vector)[0]
        result = "✅" if ("SPAM" if pred == 1 else "HAM") == expected else "❌"
        print(f"{result} '{text[:30]}...' → {'SPAM' if pred == 1 else 'HAM'} (Expected: {expected})")
    
    print("\n" + "="*60)
    print("Training completed! Run 'python main.py' to use the model.")
    print("="*60)

if __name__ == "__main__":
    train_spam_classifier()
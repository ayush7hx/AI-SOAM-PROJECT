import pandas as pd
import re
import nltk
import os
from nltk.corpus import stopwords

# Download stopwords if not already downloaded
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

def load_data(filepath=None):
    """Load and clean the email spam dataset"""
    if filepath is None:
        # Get path relative to this file
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        filepath = os.path.join(base_dir, 'data', 'email_spam.csv')
    
    df = pd.read_csv(filepath)
    # Convert labels to binary
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    return df

def clean_text(text):
    """Clean and preprocess email text data"""
    # Convert to lowercase
    text = text.lower()
    # Remove email addresses
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    # Remove phone numbers
    text = re.sub(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', '', text)
    # Remove special characters but keep spaces
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

def preprocess_data(df):
    """Apply text cleaning to the entire dataset"""
    df['clean_message'] = df['message'].apply(clean_text)
    return df
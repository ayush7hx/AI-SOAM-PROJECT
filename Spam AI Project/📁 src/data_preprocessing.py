# src/data_preprocessing.py
"""
Data preprocessing module for spam detection
Contains functions for cleaning and preparing text data
"""

import re
import nltk
from nltk.corpus import stopwords

# Download stopwords once
try:
    stop_words = set(stopwords.words('english'))
except:
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))

def clean_text(text):
    """
    Clean and preprocess text data
    
    Args:
        text (str): Raw text message
    
    Returns:
        str: Cleaned text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Remove stopwords
    words = text.split()
    words = [word for word in words if word not in stop_words]
    
    return ' '.join(words)

def load_and_prepare_data(filepath):
    """
    Load dataset and prepare for training
    
    Args:
        filepath (str): Path to CSV file
    
    Returns:
        tuple: (messages, labels)
    """
    import pandas as pd
    
    df = pd.read_csv(filepath, encoding='latin-1')
    df = df[['v1', 'v2']]
    df.columns = ['label', 'message']
    
    # Convert labels
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    
    # Clean messages
    df['clean_message'] = df['message'].apply(clean_text)
    
    return df['clean_message'].tolist(), df['label'].tolist()

if __name__ == "__main__":
    # Test the functions
    test_text = "Free WINNER!! You have won $1000! Call NOW!"
    print("Original:", test_text)
    print("Cleaned:", clean_text(test_text))
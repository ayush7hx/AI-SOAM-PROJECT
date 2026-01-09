import os
import joblib
import sys
from src.data_preprocessing import clean_text

class SpamPredictor:
    """Spam detection predictor class"""

    def __init__(self, model_path=None, vectorizer_path=None):
        """Initialize the predictor with trained model and vectorizer"""
        if model_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, 'models', 'spam_model.pkl')
        if vectorizer_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            vectorizer_path = os.path.join(base_dir, 'models', 'vectorizer.pkl')
        
        try:
            self.model = joblib.load(model_path)
            self.vectorizer = joblib.load(vectorizer_path)
            print("✅ Model loaded successfully")
        except FileNotFoundError:
            print("❌ Model files not found. Please train the model first.")
            print("   Run: python src/train_model.py")
            sys.exit(1)

    def predict_single(self, message):
        """Predict if a single message is spam"""
        # Clean the message
        clean_msg = clean_text(message)
        # Vectorize
        msg_vec = self.vectorizer.transform([clean_msg])
        # Predict
        prediction = self.model.predict(msg_vec)[0]
        probability = self.model.predict_proba(msg_vec)[0]

        result = {
            'message': message,
            'prediction': 'SPAM' if prediction == 1 else 'HAM',
            'confidence': max(probability),
            'probabilities': {
                'HAM': probability[0],
                'SPAM': probability[1]
            }
        }

        return result

    def predict(self, message):
        """Alias for predict_single for backward compatibility"""
        return self.predict_single(message)

    def predict_batch(self, messages):
        """Predict for multiple messages"""
        results = []
        for msg in messages:
            result = self.predict_single(msg)
            results.append(result)
        return results

    def get_model_info(self):
        """Get information about the trained model"""
        return {
            'model_type': type(self.model).__name__,
            'vocabulary_size': len(self.vectorizer.vocabulary_),
            'features': self.vectorizer.get_feature_names_out()[:10].tolist()  # First 10 features
        }
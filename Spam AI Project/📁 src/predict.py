# src/predict.py
"""
Prediction module for spam classifier
Contains functions to load model and make predictions
"""

import joblib
import os

class SpamPredictor:
    def __init__(self):
        """Initialize predictor by loading model and vectorizer"""
        self.model = None
        self.vectorizer = None
        self.load_models()
    
    def load_models(self):
        """Load trained model and vectorizer"""
        try:
            self.model = joblib.load('models/spam_model.pkl')
            self.vectorizer = joblib.load('models/vectorizer.pkl')
            print("✅ Models loaded successfully")
        except FileNotFoundError:
            print("❌ Model files not found. Please run training first.")
            print("   Run: python src/train_model.py")
    
    def predict(self, message):
        """
        Predict if a message is spam or ham
        
        Args:
            message (str): Input text message
        
        Returns:
            dict: Prediction results with confidence
        """
        if self.model is None or self.vectorizer is None:
            return {"error": "Model not loaded"}
        
        # Vectorize message
        vector = self.vectorizer.transform([message]).toarray()
        
        # Make prediction
        prediction = self.model.predict(vector)[0]
        probability = self.model.predict_proba(vector)[0]
        
        # Prepare result
        result = {
            "message": message[:100] + "..." if len(message) > 100 else message,
            "prediction": "SPAM" if prediction == 1 else "HAM",
            "confidence": float(max(probability)),
            "probabilities": {
                "HAM": float(probability[0]),
                "SPAM": float(probability[1])
            }
        }
        
        return result

def test_prediction():
    """Test function to verify predictions"""
    predictor = SpamPredictor()
    
    test_messages = [
        "Congratulations! You've won a free iPhone. Click here to claim!",
        "Hey, are we still meeting for lunch tomorrow?",
        "URGENT: Your bank account needs verification",
        "Mom, can you pick me up from school at 4?"
    ]
    
    print("\n🧪 Testing Predictions:")
    print("="*50)
    
    for msg in test_messages:
        result = predictor.predict(msg)
        if "error" not in result:
            print(f"\n📩 Message: {msg}")
            print(f"   Prediction: {result['prediction']}")
            print(f"   Confidence: {result['confidence']:.2%}")
        else:
            print(result["error"])

if __name__ == "__main__":
    test_prediction()
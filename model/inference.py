import os
import joblib
from pathlib import Path

class LegalAssistantModel:
    def __init__(self, model_path='saved_models'):
        """Initialize the model and vectorizer."""
        self.model = joblib.load(os.path.join(model_path, 'legal_classifier.joblib'))
        self.vectorizer = joblib.load(os.path.join(model_path, 'tfidf_vectorizer.joblib'))
    
    def predict(self, text):
        """Predict the class of the input text."""
        if not isinstance(text, list):
            text = [text]
            
        # Transform input text using the trained vectorizer
        text_vec = self.vectorizer.transform(text)
        
        # Make prediction
        prediction = self.model.predict(text_vec)
        return prediction[0] if len(prediction) == 1 else prediction

def load_model(model_path='saved_models'):
    """Load the trained model and vectorizer."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model directory not found at {model_path}")
    
    return LegalAssistantModel(model_path)

def get_legal_advice(text, model=None):
    """Get legal advice based on the input text."""
    if model is None:
        model = load_model()
    
    # Get prediction
    prediction = model.predict(text)
    
    # TODO: Add more sophisticated response generation
    # This is a placeholder response
    response = {
        'input_text': text,
        'predicted_category': prediction,
        'confidence': 0.95,  # Placeholder value
        'suggested_actions': [
            'Consult with a lawyer for specific advice',
            'Review relevant legal documents',
            'Keep records of all related communications'
        ]
    }
    
    return response

if __name__ == "__main__":
    # Example usage
    test_text = "I was injured at work and my employer is not providing compensation."
    result = get_legal_advice(test_text)
    print("Legal Assistance Result:")
    print(f"Input: {result['input_text']}")
    print(f"Category: {result['predicted_category']}")
    print("Suggested Actions:")
    for action in result['suggested_actions']:
        print(f"- {action}")

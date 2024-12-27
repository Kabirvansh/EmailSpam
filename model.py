import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from .utils import preprocessor

class SpamDetector:
    def __init__(self):
        self.vectorizer = None
        self.model = None
    
    def train(self, df):
        """Train the spam detection model."""
        # Initialize and train the vectorizer
        self.vectorizer = CountVectorizer(preprocessor=preprocessor)
        X = self.vectorizer.fit_transform(df['content'])
        
        # Train the model
        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(X, df['category'])
    
    def predict_proba(self, text):
        """Predict the probability of a text being spam."""
        if self.vectorizer is None or self.model is None:
            raise ValueError("Model not trained yet!")
        
        # Transform the text
        X = self.vectorizer.transform([text])
        
        # Get probability scores
        probabilities = self.model.predict_proba(X)
        
        # Return spam probability (second class)
        return probabilities[0][1]
    
    def save_model(self, vectorizer_path, model_path):
        """Save the trained model and vectorizer to files."""
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
    
    @classmethod
    def load_model(cls, vectorizer_path, model_path):
        """Load a trained model and vectorizer from files."""
        detector = cls()
        with open(vectorizer_path, 'rb') as f:
            detector.vectorizer = pickle.load(f)
        with open(model_path, 'rb') as f:
            detector.model = pickle.load(f)
        return detector

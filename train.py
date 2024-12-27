import pandas as pd
from utils import read_spam, read_ham
from model import SpamDetector

def train_and_save_model():
    """Train the spam detection model and save it to disk."""
    print("Loading email dataset...")
    ham = read_ham()
    spam = read_spam()
    
    print("Preparing training data...")
    df = pd.DataFrame.from_records(ham)
    df = pd.concat([df, pd.DataFrame.from_records(spam)], ignore_index=True)
    
    print("Training model...")
    detector = SpamDetector()
    detector.train(df)
    
    print("Saving model...")
    detector.save_model('vectorizer.pkl', 'model.pkl')
    print("Training complete! Model saved to disk.")

if __name__ == "__main__":
    train_and_save_model()

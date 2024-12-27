import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from model import SpamDetector

def read_spam():
    category = 'spam'
    directory = 'enron1/spam'
    return read_category(category, directory)

def read_ham():
    category = 'ham'
    directory = 'enron1/ham'
    return read_category(category, directory)

def read_category(category, directory):
    emails = []
    for filename in os.listdir(directory):
        if not filename.endswith(".txt"):
            continue
        with open(os.path.join(directory, filename), 'r') as fp:
            try:
                content = fp.read()
                emails.append({'name': filename, 'content': content, 'category': category})
            except:
                print(f'skipped {filename}')
    return emails

def train_and_save_model():
    print("Loading email dataset...")
    # Load the data
    ham = read_ham()
    spam = read_spam()
    
    print("Preparing training data...")
    df = pd.DataFrame.from_records(ham)
    df = pd.concat([df, pd.DataFrame.from_records(spam)], ignore_index=True)
    
    # Create and train the detector
    detector = SpamDetector()
    
    # Initialize and train the vectorizer
    print("Vectorizing text data...")
    vectorizer = CountVectorizer(preprocessor=detector.preprocessor)
    X = vectorizer.fit_transform(df['content'])
    
    # Split the dataset
    print("Splitting into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, df['category'], test_size=0.2, random_state=42
    )
    
    # Train the model
    print("Training model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    # Make predictions
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    
    # Print evaluation metrics
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {accuracy * 100:.2f}%")
    
    print("\nConfusion Matrix:")
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(conf_matrix)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Print feature importance analysis
    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.coef_[0]
    
    # Get feature importance
    feature_importance = list(zip(feature_names, coefficients))
    feature_importance = sorted(feature_importance, key=lambda x: abs(x[1]), reverse=True)
    
    print("\nTop 10 most important features (words) and their coefficients:")
    for word, coef in feature_importance[:10]:
        print(f"{word}: {coef:.4f}")
    
    # Separate positive and negative features
    positive_features = [(word, coef) for word, coef in feature_importance if coef > 0]
    negative_features = [(word, coef) for word, coef in feature_importance if coef < 0]
    
    positive_features = sorted(positive_features, key=lambda x: abs(x[1]), reverse=True)
    negative_features = sorted(negative_features, key=lambda x: abs(x[1]), reverse=True)
    
    print("\nTop 10 positive features (indicative of spam):")
    for word, coef in positive_features[:10]:
        print(f"{word}: {coef:.4f}")
    
    print("\nTop 10 negative features (indicative of ham):")
    for word, coef in negative_features[:10]:
        print(f"{word}: {coef:.4f}")
    
    # Save the final model and vectorizer
    print("\nSaving model and vectorizer...")
    detector.vectorizer = vectorizer
    detector.model = model
    detector.save_model('vectorizer.pkl', 'model.pkl')
    
    print("Training complete! Model saved to disk.")
    return detector

if __name__ == "__main__":
    train_and_save_model()
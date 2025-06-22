import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle
import os
import numpy as np

nltk.download('punkt')

class SentimentModelTrainer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.model = LogisticRegression(max_iter=1000, class_weight='balanced')
        
    def load_data(self, filepath='data/training_data.csv'):
        """Load and preprocess the training data"""
        df = pd.read_csv(filepath)
        
        # Convert label columns to lists
        df['positive'] = df['positive'].str.split(',')
        df['negative'] = df['negative'].str.split(',')
        
        # Fill empty lists
        df['positive'] = df['positive'].apply(lambda x: x if isinstance(x, list) else [])
        df['negative'] = df['negative'].apply(lambda x: x if isinstance(x, list) else [])
        
        return df
    
    def prepare_features(self, df):
        """Prepare features and labels"""
        # Create labels (1 for positive, -1 for negative, 0 for neutral)
        df['label'] = df.apply(
            lambda row: 1 if len(row['positive']) > len(row['negative']) else 
            (-1 if len(row['negative']) > len(row['positive']) else 0), 
            axis=1
        )
        
        # Text features
        X_text = self.vectorizer.fit_transform(df['text'])
        
        # Additional features
        df['pos_count'] = df['positive'].apply(len)
        df['neg_count'] = df['negative'].apply(len)
        df['word_count'] = df['text'].apply(lambda x: len(word_tokenize(x)))
        
        # Combine features
        X_extra = df[['pos_count', 'neg_count', 'word_count']].values
        X = np.hstack([X_text.toarray(), X_extra])
        
        return X, df['label']
    
    def train(self):
        """Train the model"""
        df = self.load_data()
        X, y = self.prepare_features(df)
        
        # Train-test split (even though we have separate test data)
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        print("Validation Results:")
        print(classification_report(y_val, self.model.predict(X_val)))
        
    def save_model(self, filename='models/sentiment_model.pkl'):
        """Save the trained model"""
        os.makedirs('models', exist_ok=True)
        with open(filename, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'vectorizer': self.vectorizer
            }, f)
        print(f"Model saved to {filename}")

if __name__ == "__main__":
    trainer = SentimentModelTrainer()
    trainer.train()
    trainer.save_model()
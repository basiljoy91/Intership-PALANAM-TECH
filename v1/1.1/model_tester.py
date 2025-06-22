import pickle
import pandas as pd
from nltk.tokenize import word_tokenize
from tabulate import tabulate
import numpy as np

class SentimentModelTester:
    def __init__(self, model_path='models/sentiment_model.pkl'):
        """Load the trained model"""
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
            self.model = data['model']
            self.vectorizer = data['vectorizer']
    
    def analyze_text(self, text):
        """Analyze text and return detailed results"""
        # Tokenize
        words = word_tokenize(text.lower())
        
        # Count positive/negative words (for display)
        positive_words = [w for w in words if w in self.positive_words]
        negative_words = [w for w in words if w in self.negative_words]
        
        # Prepare features
        X_text = self.vectorizer.transform([text])
        pos_count = len(positive_words)
        neg_count = len(negative_words)
        word_count = len(words)
        
        X_extra = np.array([[pos_count, neg_count, word_count]])
        X = np.hstack([X_text.toarray(), X_extra])
        
        # Predict
        prediction = self.model.predict(X)[0]
        proba = self.model.predict_proba(X)[0]
        
        sentiment = {
            -1: "Negative",
            0: "Neutral",
            1: "Positive"
        }[prediction]
        
        return {
            'text': text,
            'positive_words': positive_words,
            'negative_words': negative_words,
            'word_count': word_count,
            'pos_count': pos_count,
            'neg_count': neg_count,
            'sentiment': sentiment,
            'confidence': max(proba),
            'pos_ratio': pos_count / word_count if word_count > 0 else 0,
            'neg_ratio': neg_count / word_count if word_count > 0 else 0,
            'overall_score': (pos_count - neg_count) / word_count if word_count > 0 else 0
        }
    
    def display_results(self, results):
        """Display analysis results in formatted tables"""
        # Positive words table
        if results['positive_words']:
            pos_df = pd.DataFrame({
                'Word': results['positive_words'],
                'Score': [1]*len(results['positive_words'])
            })
            pos_table = tabulate(pos_df, headers='keys', tablefmt='pretty', showindex=False)
        else:
            pos_table = "No positive words found"
        
        # Negative words table
        if results['negative_words']:
            neg_df = pd.DataFrame({
                'Word': results['negative_words'],
                'Score': [-1]*len(results['negative_words'])
            })
            neg_table = tabulate(neg_df, headers='keys', tablefmt='pretty', showindex=False)
        else:
            neg_table = "No negative words found"
        
        # Metrics table
        metrics_df = pd.DataFrame({
            'Metric': ['Positive Words', 'Negative Words', 'Pos/Neg Ratio', 
                      'Overall Score', 'Sentiment', 'Confidence'],
            'Value': [
                f"{results['pos_count']} ({results['pos_ratio']:.1%})",
                f"{results['neg_count']} ({results['neg_ratio']:.1%})",
                f"{results['pos_ratio']/results['neg_ratio']:.1f}" if results['neg_ratio'] > 0 else "∞",
                f"{results['overall_score']:.2f}",
                results['sentiment'],
                f"{results['confidence']:.1%}"
            ]
        })
        metrics_table = tabulate(metrics_df, headers='keys', tablefmt='pretty', showindex=False)
        
        print("\n=== TEXT ANALYSIS RESULTS ===")
        print(f"\nInput Text: {results['text']}")
        
        print("\n=== POSITIVE WORDS ===")
        print(pos_table)
        
        print("\n=== NEGATIVE WORDS ===")
        print(neg_table)
        
        print("\n=== SENTIMENT METRICS ===")
        print(metrics_table)

if __name__ == "__main__":
    tester = SentimentModelTester()
    
    print("Sentiment Analysis Tester")
    print("Enter 'quit' to exit\n")
    
    while True:
        text = input("\nEnter text to analyze: ").strip()
        if text.lower() == 'quit':
            break
        
        if not text:
            print("Please enter some text")
            continue
            
        results = tester.analyze_text(text)
        tester.display_results(results)
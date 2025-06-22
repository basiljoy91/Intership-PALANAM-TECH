import tensorflow as tf
import numpy as np
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from tensorflow.keras.preprocessing.sequence import pad_sequences

nltk.download('stopwords')

# Load model and tokenizer
model = tf.keras.models.load_model('sentiment_model.h5')
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Settings
max_length = 100
stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer('english')

def clean_text(text):
    text = re.sub(r'@\S+|https?:\S+|http?:\S|[^A-Za-z0-9\s]', ' ', text.lower())
    tokens = [stemmer.stem(word) for word in text.split() if word not in stop_words]
    return ' '.join(tokens)

def predict_sentiment(paragraph):
    cleaned = clean_text(paragraph)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded_seq = pad_sequences(seq, maxlen=max_length, padding='post')
    prediction = model.predict(padded_seq)[0][0]
    score = int(round(prediction * 9 + 1))  # scale 0-1 → 1 to 10
    return score

# Terminal input
if __name__ == "__main__":
    paragraph = input("Enter a paragraph: ")
    score = predict_sentiment(paragraph)
    print(f"Predicted Sentiment Score (1–10): {score}")

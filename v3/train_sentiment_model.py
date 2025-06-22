import pandas as pd
import numpy as np
import re
import string
import nltk
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

# Download NLTK stopwords
nltk.download('stopwords')

# Load dataset
df = pd.read_csv('dataset.csv', encoding='latin', header=None)
df.columns = ['sentiment', 'id', 'date', 'query', 'user_id', 'text']
df = df[['sentiment', 'text']]

# Map sentiment to labels
df['sentiment'] = df['sentiment'].map({0: 0, 4: 1})  # 0 = Negative, 1 = Positive

# Clean text
stemmer = SnowballStemmer('english')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'@\S+|https?:\S+|http?:\S|[^A-Za-z0-9\s]', ' ', text.lower())
    tokens = [stemmer.stem(word) for word in text.split() if word not in stop_words]
    return ' '.join(tokens)

df['text'] = df['text'].apply(clean_text)

# Tokenization
vocab_size = 10000
max_length = 100
oov_tok = "<OOV>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(df['text'])
sequences = tokenizer.texts_to_sequences(df['text'])
padded = pad_sequences(sequences, maxlen=max_length, padding='post')

# Split data
X_train, X_test, y_train, y_test = train_test_split(padded, df['sentiment'], test_size=0.2, random_state=42)

# Model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, 64, input_length=max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Train
model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test), batch_size=128)

# Save model and tokenizer
model.save("sentiment_model.h5")
import pickle
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Model and tokenizer saved.")

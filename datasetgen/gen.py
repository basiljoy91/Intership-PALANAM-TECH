import pandas as pd
import random
from faker import Faker
import os

# Initialize faker for realistic text generation
fake = Faker()

# Create directories if they don't exist
os.makedirs('data', exist_ok=True)

# Sentiment word banks
positive_words = [
    "amazing", "excellent", "wonderful", "outstanding", "brilliant",
    "fantastic", "superb", "impressive", "delightful", "exceptional",
    "perfect", "ideal", "splendid", "magnificent", "stellar",
    "awesome", "phenomenal", "remarkable", "terrific", "fabulous",
    "incredible", "perfect", "satisfactory", "pleasing", "admirable",
    "recommended", "highly", "great", "good", "nice",
    "valuable", "worthy", "decent", "acceptable", "reasonable",
    "positive", "happy", "joyful", "delicious", "tasty",
    "powerful", "fast", "quick", "efficient", "effective",
    "user-friendly", "intuitive", "simple", "easy", "convenient"
]

negative_words = [
    "terrible", "awful", "horrible", "poor", "bad",
    "disappointing", "frustrating", "annoying", "irritating", "maddening",
    "unacceptable", "subpar", "inferior", "mediocre", "lousy",
    "cheap", "flimsy", "weak", "broken", "defective",
    "slow", "laggy", "unresponsive", "buggy", "glitchy",
    "overpriced", "expensive", "costly", "unreasonable", "unfair",
    "difficult", "complicated", "confusing", "frustrating", "annoying",
    "ugly", "unattractive", "unpleasant", "disgusting", "repulsive",
    "noisy", "loud", "harsh", "uncomfortable", "painful",
    "dangerous", "risky", "harmful", "toxic", "unhealthy"
]

neutral_words = [
    "average", "ordinary", "standard", "normal", "regular",
    "typical", "common", "usual", "expected", "predictable",
    "adequate", "sufficient", "satisfactory", "passable", "tolerable",
    "moderate", "medium", "balanced", "neutral", "indifferent",
    "fine", "okay", "alright", "acceptable", "reasonable"
]

# Generate 1000 samples (800 training, 200 testing)
def generate_sample():
    sentiment_type = random.choice(['positive', 'negative', 'mixed', 'neutral'])
    
    if sentiment_type == 'positive':
        pos_count = random.randint(2, 4)
        neg_count = 0
        pos = random.sample(positive_words, pos_count)
        neg = []
        text = fake.sentence() + " " + " ".join([f"{word}!" for word in pos]) + " " + fake.sentence()
        
    elif sentiment_type == 'negative':
        pos_count = 0
        neg_count = random.randint(2, 4)
        pos = []
        neg = random.sample(negative_words, neg_count)
        text = fake.sentence() + " " + " ".join([f"{word}." for word in neg]) + " " + fake.sentence()
        
    elif sentiment_type == 'mixed':
        pos_count = random.randint(1, 2)
        neg_count = random.randint(1, 2)
        pos = random.sample(positive_words, pos_count)
        neg = random.sample(negative_words, neg_count)
        text = fake.sentence() + " " + f"{pos[0]} but {neg[0]}." + " " + fake.sentence()
        
    else:  # neutral
        pos_count = 0
        neg_count = 0
        pos = []
        neg = []
        text = fake.sentence() + " " + random.choice(neutral_words) + " " + fake.sentence()
    
    return {
        'text': text,
        'positive': ",".join(pos),
        'negative': ",".join(neg)
    }

# Generate all samples
samples = [generate_sample() for _ in range(1000)]

# Convert to DataFrame and split
df = pd.DataFrame(samples)
train_df = df[:800]
test_df = df[800:]

# Save to CSV
train_df.to_csv('data/training_data.csv', index=False)
test_df.to_csv('data/testing_data.csv', index=False)

print("Datasets created:")
print(f"- Training data: {len(train_df)} samples (data/training_data.csv)")
print(f"- Testing data: {len(test_df)} samples (data/testing_data.csv)")
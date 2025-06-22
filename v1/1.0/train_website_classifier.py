import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from transformers import DistilBERTTokenizerFast, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset

# Load dataset
df = pd.read_csv("web_content_dataset.csv")
le = LabelEncoder()
df['label_enc'] = le.fit_transform(df['label'])  # Encode labels

# Split dataset
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label_enc'])

# Convert to Hugging Face Dataset
train_ds = Dataset.from_pandas(train_df[['text', 'label_enc']])
test_ds = Dataset.from_pandas(test_df[['text', 'label_enc']])

# Load tokenizer and model
tokenizer = DistilBERTTokenizerFast.from_pretrained('distilbert-base-uncased')

def tokenize(batch):
    return tokenizer(batch['text'], padding=True, truncation=True, max_length=512)

train_ds = train_ds.map(tokenize, batched=True)
test_ds = test_ds.map(tokenize, batched=True)

# Load model
model = DistilBertForSequenceClassification.from_pretrained(
    'distilbert-base-uncased',
    num_labels=len(le.classes_)
)

# Training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=100,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir='./logs',
    logging_steps=10,
    load_best_model_at_end=True
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=test_ds
)

# Train the model
trainer.train()

# Save model and label encoder
model.save_pretrained("website_classifier_model")
tokenizer.save_pretrained("website_classifier_model")
import pickle
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

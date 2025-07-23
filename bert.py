import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import BertTokenizer, TFBertModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model

# Load dataset
df = pd.read_csv("Phishing_Legitimate_full.csv")
df = df[['id', 'CLASS_LABEL']]  # id = URL

# Encode labels
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['CLASS_LABEL'])  # 0 = legitimate, 1 = phishing

# Split data
X_train, X_test, y_train, y_test = train_test_split(df['id'], df['label'], test_size=0.2, random_state=42)

# Tokenize URLs
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_encodings = tokenizer(list(X_train), truncation=True, padding=True, max_length=64, return_tensors='tf')
test_encodings = tokenizer(list(X_test), truncation=True, padding=True, max_length=64, return_tensors='tf')

# BERT Model
bert = TFBertModel.from_pretrained('bert-base-uncased')
input_ids = Input(shape=(64,), dtype=tf.int32, name="input_ids")
attention_mask = Input(shape=(64,), dtype=tf.int32, name="attention_mask")

outputs = bert(input_ids, attention_mask=attention_mask)[1]
dense = Dense(1, activation='sigmoid')(outputs)

model = Model(inputs=[input_ids, attention_mask], outputs=dense)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
model.fit(
    x={'input_ids': train_encodings['input_ids'], 'attention_mask': train_encodings['attention_mask']},
    y=np.array(y_train),
    validation_data=(
        {'input_ids': test_encodings['input_ids'], 'attention_mask': test_encodings['attention_mask']},
        np.array(y_test)
    ),
    epochs=3,
    batch_size=32
)

# Save model and tokenizer
model.save("bert_model")
tokenizer.save_pretrained("bert_tokenizer")

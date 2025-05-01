import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# Load cleaned data
df = pd.read_csv("cleaned_data.csv")

# Use a smaller dataset for faster training
df = df.sample(n=200000, random_state=42)  # Reduce dataset size for training speed

# Convert labels (0 = Negative, 1 = Positive)
df['target'] = df['target'].apply(lambda x: 1 if x == 4 else 0)

# Tokenization
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(df['cleaned_text'])
X = tokenizer.texts_to_sequences(df['cleaned_text'])

# Padding sequences to ensure uniform input size
X = pad_sequences(X, maxlen=100)

# Save tokenizer for later use
pickle.dump(tokenizer, open("tokenizer.pkl", "wb"))

# Split data into training and testing sets
y = df['target'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build LSTM model
model = Sequential([
    Embedding(input_dim=5000, output_dim=128, input_length=100),
    LSTM(64, return_sequences=True),
    Dropout(0.3),
    LSTM(32),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=5, batch_size=64, validation_data=(X_test, y_test))

# Save the trained model
model.save("lstm_model.h5")

# Evaluate model performance
loss, accuracy = model.evaluate(X_test, y_test)
print(f"LSTM training completed; Accuracy: {accuracy:.4f}")

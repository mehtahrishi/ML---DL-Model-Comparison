from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Embedding

# Define the LSTM model
model = Sequential([
    Embedding(input_dim=5000, output_dim=128, input_length=100),
    LSTM(128, return_sequences=True),
    Dropout(0.2),
    LSTM(64),
    Dense(1, activation='sigmoid')
])

# **Fix: Explicitly build the model**
model.build(input_shape=(None, 100))

# Optional: Print model summary to check the structure
print(model.summary())

# Generate Model Architecture Diagram
plot_model(model, to_file="lstm_architecture.png", show_shapes=True, show_layer_names=True)

print("LSTM Model Architecture saved as lstm_architecture.png")

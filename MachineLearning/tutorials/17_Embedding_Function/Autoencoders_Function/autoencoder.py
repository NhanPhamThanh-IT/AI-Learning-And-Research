import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# Generate some sample data (e.g., high-dimensional data)
# In a real scenario, this would be your actual dataset
np.random.seed(42)
X = np.random.rand(1000, 50) # 1000 samples, 50 features

# Scale data to be between 0 and 1, often helpful for Autoencoders
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, X_test = train_test_split(X_scaled, test_size=0.2, random_state=42)

# Define the Autoencoder model
input_dim = X_train.shape[1]
encoding_dim = 10 # Dimensionality of the embedding (bottleneck layer)

# Encoder layers
input_layer = Input(shape=(input_dim,))
encoder = Dense(30, activation='relu')(input_layer)
encoder = Dense(encoding_dim, activation='relu')(encoder) # Bottleneck layer

# Decoder layers
decoder = Dense(30, activation='relu')(encoder)
decoder = Dense(input_dim, activation='sigmoid')(decoder) # Output layer with sigmoid for [0, 1] scaled data

# Create the Autoencoder model
autoencoder = Model(inputs=input_layer, outputs=decoder)

# Compile the Autoencoder
autoencoder.compile(optimizer='adam', loss='mse')

# Train the Autoencoder
history = autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, shuffle=True, validation_data=(X_test, X_test))

print("\n--- Autoencoder Training Complete ---")

# Create the encoder model to get the embeddings
encoder_model = Model(inputs=input_layer, outputs=encoder)

# Get embeddings for the test data
X_test_encoded = encoder_model.predict(X_test)

print(f"Original test data shape: {X_test.shape}")
print(f"Encoded test data shape (embeddings): {X_test_encoded.shape}")

# You can now use X_test_encoded (the embeddings) for downstream tasks
# For example, visualize embeddings if dimension is 2 or 3

# Optional: Plot reconstruction loss
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Autoencoder Reconstruction Loss')
plt.xlabel('Epoch')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.grid(True)
plt.show()

# Note: For visualization of embeddings, you would typically need to embed into 2 or 3 dimensions.
# If encoding_dim > 3, you might use t-SNE or UMAP on X_test_encoded for visualization. 
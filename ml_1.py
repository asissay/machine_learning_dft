import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib
import matplotlib.pyplot as plt

# Load QM9 dataset
file_path = 'qm9_small.xyz'
data = pd.read_csv(file_path, sep = '\s+', header=0)

# Define the features and target
features = ['rotA', 'rotB', 'rotC', 'mu', 'alpha', 'lumo', 'r2', 'U0', 'U', 'H', 'G', 'Cv', 'gap', 'zpve']
target = 'homo'

# Split the dataset into training and testing sets
X = data[features]
y = data[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the scaler for later use
scaler_filename = './scaler.joblib'
joblib.dump(scaler, scaler_filename)

# Define the neural network model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1)  # Output layer for regression (predicting total energy)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Create TensorFlow datasets for training and testing
batch_size = 32
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size)

# Train the model using the datasets
history = model.fit(train_dataset, epochs=10, validation_data=test_dataset, verbose=1)
#model.fit(train_dataset, epochs=1, validation_data=test_dataset, verbose=1)

# Evaluate the model
loss = model.evaluate(test_dataset, verbose=1)
print("Mean Squared Error on test set:", loss)

# Save the trained model to a file
model_filename = './trained_model.keras'
model.save(model_filename)

print(f"Model saved to {model_filename}")
print(f"Scaler saved to {scaler_filename}")

# Load the trained model
# model_path = '/home/adonay/Documents/ml/ml_1/teach/trained_model.keras'
# model = tf.keras.models.load_model(model_path)

# # Make predictions
# predictions = model.predict(X_test)

# # Print or save the predictions
# print("Predicted HOMO values:")
# print(predictions)

# # Optionally, attach predictions to the dataframe and save
# data['predicted_homo'] = predictions.flatten()

# data.to_csv('predicted_results.csv', index=False)


# Plot the loss curves
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss (e.g., MSE)')
plt.title('Training vs Validation Loss')
plt.legend()
plt.grid(True)
plt.show()

import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt  # Import matplotlib for plotting

# Load and prepare data
X = []
y = []

all_operations = set()

with open('operation_costs.jsonl', 'r') as f:
    for line in f:
        entry = json.loads(line)
        all_operations.update(entry['operation_counts'].keys())

all_operations = sorted(all_operations)

with open('operation_costs.jsonl', 'r') as f:
    for line in f:
        entry = json.loads(line)
        counts = entry['operation_counts']
        features = [counts.get(op, 0) for op in all_operations]
        X.append(features)
        y.append(entry['cost'])

X = np.array(X)
y = np.array(y)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model and record the history
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=16,
    validation_split=0.1,
    verbose=1
)

# Evaluate the model
loss = model.evaluate(X_test, y_test)
print(f"Test loss: {loss}")

# Save the model
model.save('cost_estimation_model.h5')

# Plot the training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.savefig('training_validation_loss.png')  # Save the plot as a PNG file
plt.show()

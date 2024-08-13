import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

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

model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(X_train, y_train, epochs=10, batch_size=16, validation_split=0.1)

loss = model.evaluate(X_test, y_test)
print(f"Test loss: {loss}")

model.save('cost_estimation_model.h5')

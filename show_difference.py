import os
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from utils import create_context_with_all_dialects, parse_mlir_file
from predict_cost import predict_cost

SCALING_FACTOR = 10

def count_operations(module):
    """Counts the operations in an MLIR module."""
    operation_counts = {}
    for op in module.walk():
        op_name = op.name
        if op_name not in operation_counts:
            operation_counts[op_name] = 0
        operation_counts[op_name] += 1
    return operation_counts

def collect_data(test_folder):
    """Collects real costs from MLIR files in the specified test folder."""
    real_costs = []
    file_names = []
    
    context = create_context_with_all_dialects()
    
    if not os.path.isdir(test_folder):
        print(f"The provided path '{test_folder}' is not a directory.")
        return real_costs, file_names

    files = os.listdir(test_folder)
    mlir_files = [file for file in files if file.endswith('.mlir') and os.path.isfile(os.path.join(test_folder, file))]
    
    for file in mlir_files:
        file_path = os.path.join(test_folder, file)
        module = parse_mlir_file(file_path, context)
        operation_counts = count_operations(module)
        
        cost = sum(operation_counts.values()) * SCALING_FACTOR
        real_costs.append(cost)
        file_names.append(file)

    return real_costs, file_names

def calculate_predicted_costs(file_paths, model, scaler, all_operations, context):
    """Calculates predicted costs for each MLIR file."""
    predicted_costs = []
    for file_path in file_paths:
        try:
            cost = predict_cost(file_path, model, scaler, all_operations, context)
            predicted_costs.append(cost)
        except Exception as e:
            print(f"Error predicting cost for {file_path}: {e}")
            predicted_costs.append(None)  # Handle failed predictions
    return predicted_costs

def plot_costs(real_costs, predicted_costs, file_names):
    """Plots a graph comparing real and predicted costs."""
    plt.figure(figsize=(10, 6))
    plt.plot(file_names, real_costs, label='Real Cost', marker='o')
    plt.plot(file_names, predicted_costs, label='Predicted Cost', marker='x')
    plt.xlabel('MLIR Files')
    plt.ylabel('Cost')
    plt.title('Comparison of Real and Predicted Costs')
    plt.legend()
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

def main():
    # Load the pre-trained model
    model = tf.keras.models.load_model('cost_estimation_model.h5')

    # Load operation costs and extract unique operations
    with open('operation_costs.jsonl', 'r') as f:
        data = [json.loads(line) for line in f]
    all_operations = sorted({op for entry in data for op in entry['operation_counts'].keys()})

    # Initialize the scaler
    scaler = StandardScaler()
    scaler.fit_transform([np.zeros(len(all_operations))])

    # Prepare MLIR files and context
    test_directory = 'testing_files'
    context = create_context_with_all_dialects()

    # Collect real costs
    real_costs, file_names = collect_data(test_directory)

    # Predict costs
    file_paths = [os.path.join(test_directory, file) for file in file_names]
    predicted_costs = calculate_predicted_costs(file_paths, model, scaler, all_operations, context)

    # Plot the costs
    plot_costs(real_costs, predicted_costs, file_names)

if __name__ == "__main__":
    main()

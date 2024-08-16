import tensorflow as tf
import numpy as np
import json
from utils import create_context_with_all_dialects, parse_mlir_file
from sklearn.preprocessing import StandardScaler

def count_operations(module):
    operation_counts = {}
    
    for op in module.walk():
        op_name = op.name
        if op_name not in operation_counts:
            operation_counts[op_name] = 0
        operation_counts[op_name] += 1
    
    return operation_counts

def predict_cost(file_path, model, scaler, all_operations, context):
    module = parse_mlir_file(file_path, context)
    operation_counts = count_operations(module)
    features = [operation_counts.get(op, 0) for op in all_operations]
    
    features = scaler.transform([features])
    
    predicted_cost = model.predict(features)
    return predicted_cost[0][0]

def main():
    model = tf.keras.models.load_model('cost_estimation_model.h5')

    with open('operation_costs.jsonl', 'r') as f:
        data = [json.loads(line) for line in f]
    all_operations = sorted({op for entry in data for op in entry['operation_counts'].keys()})

    scaler = StandardScaler()
    scaler.fit_transform([np.zeros(len(all_operations))])

    context = create_context_with_all_dialects()

    file_path = r'C:\Users\ilyas\dev\xdsl\tests\filecheck\backend\convert_riscv_scf_to_riscv_cf.mlir'
    predicted_cost = predict_cost(file_path, model, scaler, all_operations, context)
    print(f"Predicted cost for {file_path}: {predicted_cost}")

if __name__ == "__main__":
    main()
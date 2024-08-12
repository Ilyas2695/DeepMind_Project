import tensorflow as tf
import numpy as np
import json
from xdsl.context import MLContext
from xdsl.dialects import get_all_dialects
from xdsl.parser import Parser
from sklearn.preprocessing import StandardScaler

def count_operations(module):
    operation_counts = {}
    
    def visit_operation(operation):
        op_name = operation.name
        if op_name not in operation_counts:
            operation_counts[op_name] = 0
        operation_counts[op_name] += 1
        
        for region in operation.regions:
            for block in region.blocks:
                for op in block.ops:
                    visit_operation(op)
    
    for op in module.ops:
        visit_operation(op)
    
    return operation_counts

def parse_mlir_file(file_path):
    with open(file_path, 'r') as f:
        content = f.read()

    context = MLContext()
    for dialect_name, dialect_factory in get_all_dialects().items():
        context.register_dialect(dialect_name, dialect_factory)
    
    parser = Parser(context, content)
    module = parser.parse_module()
    return module

def predict_cost(file_path, model, scaler, all_operations):
    module = parse_mlir_file(file_path)
    operation_counts = count_operations(module)
    features = [operation_counts.get(op, 0) for op in all_operations]
    
    features = scaler.transform([features])
    
    predicted_cost = model.predict(features)
    return predicted_cost[0][0]

if __name__ == "__main__":
    model = tf.keras.models.load_model('cost_estimation_model.h5')

    with open('operation_costs.json', 'r') as f:
        data = json.load(f)
    all_operations = sorted({op for entry in data for op in entry['operation_counts'].keys()})

    scaler = StandardScaler()
    scaler.fit_transform([np.zeros(len(all_operations))])

    file_path = r'C:\Users\ilyas\dev\xdsl\tests\filecheck\backend\riscv\convert_arith_to_riscv.mlir'
    predicted_cost = predict_cost(file_path, model, scaler, all_operations)
    print(f"Predicted cost for {file_path}: {predicted_cost}")

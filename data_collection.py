import os
import json
from xdsl.context import MLContext
from xdsl.dialects import get_all_dialects
from xdsl.parser import Parser

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

def collect_data(test_folder, output_file):
    data = []
    
    if not os.path.isdir(test_folder):
        print(f"The provided path '{test_folder}' is not a directory.")
        return

    files = os.listdir(test_folder)
    mlir_files = [file for file in files if file.endswith('.mlir') and os.path.isfile(os.path.join(test_folder, file))]
    
    if mlir_files:
        for file in mlir_files:
            file_path = os.path.join(test_folder, file)
            try:
                module = parse_mlir_file(file_path)
                operation_counts = count_operations(module)
                
                cost = sum(operation_counts.values()) * 10
                
                data.append({
                    "file": file,
                    "operation_counts": operation_counts,
                    "cost": cost
                })
            except Exception as e:
                print(f"Error parsing file {file}: {e}")
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4)

if __name__ == "__main__":
    collect_data(r'C:\Users\ilyas\dev\xdsl\tests\filecheck\backend', 'operation_costs.json')

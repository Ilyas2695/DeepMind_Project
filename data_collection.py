import os
import json
import re
from xdsl.context import MLContext
from xdsl.dialects import get_all_dialects
from xdsl.parser import Parser

def remove_comments(content):
    content = re.sub(r'//.*', '', content)
    content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
    return content

def count_operations(module):
    operation_counts = {}
    
    for op in module.walk():
        op_name = op.name
        if op_name not in operation_counts:
            operation_counts[op_name] = 0
        operation_counts[op_name] += 1
    
    return operation_counts

def parse_mlir_file(file_path, context):
    with open(file_path, 'r') as f:
        content = f.read()

    content = remove_comments(content)

    parser = Parser(context, content)
    module = parser.parse_module()
    return module

def collect_data(test_folder, output_file):
    context = MLContext()
    for dialect_name, dialect_factory in get_all_dialects().items():
        context.register_dialect(dialect_name, dialect_factory)
    
    if not os.path.isdir(test_folder):
        print(f"The provided path '{test_folder}' is not a directory.")
        return

    files = os.listdir(test_folder)
    mlir_files = [file for file in files if file.endswith('.mlir') and os.path.isfile(os.path.join(test_folder, file))]
    
    if mlir_files:
        with open(output_file, 'w') as f:
            for file in mlir_files:
                file_path = os.path.join(test_folder, file)
                try:
                    module = parse_mlir_file(file_path, context)
                    operation_counts = count_operations(module)
                    
                    cost = sum(operation_counts.values()) * 10
                    
                    record = {
                        "file": file,
                        "operation_counts": operation_counts,
                        "cost": cost
                    }
                    f.write(json.dumps(record) + "\n")
                except Exception as e:
                    print(f"Error parsing file {file}: {e}")

if __name__ == "__main__":
    collect_data(r'C:\Users\ilyas\dev\DeepMind_Project\training_files', 'operation_costs.jsonl')

import os
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

def main():
    test_folder = r'C:\Users\ilyas\dev\DeepMind_Project\testing_files'
    
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
                
                print(f"Operation counts in file {file}:")
                for op_name, count in operation_counts.items():
                    print(f"  {op_name}: {count}")
            except Exception as e:
                print(f"Error parsing file {file}: {e}")
    else:
        print("No .mlir files found in the test folder.")

if __name__ == "__main__":
    main()

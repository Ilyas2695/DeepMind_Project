import os
import re
import matplotlib.pyplot as plt
from collections import Counter
from xdsl.context import MLContext
from xdsl.dialects import get_all_dialects
from xdsl.parser import Parser

def remove_comments(content):
    content = re.sub(r'//.*', '', content)
    content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
    return content

def count_operations(module):
    operation_counts = Counter()
    
    for op in module.walk():
        operation_counts[op.name] += 1
    
    return operation_counts

def parse_mlir_file(file_path, context):
    with open(file_path, 'r') as f:
        content = f.read()

    content = remove_comments(content)

    parser = Parser(context, content)
    module = parser.parse_module()
    return module

def collect_operation_counts(test_folder):
    context = MLContext()
    for dialect_name, dialect_factory in get_all_dialects().items():
        context.register_dialect(dialect_name, dialect_factory)
    
    if not os.path.isdir(test_folder):
        print(f"The provided path '{test_folder}' is not a directory.")
        return

    operation_counts = Counter()

    files = os.listdir(test_folder)
    mlir_files = [file for file in files if file.endswith('.mlir') and os.path.isfile(os.path.join(test_folder, file))]
    
    if mlir_files:
        for file in mlir_files:
            file_path = os.path.join(test_folder, file)
            try:
                module = parse_mlir_file(file_path, context)
                file_operation_counts = count_operations(module)
                operation_counts.update(file_operation_counts)
            except Exception as e:
                print(f"Error parsing file {file}: {e}")
    
    return operation_counts

def plot_histogram(operation_counts, top_n=10):
    common_operations = operation_counts.most_common(top_n)
    operations, counts = zip(*common_operations)

    plt.figure(figsize=(10, 6))
    plt.bar(operations, counts, color='blue')
    plt.xlabel('Operations')
    plt.ylabel('Count')
    plt.title(f'Top {top_n} Most Common Operations in .mlir Files')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    folder_path = r'C:\Users\ilyas\dev\DeepMind_Project\training_files'
    operation_counts = collect_operation_counts(folder_path)
    plot_histogram(operation_counts, top_n=10)

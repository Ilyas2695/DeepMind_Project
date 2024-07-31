import re
import os

def remove_comments(code):
    code = re.sub(r'//.*', '', code)
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    return code

def extract_operations(file_path, operations):
    with open(file_path, 'r') as file:
        content = file.read()

    content = remove_comments(content)

    operation_counts = {op: 0 for op in operations}

    for op in operations:
        operation_counts[op] = content.count(op)
    
    return operation_counts

def main():
    test_folder = r'C:\Users\ilyas\dev\xdsl\tests\filecheck\backend'
    if not os.path.isdir(test_folder):
        print(f"The provided path '{test_folder}' is not a directory.")
        return

    operations = [
        'add', 'sub', 'mul', 'div', 'rem', 'neg', 'and', 'or', 'xor', 'not', 'cmp', 'eq', 'ne', 'lt', 'le', 'gt', 'ge', 
        'index_cast', 'sext', 'zext', 'trunc', 'alloc', 'dealloc', 'load', 'store',
        'affine_map', 'affine.for', 'affine.parallel', 'affine.if', 'affine.else', 'affine.load', 'affine.store', 
        'affine.min', 'affine.max', 'scf.for', 'scf.parallel', 'scf.while', 'scf.if', 'scf.else', 'scf.yield', 
        'llvm.add', 'llvm.sub', 'llvm.mul', 'llvm.and', 'llvm.or', 'llvm.xor', 'llvm.alloca', 'llvm.load', 'llvm.store', 
        'llvm.br', 'llvm.cond_br', 'llvm.switch', 'llvm.ret', 'llvm.intrinsic', 'tensor.extract', 'tensor.insert', 
        'tensor.cast', 'linalg.matmul', 'linalg.batch_matmul', 'linalg.vecmatmul', 'linalg.conv', 'linalg.depthwise_conv', 
        'linalg.reduce'
    ]

    files = os.listdir(test_folder)
    mlir_files = [file for file in files if file.endswith('.mlir') and os.path.isfile(os.path.join(test_folder, file))]
    
    if mlir_files:
        for file in mlir_files:
            file_path = os.path.join(test_folder, file)
            operation_counts = extract_operations(file_path, operations)
            if any(operation_counts.values()):
                print(f"Operation counts in file {file}:")
                for op, count in operation_counts.items():
                    if count > 0:
                        print(f"{op}: {count}")
            else:
                print(f"No operations found in file {file}.")
    else:
        print("No .mlir files found in the test folder.")

if __name__ == "__main__":
    main()

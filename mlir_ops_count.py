import re
import os
import jax.numpy as jnp
from jax import lax

def remove_comments(code):
    code = re.sub(r'//.*', '', code)
    code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
    return code

def extract_operations(file_path, operations):
    with open(file_path, 'r') as file:
        content = file.read()

    content = remove_comments(content)

    operation_counts = jnp.zeros(len(operations), dtype=jnp.int32)

    for i, op in enumerate(operations):
        pattern = re.compile(r'\b' + re.escape(op) + r'\b')
        count = len(pattern.findall(content))
        operation_counts = lax.dynamic_update_index_in_dim(operation_counts, count, i, 0)
    
    return operation_counts

def main():
    test_folder = r'C:\Users\ilyas\dev\xdsl\tests\filecheck\backend'
    if not os.path.isdir(test_folder):
        print(f"The provided path '{test_folder}' is not a directory.")
        return

    operations = [
        # Standard Dialect
        'add', 'sub', 'mul', 'div', 'rem', 'neg', 'and', 'or', 'xor', 'not', 'cmp', 'eq', 'ne', 'lt', 'le', 'gt', 'ge', 
        'index_cast', 'sext', 'zext', 'trunc', 'alloc', 'dealloc', 'load', 'store',
        # Affine Dialect
        'affine_map', 'affine.for', 'affine.parallel', 'affine.if', 'affine.else', 'affine.load', 'affine.store', 
        'affine.min', 'affine.max', 
        # SCF Dialect
        'scf.for', 'scf.parallel', 'scf.while', 'scf.if', 'scf.else', 'scf.yield', 
        # LLVM Dialect
        'llvm.add', 'llvm.sub', 'llvm.mul', 'llvm.and', 'llvm.or', 'llvm.xor', 'llvm.alloca', 'llvm.load', 'llvm.store', 
        'llvm.br', 'llvm.cond_br', 'llvm.switch', 'llvm.ret', 'llvm.intrinsic', 
        # Tensor Dialect
        'tensor.extract', 'tensor.insert', 'tensor.cast', 
        # Linalg Dialect
        'linalg.matmul', 'linalg.batch_matmul', 'linalg.vecmatmul', 'linalg.conv', 'linalg.depthwise_conv', 'linalg.reduce',
        # Additional operations
        'arith.constant', 'arith.addi', 'builtin.module', 'builtin.unrealized_conversion_cast', 'riscv.mv', 'riscv.fmv.s', 
        'riscv.fmv.d', 'riscv_scf.for', 'riscv_scf.yield'
    ]

    files = os.listdir(test_folder)
    mlir_files = [file for file in files if file.endswith('.mlir') and os.path.isfile(os.path.join(test_folder, file))]
    
    if mlir_files:
        for file in mlir_files:
            file_path = os.path.join(test_folder, file)
            operation_counts = extract_operations(file_path, operations)
            if jnp.any(operation_counts > 0):
                print(f"Operation counts in file {file}:")
                for i, count in enumerate(operation_counts):
                    if count > 0:
                        print(f"  {operations[i]}: {count}")
            else:
                print(f"No operations found in file {file}.")
    else:
        print("No .mlir files found in the test folder.")

if __name__ == "__main__":
    main()

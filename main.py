import os
import jax
import jax.numpy as jnp
from jax import lax

def count_operations(file_path):
    with open(file_path, 'r') as file:
        code = file.read()

    operations = ['+', '-', '*', '/', '//', '%', '**', '>>', '<<', '&', '|', '^', '~']
    operation_counts = jnp.zeros(len(operations), dtype=jnp.int32)

    for i, op in enumerate(operations):
        count = code.count(op)
        operation_counts = lax.dynamic_update_index_in_dim(operation_counts, count, i, 0)
    
    return operation_counts

def main():
    test_folder = r'C:\Users\ilyas\dev\xdsl\tests\backend\riscv'
    files = os.listdir(test_folder)
    if files:
        file_path = os.path.join(test_folder, files[0])
        operation_counts = count_operations(file_path)
        print(f"Operation counts in file {files[0]}: {operation_counts}")
    else:
        print("No files found in the test folder.")

if __name__ == "__main__":
    main()

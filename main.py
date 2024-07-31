import os
import jax
import jax.numpy as jnp
from jax import lax

def count_operations(file_path):
    # Read the file content
    with open(file_path, 'r') as file:
        code = file.read()

    # Define operation keywords to look for (assuming basic Python operations)
    operations = ['+', '-', '*', '/', '//', '%', '**', '>>', '<<', '&', '|', '^', '~']
    operation_counts = jnp.zeros(len(operations), dtype=jnp.int32)

    # Count occurrences of each operation
    for i, op in enumerate(operations):
        count = code.count(op)
        operation_counts = lax.dynamic_update_index_in_dim(operation_counts, count, i, 0)
    
    return operation_counts

def main():
    test_folder = r'C:\Users\ilyas\dev\xdsl\tests\backend\riscv'
    # List files in the test folder
    files = os.listdir(test_folder)
    # Assuming you want to count operations in the first file
    if files:
        file_path = os.path.join(test_folder, files[0])
        operation_counts = count_operations(file_path)
        print(f"Operation counts in file {files[0]}: {operation_counts}")
    else:
        print("No files found in the test folder.")

if __name__ == "__main__":
    main()

import os
import jax
import jax.numpy as jnp
from jax import lax

def count_operations(file_path):
    try:
        with open(file_path, 'r') as file:
            code = file.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None, None

    operations = ['+', '-', '*', '/', '//', '%', '**', '>>', '<<', '&', '|', '^', '~']
    operation_counts = jnp.zeros(len(operations), dtype=jnp.int32)

    for i, op in enumerate(operations):
        count = code.count(op)
        operation_counts = lax.dynamic_update_index_in_dim(operation_counts, count, i, 0)
    
    return operations, operation_counts

def main():
    test_folder = r'C:\Users\ilyas\dev\xdsl\tests\backend\riscv'
    files = os.listdir(test_folder)
    if files:
        for file in files:
            file_path = os.path.join(test_folder, file)
            operations, operation_counts = count_operations(file_path)
            if operations is not None and operation_counts is not None:
                if jnp.all(operation_counts == 0):
                    print(f"No operations found in file {file}.")
                else:
                    print(f"Operation counts in file {file}:")
                    for i in range(len(operation_counts)):
                        if operation_counts[i] != 0:
                            print(f"{operations[i]}: {operation_counts[i]}")
    else:
        print("No files found in the test folder.")

if __name__ == "__main__":
    main()

import os

def check_permissions(file_path):
    permissions = {
        'readable': os.access(file_path, os.R_OK),
        'writable': os.access(file_path, os.W_OK),
        'executable': os.access(file_path, os.X_OK),
    }
    return permissions

def main():
    test_folder = r'C:\Users\ilyas\dev\xdsl\tests\backend\riscv'
    # List files in the test folder
    files = os.listdir(test_folder)
    # Filter out directories from files list
    files = [f for f in files if os.path.isfile(os.path.join(test_folder, f))]
    # Assuming you want to check permissions for the first file
    if files:
        file_path = os.path.join(test_folder, files[0])
        permissions = check_permissions(file_path)
        print(f"Permissions for file {files[0]}: {permissions}")
    else:
        print("No files found in the test folder.")

if __name__ == "__main__":
    main()

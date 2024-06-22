import os

# Concatenate all Python files in a directory and its subdirectories into a single file
# To use this script: python -m factchecker.tools.python_script_concatenator

def concatenate_python_files(source_dir, output_file):
    '''
    Concatenate all Python files in a directory and its subdirectories into a single file.
    This might be useful for AI based code review.
    '''
    with open(output_file, 'w') as outfile:
        for root, dirs, files in os.walk(source_dir):
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.join(root, file)
                    with open(file_path, 'r') as infile:
                        outfile.write(f"# Start of {file_path}\n")
                        outfile.write(infile.read())
                        outfile.write(f"\n# End of {file_path}\n\n")

def main():

    source_directory = "factchecker"  # Set the path to source directory
    output_file = "concatenated_python_code.py" # Define the output file

    concatenate_python_files(source_directory, output_file)
    print(f"All Python files from {source_directory} have been concatenated into {output_file}.")

if __name__ == "__main__":
    main()
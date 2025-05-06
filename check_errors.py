# check_errors.py
# Simple script to run main.py and show any errors

import subprocess
import sys

def run_with_task(task):
    print(f"\nTesting with task: {task}")
    cmd = [
        "python", "main.py",
        "--dataset", "glue",
        "--task", task,
        "--method", "non-private",
        "--epsilon", "8.0",
        "--num_epochs", "1"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    # Run the command and capture output
    process = subprocess.run(cmd, capture_output=True, text=True)
    
    # Print return code
    print(f"Return code: {process.returncode}")
    
    # Print stdout if any
    if process.stdout:
        print("\nStandard output:")
        print(process.stdout)
    else:
        print("\nNo standard output")
    
    # Print stderr if any
    if process.stderr:
        print("\nError output:")
        print(process.stderr)
    else:
        print("\nNo error output")

# Run tests with different tasks
tasks = ["sst2", "qqp", "mnli", "cola"]
for task in tasks:
    run_with_task(task)

print("\nDone testing main.py")

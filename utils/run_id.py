import os

def get_next_run_id(file_path="run_id.txt"):
    # If the file doesn't exist, create it with 0
    if not os.path.exists(file_path):
        with open(file_path, "w") as f:
            f.write("0")

    # Read the current run ID
    with open(file_path, "r") as f:
        run_id = int(f.read().strip())

    # Increment and write back
    new_run_id = run_id + 1
    with open(file_path, "w") as f:
        f.write(str(new_run_id))

    return new_run_id
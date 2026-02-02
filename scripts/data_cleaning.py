import os
import shutil

# --- CONFIGURATION ---
# The text file containing the folder names (one per line)
list_file = "missing_videos.txt"

# The directory where these folders are located
# Use '.' for the current directory where this script is running
base_directory = './data/cadp/extracted_frames'
# ---------------------

def delete_folders_from_list():
    # Check if the list file exists first
    if not os.path.exists(list_file):
        print(f"Error: The file '{list_file}' was not found.")
        return

    print(f"Reading from: {list_file}")
    print(f"Target directory: {os.path.abspath(base_directory)}\n")

    with open(list_file, 'r') as f:
        # Read lines and strip whitespace/newlines
        folders = [line.strip() for line in f.readlines()]

    for folder_name in folders:
        # Skip empty lines in the text file
        if not folder_name:
            continue

        full_path = os.path.join(base_directory, folder_name)

        if os.path.exists(full_path):
            try:
                # shutil.rmtree removes the directory and all its contents
                shutil.rmtree(full_path)
                print(f"[DELETED] {folder_name}")
            except Exception as e:
                print(f"[ERROR] Could not delete {folder_name}: {e}")
        else:
            print(f"[SKIPPED] {folder_name} not found.")

if __name__ == "__main__":
    delete_folders_from_list()
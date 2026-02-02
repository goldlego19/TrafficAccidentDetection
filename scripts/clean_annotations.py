import csv
import os

# --- CONFIGURATION ---
input_csv = 'annotations/accidents.csv'          # Your current dirty CSV
output_csv = 'annotations/accidents_cleaned.csv' # The new clean CSV to generate
video_directory = 'data/cadp/extracted_frames'  # FOLDER WHERE YOUR VIDEO FOLDERS LIVE
id_column = 'video_id'                           # Header name in CSV
# ---------------------

def clean_csv():
    # 1. Get a list of the actual folders you have on disk
    if not os.path.exists(video_directory):
        print(f"Error: The directory '{video_directory}' does not exist.")
        return
        
    print(f"Scanning '{video_directory}' for folders...")
    # This creates a set of all folder names in your data directory
    actual_folders = set(os.listdir(video_directory))
    
    print(f"Found {len(actual_folders)} items in data directory.")

    # 2. Read the CSV and filter rows
    kept_rows = []
    removed_count = 0

    try:
        with open(input_csv, 'r', newline='', encoding='utf-8') as f_in:
            reader = csv.DictReader(f_in)
            
            # Verify header
            if id_column not in reader.fieldnames:
                print(f"Error: Column '{id_column}' not found in CSV.")
                return

            print(f"Filtering '{input_csv}'...")
            
            for row in reader:
                vid_id = row[id_column].strip()
                
                # CHECK: Does this ID exist in the actual folders?
                if vid_id in actual_folders:
                    kept_rows.append(row)
                else:
                    removed_count += 1
                    # Optional: Print what is being removed
                    # print(f"Removing annotation for {vid_id} (Folder not found)")

    except FileNotFoundError:
        print(f"Error: Could not find '{input_csv}'")
        return

    # 3. Write the new Clean CSV
    if kept_rows:
        with open(output_csv, 'w', newline='', encoding='utf-8') as f_out:
            writer = csv.DictWriter(f_out, fieldnames=reader.fieldnames)
            writer.writeheader()
            writer.writerows(kept_rows)
            
        print("-" * 30)
        print(f"Original Annotations: {len(kept_rows) + removed_count}")
        print(f"Rows Removed:         {removed_count}")
        print(f"Rows Kept:            {len(kept_rows)}")
        print(f"New CSV saved to:     {output_csv}")
        print("-" * 30)
    else:
        print("Error: No rows were kept! Check if your video_directory path is correct.")

if __name__ == "__main__":
    clean_csv()
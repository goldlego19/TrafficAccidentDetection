import csv
import os

# --- CONFIGURATION ---
input_csv = 'annotations/accidents.csv'         # Ensure your CSV file is named this
output_file = 'missing_videos.txt'  # The file to be created
column_name = 'video_id'            # Matches your CSV header exactly
start_id = 0
end_id = 1415                       # Will check up to 001415
# ---------------------

def find_missing():
    # 1. Generate the full set of expected IDs (000000 to 001415)
    expected_ids = set(f"{i:06d}" for i in range(start_id, end_id + 1))
    
    existing_ids = set()

    if not os.path.exists(input_csv):
        print(f"Error: Could not find file '{input_csv}'. Make sure it is in the same folder.")
        return

    print(f"Reading '{input_csv}'...")

    try:
        with open(input_csv, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            # Check if headers match
            if column_name not in reader.fieldnames:
                print(f"Error: Column '{column_name}' not found. Found: {reader.fieldnames}")
                return

            for row in reader:
                val = row[column_name].strip()
                if val.isdigit():
                    # Ensure we handle IDs like '5' as '000005' to match the list
                    existing_ids.add(f"{int(val):06d}")

    except Exception as e:
        print(f"An error occurred: {e}")
        return

    # 2. Calculate Missing (Expected - Existing)
    missing = sorted(list(expected_ids - existing_ids))

    # 3. Write to file
    print(f"Analyzed {len(existing_ids)} videos.")
    print(f"Found {len(missing)} missing videos.")
    
    with open(output_file, 'w') as f:
        for video_id in missing:
            f.write(video_id + '\n')

    print(f"Success! List written to '{output_file}'")
    
    # Optional: Print first few missing to console for verification
    if missing:
        print(f"\nPreview (First 5 missing): {missing[:5]}")

if __name__ == "__main__":
    find_missing()
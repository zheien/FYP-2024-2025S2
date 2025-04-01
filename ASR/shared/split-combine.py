import json
import os
import glob
import math

# Set the path where your JSON files are located
json_folder = "../Final/#5Earnings/output"  # Corrected path
output_file = "combined.json"  # Output file name

json_files = glob.glob(os.path.join(json_folder, "*.json"))

combined_data = {}

# Load existing data if the output file already exists
if os.path.exists(output_file):
    with open(output_file, 'r') as f:
        try:
            existing_data = json.load(f)
            if isinstance(existing_data, dict):
                combined_data = existing_data
        except json.JSONDecodeError:
            combined_data = {}

# Iterate over each JSON file and merge the contents
for file in json_files:
    with open(file, 'r') as f:
        try:
            data = json.load(f)
            if isinstance(data, dict):
                for key, value in data.items():
                    if key in combined_data and isinstance(combined_data[key], list) and isinstance(value, list):
                        combined_data[key].extend(value)
                    else:
                        combined_data[key] = value
        except json.JSONDecodeError:
            print(f"Skipping invalid JSON file: {file}")

# Save the combined JSON
with open(output_file, 'w') as f:
    json.dump(combined_data, f, indent=4)

print(f"Successfully combined {len(json_files)} files into {output_file}")
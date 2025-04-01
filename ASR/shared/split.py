import json
import os
import math

# Set the input JSON file path
# input_file = "../#1GTN/gtn.json"
input_file = "../Final/#5Earnings/output/combined.json"  #
# Create output directories
os.makedirs("train", exist_ok=True)
os.makedirs("test", exist_ok=True)
os.makedirs("validation", exist_ok=True)

# Read the input JSON file
try:
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    if not isinstance(data, list):
        print("Error: Expected JSON file to contain a list of items")
        exit()
        
    total_items = len(data)
    print(f"\nTotal items in JSON: {total_items}")
    
    # Calculate split sizes
    test_size = math.floor(total_items * 0.05)  # 5% for test
    val_size = math.floor(total_items * 0.05)   # 5% for validation
    train_size = total_items - (test_size + val_size)  # rest for training
    
    # Split the data
    train_data = data[:train_size]
    test_data = data[train_size:train_size + test_size]
    val_data = data[train_size + test_size:]
    
    # Save splits to separate files
    with open("train/gtn_train.json", 'w', encoding='utf-8') as f:
        json.dump(train_data, f, indent=4)
        
    with open("test/gtn_test.json", 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=4)
        
    with open("validation/gtn_val.json", 'w', encoding='utf-8') as f:
        json.dump(val_data, f, indent=4)
    
    # Print split information
    print("\nSplit Results:")
    print(f"Train: {len(train_data)} items ({len(train_data)/total_items*100:.1f}%)")
    print(f"Test: {len(test_data)} items ({len(test_data)/total_items*100:.1f}%)")
    print(f"Validation: {len(val_data)} items ({len(val_data)/total_items*100:.1f}%)")
    
except FileNotFoundError:
    print(f"Error: Could not find file {input_file}")
    print("Make sure you're running the script from the correct directory")
except json.JSONDecodeError:
    print("Error: Invalid JSON file")
except Exception as e:
    print(f"An error occurred: {str(e)}")
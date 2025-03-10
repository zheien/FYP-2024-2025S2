import json
import random
import math

def combine_segments(input_file, output_file, multiplier=None):

    # Load the JSON file
    with open(input_file, 'r', encoding='utf-8') as infile:
        data = json.load(infile)

    for item in data:
        segments = item.get('segments', [])
        combined_segments = []

        # Skip if segments are empty
        if not segments:
            print(f"Skipping an item with no segments: {item}")
            continue

        # Default multiplier to 1 if not specified
        if multiplier is None:
            multiplier = 1

        # Calculate the number of sentences based on the multiplier, rounded up
        # 9 sentence * 0.5 multiplier = 4.5 (rounded up to 5 sentence)
        num_sentences = max(1, math.ceil(len(segments) * multiplier))

        # Randomly combine segments where segment_end == segment_start
        valid_combinations = [
            (seg1, seg2)
            for seg1 in segments
            for seg2 in segments
            if seg1["segment_end"] == seg2["segment_start"]
        ]

        # If no valid combinations are found, skip the item
        if not valid_combinations:
            print(f"No valid combinations found for item: {item}")
            continue

        # Shuffle and pick up to `num_sentences` combinations
        random.shuffle(valid_combinations)
        for seg1, seg2 in valid_combinations[:num_sentences]:
            combined_segment = {
                "segment_start": seg1["segment_start"],
                "segment_end": seg2["segment_end"],
                "normalized": f"{seg1['normalized']} {seg2['normalized']}",
                "unnormalized": f"{seg1['unnormalized']} {seg2['unnormalized']}"
            }
            combined_segments.append(combined_segment)

        # Append combined segments back to the original list
        item['segments'].extend(combined_segments)

    # Save the updated data to the output file
    with open(output_file, "w", encoding='utf-8') as file:
        json.dump(data, file, indent=2)

    print(f"Processed data saved to '{output_file}'.")


if __name__ == "__main__":
    input_file = 'ami.json'
    output_file = 'ami-1.0.json'

    # Prompt the user for the multiplier
    try:
        user_input = input("Enter a multiplier for the number of sentences (e.g., 0.5 for half, 1 for all): ").strip()
        multiplier = float(user_input) if user_input else None
    except ValueError:
        print("Invalid input. Using default multiplier of 1.")
        multiplier = None

    # Call the function with user input
    combine_segments(input_file, output_file, multiplier)

import json
import random
import math

def combine_segments(input_file, output_file, multiplier=1):
    with open(input_file, 'r', encoding='utf-8') as infile:
        data = json.load(infile)
    
    for item in data:
        segments = item.get('segments', [])
        if not segments:
            continue
        
        original_segments = segments.copy()
        combined_segments = []  # To store original and newly created segments
        # combined_segments.extend(original_segments)  # Keep original segments intact
        
        new_segments = []  # To store the newly created segments
        
        # Track used segments to avoid duplicates
        used_segments = set()
        
        # Calculate the number of new segments to generate based on the multiplier
        new_segments_count = max(0, math.ceil(multiplier * len(original_segments)))  # Calculate new segments count
        
        # Generate the specified number of new segments
        for _ in range(new_segments_count):
            # Pick random segments and combine them
            for seg in original_segments:
                if seg["segment_start"] in used_segments:
                    continue  # Skip already used segments
                
                new_segment = seg.copy()
                used_segments.add(seg["segment_start"])  # Mark this segment as used
                
                # Try to combine this segment with the next available one
                while True:
                    next_seg = next((s for s in original_segments
                                     if s["segment_start"] == new_segment["segment_end"] and
                                     s["segment_start"] not in used_segments), None)
                    
                    if not next_seg or random.random() > 0.5:
                        break  # Stop combining if no valid next segment or random chance fails
                    
                    # Merge the segments
                    new_segment = {
                        "segment_start": new_segment["segment_start"],
                        "segment_end": next_seg["segment_end"],
                        "normalized": f"{new_segment['normalized']} {next_seg['normalized']}",
                        "unnormalized": f"{new_segment['unnormalized']} {next_seg['unnormalized']}"
                    }
                    used_segments.add(next_seg["segment_start"])  # Mark the next segment as used
                
                # Add the newly created segment to the list
                new_segments.append(new_segment)
                
                if len(new_segments) >= new_segments_count:
                    break  # Stop when we have generated enough new segments
        
            if len(new_segments) >= new_segments_count:
                break  # Ensure we stop once we hit the required number of new segments
        
        # Add the newly created segments to the final list
        combined_segments.extend(new_segments)
        
        # Update the item with the combined segments (original + new)
        item['segments'] = combined_segments
    
    # Save the updated data to the output file
    with open(output_file, "w", encoding='utf-8') as file:
        json.dump(data, file, indent=2)
    
    print(f"Processed data saved to '{output_file}'.")

if __name__ == "__main__":
    input_file = 'ami.json'
    output_file = 'ami-output2.165-5.json'
    
    try:
        user_input = input("Enter a multiplier for additional sentences: ").strip()
        multiplier = float(user_input) if user_input else 1
    except ValueError:
        print("Invalid input. Using default multiplier of 1.")
        multiplier = 1
    
    combine_segments(input_file, output_file, multiplier)

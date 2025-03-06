import json
import random
import string
from typing import List, Dict, Any, Optional, Tuple
import traceback
import re

def remove_punctuation(text: str) -> str:
    """Removes punctuation only if it has a space before or after it."""
    # Remove punctuation that is surrounded by spaces
    text = re.sub(r'\s([,.!?;:])\s', ' ', text)
    text = re.sub(r'\s([,.!?;:])$', ' ', text)  # Remove trailing punctuation with space
    return text.strip()

def load_json(file_path: str) -> Optional[List[Dict[str, Any]]]:
    """Loads a JSON file."""
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return None

def group_values(normalized_dict: Dict[str, float]) -> Tuple[List[str], List[str], List[str]]:
    """Groups the normalized values into three categories based on proximity."""
    try:
        # Sort the normalized values in descending order
        sorted_normalized = sorted(normalized_dict.items(), key=lambda x: x[1], reverse=True)
        
        values_sorted = [item[1] for item in sorted_normalized]
        words_sorted = [item[0] for item in sorted_normalized]
        
        if not values_sorted:
            return [], [], []
        
        # Case: All values are the same → Put all in most_likely
        if len(set(values_sorted)) == 1:
            return [], [], words_sorted

        # Special case: If there are exactly three values, assign them directly
        if len(values_sorted) == 3:
            return [words_sorted[2]], [words_sorted[1]], [words_sorted[0]]

        # Initialize the groups
        most_likely = [words_sorted[0]]  # Highest probability alone
        second_most_likely = []
        others = []

        # Define threshold for grouping
        threshold1 = 0.1  # Difference that separates most_likely from second_most_likely
        threshold2 = 0.01  # Difference that groups small values together

        for i in range(1, len(values_sorted)):
            if values_sorted[i - 1] - values_sorted[i] > threshold1:
                second_most_likely.append(words_sorted[i])
            elif values_sorted[i - 1] - values_sorted[i] > threshold2:
                others.append(words_sorted[i])
            else:
                others.append(words_sorted[i])

        # Ensure second_most_likely is not empty if others is filled
        if not second_most_likely and others:
            second_most_likely.append(others.pop(0))

        return others, second_most_likely, most_likely

    except Exception as e:
        print(f"Error in group_values: {str(e)}\n{traceback.format_exc()}")
        return [], [], []

def select_normalized(normalized_dict: Dict[str, float]) -> Optional[str]:
    """Selects a normalized word based on weighted probabilities."""
    try:
        if not isinstance(normalized_dict, dict):
            raise ValueError(f"Expected a dictionary for normalized values, but got {type(normalized_dict).__name__}")

        others, second_most_likely, most_likely = group_values(normalized_dict)

        # print("most_likely:", most_likely)
        # print("second_most_likely:", second_most_likely)
        # print("others:", others)

        if not most_likely and not second_most_likely and not others:
            raise ValueError("No valid normalized forms found.")

        rand = random.random()

        if rand <= 0.3 and second_most_likely:
            return random.choice(second_most_likely)
        elif rand <= 0.9:
            return random.choice(most_likely)
        else:
            return random.choice(others if others else most_likely)
    
    except Exception as e:
        print(f"Error in select_normalized: {str(e)}\n{traceback.format_exc()}")
        return None
    
# def select_normalized(normalized_dict: Dict[str, float]) -> Optional[str]:
#     """Selects a word based on weighted probabilities."""
#     if not normalized_dict:
#         return None
#     sorted_words = sorted(normalized_dict.items(), key=lambda x: x[1], reverse=True)
#     return random.choice([word[0] for word in sorted_words])

def generate_sentences(
    word_list: List[Dict[str, Any]], min_length: int = 5, max_length: int = 50, num_sentences: int = 5
) -> List[Dict[str, str]]:
    """Generates sentences ensuring all words appear in consecutive order while respecting min/max length."""
    
    # Extract normalized words using select_normalized
    words_normalized = []
    words_unnormalized = []
    
    for word_data in word_list:
        if "normalized" in word_data and isinstance(word_data["normalized"], dict):
            normalized_dict = word_data["normalized"]
            selected_word = select_normalized(normalized_dict)
            if selected_word:
                words_normalized.append(selected_word)
        if "unnormalized" in word_data:
            words_unnormalized.append(word_data["unnormalized"])

    # print(f"Words Normalized: {words_normalized}")
    # print(f"Words Unnormalized: {words_unnormalized}")

    if len(words_normalized) < min_length:
        return []

    sentences = []
    sentence_start = 0

    while sentence_start < len(words_normalized):
        sentence_end = sentence_start + random.randint(min_length, max_length)
        sentence_end = min(sentence_end, len(words_normalized))  # Ensure no out-of-bounds

        chunk_normalized = words_normalized[sentence_start:sentence_end]
        chunk_unnormalized = words_unnormalized[sentence_start:sentence_end]

        if chunk_normalized and chunk_unnormalized:
            sentence_normalized = remove_punctuation(" ".join(chunk_normalized))
            sentence_unnormalized = remove_punctuation(" ".join(chunk_unnormalized))

            sentences.append({
                "normalized": sentence_normalized, 
                "unnormalized": sentence_unnormalized
            })

        sentence_start = sentence_end  # Move the start point for the next sentence

    return sentences

# Main script execution
if __name__ == "__main__":
    input_file = 'earnings.json'
    output_file = 'output.json'

    # try:
    #     num_sentences = int(input("Enter the number of random sentences to generate: "))
    #     if num_sentences <= 0:
    #         raise ValueError("Number of sentences must be greater than 0.")
    # except ValueError as e:
    #     print(f"Invalid input: {e}")
    #     num_sentences = 5

    data = load_json(input_file)
    if data is not None:
        processed_data = {}
        for item in data:
            if not isinstance(item, dict):
                print(f"Skipping invalid item: {item}")
                continue

            for id_, word_lists in item.items():
                if not isinstance(word_lists, list) or not word_lists:
                    print(f"Skipping invalid word_lists for ID {id_}: {word_lists}")
                    continue
                
                all_sentences = []
                for word_list in word_lists:
                    if not isinstance(word_list, list):
                        print(f"Skipping non-list entry in ID {id_}: {word_list}")
                        continue
                    
                    sentences = generate_sentences(word_list)
                    all_sentences.extend(sentences)

                processed_data[id_] = all_sentences  # Store all generated sentences for the ID

        # Check if there are any processed data
        if processed_data:
            try:
                with open(output_file, 'w') as file:
                    json.dump(processed_data, file, indent=4)
                print(f"✅ Successfully saved output to {output_file}")
            except Exception as e:
                print(f"Failed to write output to {output_file}: {str(e)}\n{traceback.format_exc()}")
        else:
            print("No sentences were generated, output file not written.")
    else:
        print("No data loaded. Exiting.")

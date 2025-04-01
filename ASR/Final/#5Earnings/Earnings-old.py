import json
import random
import traceback
import time
from bisect import bisect
from typing import List, Dict, Any, Optional, Set 
import math

# load file
def load_json(file_path: str) -> Optional[List[Dict[str, Any]]]:
        with open(file_path, 'r') as file:
            data = json.load(file)
            return data

# select word based on weights
def select_normalized(normalized_dict):
    """Selects a normalized word based on weighted probabilities."""
    try:
        if not isinstance(normalized_dict, dict):
            raise ValueError(f"Expected a dictionary for normalized values, but got {type(normalized_dict).__name__}")

        sorted_normalized = sorted(normalized_dict.items(), key=lambda x: x[1], reverse=True)
        normalized_forms = [item[0] for item in sorted_normalized]
        # probabilities = [item[1] for item in sorted_normalized]

        if not normalized_forms:
            raise ValueError("Empty normalized forms list")
        
        # if len(sorted_normalized) > 2:
        #     print(sorted_normalized)
        # print(normalized_forms)

        if len(normalized_forms) == 1:
            return normalized_forms[0]
        elif len(normalized_forms) > 1:
            rand = random.random()
            print(rand)
            # print(normalized_forms)
            if rand <= 0.3:
                print(normalized_forms[1])
                return normalized_forms[1]
            elif rand <= 0.9 and len(normalized_forms) > 1:
                print(normalized_forms[0])
                return normalized_forms[0]
            else:
                # print(normalized_forms[2:])
                tempRandom = random.choice(normalized_forms[2:] if len(normalized_forms) > 2 else normalized_forms)
                print(tempRandom)
                return tempRandom
    except Exception as e:
        print(f"Error in select_normalized: {str(e)}\n{traceback.format_exc()}")
        return None

def calculate_max_iterations(n: int, min_length: int, max_length: int) -> int:
    """Calculate a reasonable max_iterations based on sentence combinations."""
    if n == 0:
        return 100  # Default for empty list
    
    # Adjust max_length to be no larger than n
    max_length = min(max_length, n)

    # Calculate possible sentences by summing valid sentence lengths
    possible_sentences = sum(n - length + 1 for length in range(min_length, max_length + 1))

    # Use logarithmic scaling for the number of possible sentences to avoid excessively large values
    # return min(int(possible_sentences * math.log2(possible_sentences + 1)), 1_000_000)
    
    # Multiply by √N for controlled variation
    return min(int(possible_sentences * math.sqrt(possible_sentences)), 1_000_000)

# Dynamic multiplier: larger inputs get smaller multipliers
    # if possible_sentences < 100:
    #     multiplier = 10
    # elif possible_sentences < 1_000:
    #     multiplier = 5
    # elif possible_sentences < 10_000:
    #     multiplier = 3
    # else:
    #     multiplier = 2  # Limit for very large inputs

    # return min(possible_sentences * multiplier, 1_000_000)

    # multiplier = max(2, int(10 / (math.log(possible_sentences + 1) + math.sqrt(possible_sentences + 1))))

    # return min(possible_sentences * multiplier, 1_000_000)

def generate_sentences(
    word_list: List[Dict[str, Any]], 
    min_length: int = 3, # sentence should have at least 3 words
    max_length: int = 10, 
    num_sentences: int = 5
) -> List[str]:
    """Generates random sentences with a time constraint."""
    try:
        # Validate input
        if not isinstance(word_list, list):
            raise ValueError("word_list must be a list of words.")
        
        if len(word_list) < min_length:
            print("Word list is too small to generate meaningful sentences.")
            return []
        
        sentences: Set[str] = set()
        start_time = time.time()
        time_limit = 20  # 20 seconds limit
        stale_count, stale_threshold = 0, 500
        
        # Define currency words and multipliers
        CURRENCY_WORDS = {"dollars", "dollar", "bucks", "buck", "cents", "cent", "pesos", "peso"}
        MULTIPLIERS = {"million", "millions", "billion", "trillion", "thousand", "thousands", "hundred"}
        
        # Calculate max_iterations based on possible unique sentences
        max_iterations = calculate_max_iterations(len(word_list), min_length, max_length)
        
        # Precompute maximum valid starting index
        max_start_index = max(0, len(word_list) - min_length)
        
        iterations = 0
        # print(len(word_list))
        # print("here")
        
        while len(sentences) < num_sentences and iterations < max_iterations:
            # print(len(sentences))
            # print( num_sentences)
            if len(sentences) >= num_sentences:
                break

            # Check for stopping conditions
            if iterations > max_iterations or stale_count >= stale_threshold or time.time() - start_time > time_limit:
                print(
                    f"Stopping due to: {'max iterations' if iterations > max_iterations else ''} "
                    f"{'too many repeated sentences' if stale_count >= stale_threshold else ''} "
                    f"{'time limit' if time.time() - start_time > time_limit else ''}"
                )
                break

            # Generate a random sentence
            start = random.randint(0, max_start_index)
            length = random.randint(min_length, min(max_length, len(word_list) - start))
            sentence_words = []
            last_currency_index = None
            
            for i in range(start, start + length):
                if i >= len(word_list):
                    break

                word = word_list[i]
                if not isinstance(word, dict):
                    print(f"Invalid word format at index {i}: {word}")
                    continue

                normalized_form = select_normalized(word.get('normalized', {}))
                if normalized_form is None:
                    print(f"Invalid normalized form at index {i} in word_list")
                    continue
                print(normalized_form)
                # Check if the current word is a currency word
                
                # print("second "+normalized_form)            
                lastSplit = normalized_form.strip().split()
                last = lastSplit[-1] if lastSplit else ""

                
                if last.lower() in CURRENCY_WORDS:
                    # print(last)
                    if lastSplit:
                        last_word = lastSplit.pop()
                        afterPop = " ".join(lastSplit)
                        # print("first "+afterPop)

                    
                    
                    last_currency_index = len(sentence_words)  # Track index of last currency word
                    sentence_words.append(normalized_form)
                    continue  # Proceed to next word

                # Check if the current word is a multiplier
                if normalized_form.lower() in MULTIPLIERS and last_currency_index is not None:
                    # print(f"test2") 
                    if 0 <= last_currency_index < len(sentence_words):  # Ensure a valid index
                        
                        # print("last_word: "+last_word)
                        # print("last: "+last)
                        # print("afterPop: "+afterPop)
                        currency_word = sentence_words.pop()  # Remove the currency word
                        sentence_words.append(afterPop)  # Then add the currency word after swapping
                        # sentence_words.append(normalized_form)  # Add the multiplier first
                        sentence_words.append(last)  # Then add the currency word after swapping
                        
                        sentence_words.append(last_word)  # Then add the currency word after swapping
                        
                        last_currency_index = None  # Reset currency tracking
                        # print(f"test")
                        continue  # Continue processing

                # Normal word, just append
                sentence_words.append(normalized_form)
                
                # Check if the last two words are a currency word followed by a multiplier
                if len(sentence_words) >= 2:
                    last_word = sentence_words[-2].lower()
                    current_word = sentence_words[-1].lower()
                    
                    if last_word in CURRENCY_WORDS and current_word in MULTIPLIERS:
                        # Swap the last two words
                        sentence_words[-2], sentence_words[-1] = sentence_words[-1], sentence_words[-2]
            
            sentence_str = ' '.join(sentence_words)

            if len(sentences) >= num_sentences:
                break 
            # print(len(sentences))
            if sentence_str not in sentences and len(sentences) < num_sentences:
                sentences.add(sentence_str)
                stale_count = 0
            else:
                stale_count += 1
            
            iterations += 1
        
        return list(sentences)[:num_sentences]
    except Exception as e:
        print(f"Error in generate_sentences: {str(e)}\n{traceback.format_exc()}")
        return []

# Main script execution
if __name__ == "__main__":
    input_file = 'earnings-test.json'
    output_file = 'output-test.json'

    try:
        num_sentences = int(input("Enter the number of random sentences to generate: "))
        if num_sentences <= 0:
            raise ValueError("Number of sentences must be greater than 0.")
    except ValueError as e:
        print(f"Invalid input: {e}")
        print(f"Invalid input: {e}. Using default value of 5 sentences.", flush=True)
        num_sentences = 5

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
                    
                    sentences = generate_sentences(word_list, num_sentences=num_sentences)
                    all_sentences.extend(sentences)

                processed_data[id_] = all_sentences[:num_sentences]  # Store all generated sentences for the ID

        # Write to the file after processing all IDs
        try:
            with open(output_file, 'w') as file:
                json.dump(processed_data, file, indent=4)
            print(f"✅ Successfully saved output to {output_file}", flush=True)
        except Exception as e:
            print(f"Failed to write output to {output_file}: {str(e)}\n{traceback.format_exc()}")
    else:
        print("No data loaded. Exiting.", flush=True)
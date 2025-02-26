import pandas as pd
import re
from difflib import SequenceMatcher

# Load data from TSV
# input_file = "spgi_test_1.tsv" 
input_file = "spgi_test_2.tsv" 
# input_file = "spgi_test_1_processed.tsv" 
data = pd.read_csv(input_file, delimiter="\t", dtype=str)

# Strip column names of leading/trailing spaces
data.rename(columns=lambda x: x.strip(), inplace=True)

# Ensure required columns exist
required_columns = {'unnormalized', 'normalized'}
if not required_columns.issubset(data.columns):
    raise ValueError(f"Missing required columns: {required_columns - set(data.columns)} in the input file.")

# Fill NaN values with empty strings
data[['unnormalized', 'normalized']] = data[['unnormalized', 'normalized']].fillna('')

# Function to remove punctuation except apostrophes
def remove_punctuation(text):
    text = re.sub(r"-", " ", text)  # Replace hyphens with spaces
    return re.sub(r"[^\w\s']", '', text)  # Remove all other punctuation except apostrophes

def contains_digit_or_symbol(word):
    """Check if the word contains any character that is NOT a letter or an apostrophe."""
    return bool(re.search(r"[^a-zA-Z']", word))  # Matches anything that is NOT a letter or an apostrophe

# Function to remove numbers and symbols
def remove_numbers_symbols(text):
    return re.sub(r'\d+', '', text)  # Removes all digits

# Function to check if normalized text is valid
def is_valid_normalized(text):
    return bool(text.strip())  # Returns False if text is empty or just spaces

# Function to detect abbreviations in the unnormalized text
def find_abbreviations(text):
    # Match abbreviations (uppercase words with optional dots)
    pattern = r'\b(?:[A-Z]\.?){2,}\b'
    return re.findall(pattern, text)

# Function to format abbreviations: lowercase + spaces between letters
def format_abbreviation(abbr):
    # return ' '.join(list(abbr.replace('.', '').lower()))
    # print("abbr "+abbr)
    # return re.sub(r'\b[A-Z]+\b', lambda x: x.group(0).lower(), abbr)
    sentence = ' '.join(list(abbr.lower()))
    return sentence
    # return abbr
    # return ' '.join(list(re.sub(r'\b[A-Z]{2,}\b', lambda x: x.group(0).lower(), abbr)))

# Function to categorize errors
def categorize_errors(unnormalized, normalized):
    if not isinstance(unnormalized, str) or not isinstance(normalized, str):
        return {'substitution': [], 'insertion': [], 'deletion': []}

    # Convert to lowercase and remove unwanted punctuation before sequence matching 
    unnormalized = remove_punctuation(unnormalized.lower())
    normalized = remove_punctuation(normalized.lower())

    # If normalized is empty or invalid, return empty string
    if not is_valid_normalized(normalized):
        return ""

    # Split into words
    unnormalized_words = unnormalized.split()
    normalized_words = normalized.split()

    # If normalized is empty or invalid, return empty errors
    if not is_valid_normalized(normalized):
        return {'substitution': [], 'insertion': [], 'deletion': []}

    # Create a SequenceMatcher instance
    sm = SequenceMatcher(None, unnormalized_words, normalized_words)

    # Error categories
    errors = {'substitution': [], 'insertion': [], 'deletion': []}

    # Compare sequences and categorize errors
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'replace':  # Substitution
            subs_unnormalized = unnormalized_words[i1:i2] if i1 < len(unnormalized_words) else [] # Words from unnormalized
            subs_normalized = normalized_words[j1:j2] if j1 < len(normalized_words) else []      # Words from normalized
            # print(subs_normalized)
            # print(subs_unnormalized)
            # if any(re.search(r'\d', item) for item in subs_unnormalized):
            #     errors['substitution'].append((subs_unnormalized, subs_normalized))
            # else:
            errors['substitution'].append((subs_unnormalized, subs_normalized))

        elif tag == 'insert':  # Insertion
            subs_unnormalized = unnormalized_words[i1:i2] if i1 < len(unnormalized_words) else [] # Words from unnormalized
            subs_normalized = normalized_words[j1:j2] if j1 < len(normalized_words) else []      # Words from normalized
            errors['insertion'].append((subs_unnormalized, subs_normalized))

        elif tag == 'delete':  # Deletion
            subs_unnormalized = unnormalized_words[i1:i2] if i1 < len(unnormalized_words) else [] # Words from unnormalized
            subs_normalized = normalized_words[j1:j2] if j1 < len(normalized_words) else []      # Words from normalized
            errors['deletion'].append((subs_unnormalized, subs_normalized))

    return errors

def generate_final_output(unnormalized, normalized):
    if not isinstance(unnormalized, str) or not isinstance(normalized, str):
        return ""

    # Convert to lowercase and remove unwanted punctuation
    unnormalized_clean = remove_punctuation(unnormalized)
    normalized_clean = remove_punctuation(normalized)

    # If normalized is empty or invalid, return empty string
    if not is_valid_normalized(normalized):
        return ""
    
    # Split into words
    unnormalized_words = unnormalized_clean.split()
    normalized_words = normalized_clean.split()
    unnormalized_count = len(unnormalized_clean.split())

    # print(un_count)

    # Identify errors
    errors = categorize_errors(unnormalized, normalized)

    # If no errors, return the normalized form (without numbers/symbols)
    if not errors['substitution'] and not errors['insertion'] and not errors['deletion']:
        # If no errors, but digits found in normalized form, return empty string
        # if any(re.search(r'\d', item) for item in normalized_words):
        #     return ""
        # else:
        normalized_words = [
            format_abbreviation(word) if re.fullmatch(r'[A-Z]{2,}(s|S|ed|ED|Ed|eD)?', word) else word
            for word in normalized_words
        ]

        return ' '.join(normalized_words) 

    # Reconstruct sentence while keeping necessary corrections
    final_output = []

    # if ((len(normalized_words)-len(unnormalized_words))/len(unnormalized_words)*100>30):
    #     final_output = ""
    #     return final_output
        # print((len(normalized_words)-len(unnormalized_words))/len(normalized_words)*100>30)
        # print(len(unnormalized_words))
        # print(len(normalized_words))


    for tag, i1, i2, j1, j2 in SequenceMatcher(None, unnormalized_words, normalized_words).get_opcodes():
        if tag == 'replace':
            subs_unnormalized = unnormalized_words[i1:i2] 
            # if i1 < len(unnormalized_words) else [] # Words from unnormalized
            subs_normalized = normalized_words[j1:j2] 
            substituted_count = len(subs_normalized)
            # if j1 < len(normalized_words) else []      # Words from normalized

            # print(subs_normalized)
            # print(subs_unnormalized)
            # if any(contains_digit_or_symbol(word) for word in subs_unnormalized):
            #     # If any word in unnormalized contains a digit, append ALL normalized words
            #     final_output.extend(subs_normalized)
            # elif any(contains_digit_or_symbol(word) for word in subs_normalized):
            #     # If any word in unnormalized contains a digit, append ALL normalized words
            #     final_output.extend(subs_unnormalized)
            # else:
            for unnorm_word, norm_word in zip(subs_unnormalized, subs_normalized):
                
                # debug
                # print("un: " + unnorm_word)
                # print("norm:" + norm_word)
                # print(tag)
                # print(subs_unnormalized)
                # print(subs_normalized)
                # print(substituted_count)
                # print(unnormalized_count)
                # print(substituted_count/unnormalized_count)
                # print(final_output)

                if(substituted_count/unnormalized_count>0.33):
                    final_output = ""
                    return final_output
                else:
                    #unnorm contains digits
                    if any(contains_digit_or_symbol(word)for word in subs_unnormalized) and not contains_digit_or_symbol(norm_word):
                        for word in subs_normalized:
                            final_output.append(word)
                        break

                    #norm contains digits
                    elif not contains_digit_or_symbol(unnorm_word) and any(contains_digit_or_symbol(word)for word in subs_normalized):
                        for word in subs_unnormalized:
                            final_output.append(word)
                        break

                    #both dont contain digits
                    elif not contains_digit_or_symbol(unnorm_word) and not contains_digit_or_symbol(norm_word):
                        for word in subs_unnormalized:
                            final_output.append(word)
                        break

                    #both contains
                    elif contains_digit_or_symbol(unnorm_word) and contains_digit_or_symbol(norm_word) :
                        final_output = ""
                        return final_output

                    else:
                        return ""

        # elif tag == 'insert':
        #     subs_unnormalized = unnormalized_words[i1:i2] if i1 < len(unnormalized_words) else [] # Words from unnormalized
        #     subs_normalized = normalized_words[j1:j2] if j1 < len(normalized_words) else []      # Words from normalized
        #     for word in normalized_words[j1:j2]:
        #         if not contains_digit_or_symbol(word): 
        #             final_output.append(word)

        elif tag == 'equal':

            # debug
            # print('equal')
            # print(tag)
            # print(unnormalized_words[i1:i2])

            final_output.extend(unnormalized_words[i1:i2])  # Keep equal words

        elif tag == 'delete':
            subs_unnormalized = unnormalized_words[i1:i2] if i1 < len(unnormalized_words) else [] # Words from unnormalized
            subs_normalized = normalized_words[j1:j2] if j1 < len(normalized_words) else []      # Words from normalized
            deleted_count = len(subs_unnormalized)

            if (deleted_count/unnormalized_count)>0.33:
                final_output=""
                return final_output
            else:
                for word in unnormalized_words[i1:i2]:
                    if not contains_digit_or_symbol(word): 
                        final_output.append(word)

    # Detect abbreviations and apply formatting
    final_output = [
        # format_abbreviation(word) if len(word) >= 2 and word.isupper() else word
        format_abbreviation(word) if re.fullmatch(r'[A-Z]{2,}(s|S|ed|ED|Ed|eD)?', word) else word
                
        for word in final_output
    ]
    # for word in final_output:
    # print(final_output)
    return remove_numbers_symbols(' '.join(final_output))

# Apply functions safely
try:
    data['errors'] = data.apply(lambda row: categorize_errors(row['unnormalized'], row['normalized']), axis=1)
    data['final_output'] = data.apply(lambda row: generate_final_output(row['unnormalized'], row['normalized']), axis=1)
    data['compare'] = data.apply(lambda row: categorize_errors(row['unnormalized'], row['final_output']), axis=1)
  
except Exception as e:
    print(f"Error during processing: {e}")
    exit()

# Print results
# for index, row in data.iterrows():
#     print(f"Filename: {row.get('filename', 'N/A')}")
#     print(f"Unnormalized: {row['unnormalized']}")
#     print(f"Normalized: {row['normalized']}")
#     print(f"Errors: {row['errors']}")
#     print(f"Final Output: {row['final_output']}")
#     print('-' * 50)

# Save output as TSV
# data.to_csv("spgi_output_1.csv", index=False)
data.to_csv("spgi_output_2.csv", index=False)
# data.to_csv("spgi_test_1_processed.csv", index=False)
# data.to_csv("spgi_output_2.tsv", index=False, sep='\t')

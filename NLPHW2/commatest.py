# import json
# import os
# import sys
# import math
# import collections
#
# """
# This script analyzes the influence of commas in the dataset.
# It will:
# 1. Read the original JSONL file.
# 2. Count how often commas appear as tokens.
# 3. Optionally create a "cleaned" version of the dataset where commas are removed from the text.
# 4. The idea is to later run the perplexity calculation on both the original and the cleaned dataset
#    (using the previously built system) and compare results.
#
# This script only performs the analysis and optional cleaning, not the perplexity test.
# You can integrate this cleaned dataset into the main script and see if perplexity improves.
#
# Instructions:
# - Make sure 'result.jsonl' is in the same directory.
# - This script will produce 'result_cleaned.jsonl' (if chosen) with commas removed.
#
# Note:
# We're not changing the main code here. This is just a separate script as requested.
# """
#
# def analyze_and_clean_dataset(input_file='result.jsonl', output_file='result_cleaned.jsonl', remove_commas=True):
#     if not os.path.exists(input_file):
#         print(f"Input file '{input_file}' not found.")
#         sys.exit(1)
#
#     print(f"Reading dataset from {input_file}...")
#
#     total_tokens = 0
#     comma_count = 0
#
#     records = []
#     with open(input_file, 'r', encoding='utf-8') as f:
#         for line in f:
#             line = line.strip()
#             if not line:
#                 continue
#             obj = json.loads(line)
#             text = obj.get('text_sentence', '').strip()
#             tokens = text.split()
#
#             total_tokens += len(tokens)
#             # Count how many commas are present as separate tokens
#             # We assume commas are separate tokens (e.g., ",").
#             # If commas appear attached to words, might need a different approach.
#             for t in tokens:
#                 if t == ',':
#                     comma_count += 1
#
#             if remove_commas:
#                 # Remove commas from the token list
#                 tokens = [t for t in tokens if t != ',']
#                 obj['text_sentence'] = ' '.join(tokens)
#
#             records.append(obj)
#
#     comma_ratio = (comma_count / total_tokens) if total_tokens > 0 else 0
#
#     print(f"Total tokens: {total_tokens}")
#     print(f"Comma tokens: {comma_count}")
#     print(f"Ratio of commas: {comma_ratio:.4f} (commas per token)")
#
#     if remove_commas:
#         print(f"Writing cleaned dataset to {output_file} without commas...")
#         with open(output_file, 'w', encoding='utf-8') as out_f:
#             for r in records:
#                 out_f.write(json.dumps(r, ensure_ascii=False) + '\n')
#         print("Cleaned dataset written.")
#
#     # At this point, you can run your main script on 'result_cleaned.jsonl'
#     # to compare perplexity before and after removing commas.
#
# if __name__ == "__main__":
#     # Analyze the original dataset for commas and produce a cleaned version
#     # Set remove_commas=False if you only want to analyze without producing the cleaned file.
#     analyze_and_clean_dataset(remove_commas=True)

import json
import re

# Define the file name
file_name = 'result_cleaned.jsonl'

# Counters for specific conditions
start_end_punct_count = 0
non_hebrew_num_end_count = 0
empty_ending_count=0
# Open the JSONL file and process each line
try:
    with open(file_name, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                # Parse the JSON object from the line
                data = json.loads(line)

                # Check if "text_sentence" exists
                if 'text_sentence' in data:
                    text = data['text_sentence']

                    # Condition 1: Starts/ends with punctuation
                    if text.startswith((',', '-', '.')) or text.endswith((',', '-', '.')):
                        start_end_punct_count += 1
                        print(text)

                    # Condition 2: Does not end with a Hebrew character or a number
                    if not re.search(r'[\u0590-\u05FF\d]$', text):
                        non_hebrew_num_end_count += 1

                    if text.endswith((' ', '\n','\t')):
                        empty_ending_count+=1

            except json.JSONDecodeError:
                print("Error decoding JSON line:", line)

    # Print the counts
    print(f"Count of sentences starting/ending with punctuation: {start_end_punct_count}")
    print(f"Count of sentences not ending with a Hebrew character or number: {non_hebrew_num_end_count}")
    print(f"Count of sents ending iwht space/newline/tab: {empty_ending_count}")

except FileNotFoundError:
    print(f"File '{file_name}' not found.")

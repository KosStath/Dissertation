import json
import os
import itertools
from collections import defaultdict


output_dir = r'C:\Users\kstat\Documents\Dissertation\Data\Methods\Method_2\Co-Occurrence'
os.makedirs(output_dir, exist_ok=True)

# Function to split the keyword string into a list of individual keywords
def split_keywords(keywords_str):
    return [keyword.strip() for keyword in keywords_str.split(',')] if keywords_str else []

# Function to process a batch of records and calculate co-occurrences
def process_batch(batch_data, co_occurrence_data, no_keywords_data, abstracts_per_year):
    for entry in batch_data:
        year = entry["Year"]
        keywords = split_keywords(entry.get("Extracted_Keywords", ""))

        # If no keywords are found, track that for the "No Keywords" analysis
        if not keywords:
            no_keywords_data[year] += 1
            continue  # Skip processing co-occurrences for this abstract

        # Track the number of abstracts per year
        abstracts_per_year[year] += 1
        
        # Generate all pairs of co-occurring keywords in this abstract
        keyword_pairs = list(itertools.combinations(keywords, 2))
        
        # Count co-occurrences for each pair of keywords for the current year
        for pair in keyword_pairs:
            pair_string = '-'.join(pair)
            co_occurrence_data[year][pair_string] += 1

# Main function to generate co-occurrence frequencies and save results
def generate_co_occurrence_frequencies(input_file_path, output_file_path):
    
    with open(input_file_path, 'r', encoding='utf-8') as infile:
        data = json.load(infile)

    print(f"Starting co-occurrence analysis for {len(data)} records...")

    # Dictionaries to store co-occurrence frequencies by year
    co_occurrence_data = defaultdict(lambda: defaultdict(int))
    no_keywords_data = defaultdict(int)
    abstracts_per_year = defaultdict(int)

    # Process the data in batches 
    batch_size = 1000  
    for i in range(0, len(data), batch_size):
        batch_data = data[i:i + batch_size]
        print(f"Processing batch {i // batch_size + 1} ({i + 1} to {min(i + batch_size, len(data))})...")

        process_batch(batch_data, co_occurrence_data, no_keywords_data, abstracts_per_year)

    # Normalize co-occurrence data by abstracts per year
    normalized_co_occurrence_data = defaultdict(lambda: defaultdict(float))
    for year, pairs in co_occurrence_data.items():
        for pair, count in pairs.items():
            normalized_co_occurrence_data[year][pair] = count / abstracts_per_year[year]

    # Prepare the top 30 co-occurrence pairs for each year
    top_n = 30
    top_co_occurrence_data = defaultdict(list)
    for year, pairs in normalized_co_occurrence_data.items():
        # Sort the pairs for this year based on normalized count and get top N
        sorted_pairs = sorted(pairs.items(), key=lambda x: x[1], reverse=True)[:top_n]
        top_co_occurrence_data[year] = sorted_pairs

    
    co_occurrence_output = {
        "co_occurrence_frequencies": top_co_occurrence_data,
        "no_keywords_count_by_year": no_keywords_data
    }

    
    with open(output_file_path, 'w', encoding='utf-8') as outfile:
        json.dump(co_occurrence_output, outfile, indent=4, ensure_ascii=False)

    print(f"Co-occurrence frequencies saved to {output_file_path}")


input_file_path = r'C:\Users\kstat\Documents\Dissertation\Data\datasetWithKeys.json'
output_file_path = os.path.join(output_dir, 'co-occurrenceFreqsNormalizedTop30ByYear.json')


generate_co_occurrence_frequencies(input_file_path, output_file_path)

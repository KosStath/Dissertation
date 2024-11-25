import json
from rapidfuzz import fuzz
import spacy
import re


nlp = spacy.load("en_core_sci_md")


SIMILARITY_THRESHOLD = 90  
BATCH_SIZE = 1000  

# Load glossaries
with open(r'C:\Users\kstat\Documents\Dissertation\Data\Lexicons\glossaryKeys_Hyphenated.json', 'r', encoding='utf-8') as f:
    hyphenated_glossary = json.load(f)

with open(r'C:\Users\kstat\Documents\Dissertation\Data\Lexicons\glossaryKeys_Spaced.json', 'r', encoding='utf-8') as f:
    spaced_glossary = json.load(f)

# Function to check SciSpaCy vocabulary
def is_in_scispacy_vocab(term):
    """Check if a term is in the SciSpaCy model's vocabulary."""
    return term in nlp.vocab

# Function for processing a batch of abstracts
def process_batch(batch, results, recognized_terms, unrecognized_terms):
    """Process a batch of abstracts to match glossary terms and check SciSpaCy matching."""
    for record in batch:
        abstract_id = record['id']
        abstract_text = record['Abstract']

        # Initialize tracking for this abstract
        results[abstract_id] = {
            "direct_matches": [],
            "fuzzy_matches": []
        }
        matched_terms = set()

        # Direct matching with hyphenated glossary
        for term in hyphenated_glossary:
            if term in abstract_text:
                results[abstract_id]["direct_matches"].append(term)
                matched_terms.add(term)

        # Fuzzy matching with spaced glossary for unmatched terms
        for term in spaced_glossary:
            if term not in matched_terms:  # Skip if already matched directly
                similarity = fuzz.partial_ratio(term, abstract_text)
                if similarity >= SIMILARITY_THRESHOLD:
                    results[abstract_id]["fuzzy_matches"].append({"term": term, "score": similarity})
                    matched_terms.add(term)

        # Track terms for SciSpaCy matching
        for term in matched_terms:
            if is_in_scispacy_vocab(term):
                if term not in recognized_terms:
                    recognized_terms[term] = {"count": 0, "abstracts": []}
                recognized_terms[term]["count"] += 1
                recognized_terms[term]["abstracts"].append(abstract_id)
            else:
                if term not in unrecognized_terms:
                    unrecognized_terms[term] = {"count": 0, "abstracts": []}
                unrecognized_terms[term]["count"] += 1
                unrecognized_terms[term]["abstracts"].append(abstract_id)

# Main function for processing the dataset
def process_dataset(input_file, output_file):
    """Process the dataset in batches and generate the final JSON output."""
    with open(input_file, 'r', encoding='utf-8') as infile:
        # Load the preprocessed abstracts
        data = json.load(infile)

    total_records = len(data)
    results = {}
    recognized_terms = {}
    unrecognized_terms = {}

    # Process the data in batches
    for start in range(0, total_records, BATCH_SIZE):
        end = min(start + BATCH_SIZE, total_records)
        batch = data[start:end]

        
        process_batch(batch, results, recognized_terms, unrecognized_terms)

        
        print(f"Processed {end}/{total_records} records.")

    # Prepare the final output
    output = {
        "summary": {
            "total_detected_terms": len(recognized_terms) + len(unrecognized_terms),
            "recognized_by_scispacy": len(recognized_terms),
            "unrecognized_by_scispacy": len(unrecognized_terms)
        },
        "details": {
            "recognized_terms": recognized_terms,
            "unrecognized_terms": unrecognized_terms
        }
    }

    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(output, outfile, ensure_ascii=False, indent=4)

    print(f"Processing completed. Results saved to {output_file}")


input_file = r'C:\Users\kstat\Documents\Dissertation\Data\Methods\Method_2\Keywords(Direct)\preproAbs_NLTK.json'
output_file = r'C:\Users\kstat\Documents\Dissertation\Data\Methods\Method_2\Keywords(Direct)\terms_SciSpaCy.json'


process_dataset(input_file, output_file)

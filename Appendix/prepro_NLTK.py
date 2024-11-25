import nltk
import json
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist


nltk.download('punkt')
nltk.download('stopwords')


stop_words = set(stopwords.words('english'))

# Function to load glossary terms
def load_glossary(glossary_file):
    with open(glossary_file, 'r', encoding='utf-8') as file:
        glossary_terms = json.load(file)
    # Sort terms by length to handle multiwords
    return sorted(glossary_terms, key=len, reverse=True)

# Function to preprocess a single abstract
def preprocess_abstract(abstract, glossary_terms):
    # Step 1: Replace glossary terms with placeholders
    placeholder_map = {}  # Map for storing glossary terms to placeholders
    counter = 1  # Counter for generating unique placeholders
    
    for term in glossary_terms:
        placeholder = f"__GLOSSARY_TERM_{counter}__"
        placeholder_map[placeholder] = term
        # Use regex with word boundaries to match exact terms
        abstract = re.sub(r'\b' + re.escape(term) + r'\b', placeholder, abstract)
        counter += 1
    
    # Step 2: Tokenize the abstract
    tokens = word_tokenize(abstract)
    
    # Step 3: Process tokens
    processed_tokens = []
    for token in tokens:
        # Preserve numbers and mathematical symbols
        if re.match(r'^[0-9.,:><=%]+$', token) or re.search(r'[^\x00-\x7F]', token):
            processed_tokens.append(token)
        # Ignore stopwords
        elif token.lower() not in stop_words:
            # Remove punctuation around tokens, except for numbers
            clean_token = re.sub(r'^[^\w]+|[^\w]+$', '', token)  # Strip leading/trailing punctuation
            processed_tokens.append(clean_token)
    
    # Step 4: Restore glossary terms from placeholders
    restored_tokens = []
    for token in processed_tokens:
        if token in placeholder_map:
            restored_tokens.append(placeholder_map[token])
        else:
            restored_tokens.append(token)
    
    # Step 5: Remove empty tokens caused by over-cleaning
    restored_tokens = [token for token in restored_tokens if token.strip()]
    
    return restored_tokens

# Function to process abstracts in batches
def process_in_batches(input_file, glossary_file, batch_size, output_file_abstracts, output_file_frequencies):
    # Load dataset and glossary
    with open(input_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    glossary_terms = load_glossary(glossary_file)

    # Initialize frequency distribution and tracking
    freq_dist = FreqDist()
    term_abstract_map = {}

    # Process abstracts in batches
    total_records = len(data)
    for i in range(0, total_records, batch_size):
        batch = data[i:i + batch_size]
        print(f"Processing batch {i // batch_size + 1}/{(total_records + batch_size - 1) // batch_size}...")
        
        for record in batch:
            abstract = record['Abstract']
            preprocessed_tokens = preprocess_abstract(abstract, glossary_terms)
            record['Abstract'] = preprocessed_tokens  # Update the record with the preprocessed abstract
            
            # Detect glossary terms and update frequency distribution
            detected_terms = [term for term in set(preprocessed_tokens) if term in glossary_terms]
            for term in detected_terms:
                freq_dist[term] += 1
                if term not in term_abstract_map:
                    term_abstract_map[term] = []
                term_abstract_map[term].append(record['id'])
        
        print(f"Completed processing {min(i + batch_size, total_records)} of {total_records} records.")
    
    # Save preprocessed abstracts to a new file
    with open(output_file_abstracts, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
    
    # Save frequency distribution and abstract IDs to a new file
    frequencies_output = {
        term: {
            "global_count": freq_dist[term],
            "abstracts": term_abstract_map[term]
        }
        for term in freq_dist.keys()
    }
    with open(output_file_frequencies, 'w', encoding='utf-8') as file:
        json.dump(frequencies_output, file, ensure_ascii=False, indent=4)

# Define file paths and parameters
input_file = r'C:\Users\kstat\Documents\Dissertation\Data\finalDatasetHyphenated.json'  # Path to the input JSON file
glossary_file = r'C:\Users\kstat\Documents\Dissertation\Data\Lexicons\glossaryKeys_Hyphenated.json'  # Path to the glossary JSON file
output_file_abstracts = 'preproAbs_NLTK.json'  # File to save preprocessed abstracts
output_file_frequencies = 'termsFreqWithAbsIds.json'  # File to save term frequencies
batch_size = 1000  # Number of records to process in each batch

# Run the batch processing
process_in_batches(input_file, glossary_file, batch_size, output_file_abstracts, output_file_frequencies)

from sentence_transformers import SentenceTransformer
import json
# NLTK is no longer needed for tokenization, but we'll keep the import in case of other uses.
# You can safely remove it if you don't use it elsewhere.
import nltk 
import numpy as np
import pandas as pd
from tqdm import tqdm

# --- Step 1: Initialize Model and Load Data ---
print("Step 1: Initializing model and loading data...")
model_name = 'all-mpnet-base-v2'
model = SentenceTransformer(model_name, device='cuda')

with open(r'cleaned_texts\all_processed.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# --- Step 2: Prepare Paragraphs for Encoding ---
# This section has been modified to work with paragraphs directly.
print("\nStep 2: Preparing paragraphs for encoding...")
all_paragraphs_flat = []
# This new list will help us reconstruct the hierarchy for aggregation later.
paragraph_counts_per_chapter = [] 

for book in tqdm(data, desc='Processing books into paragraphs'):
    book_paragraph_counts = []
    for chapter in book['content']:
        # We only consider non-empty paragraphs
        valid_paragraphs = [p for p in chapter if p.strip()] 
        if valid_paragraphs:
            all_paragraphs_flat.extend(valid_paragraphs)
            book_paragraph_counts.append(len(valid_paragraphs))
        else:
            book_paragraph_counts.append(0)
    paragraph_counts_per_chapter.append(book_paragraph_counts)

print(f'Total paragraphs to be embedded: {len(all_paragraphs_flat)}')

# --- Step 3: Encode Paragraphs ---
# This section now encodes the flat list of paragraphs.
print("\nStep 3: Encoding paragraphs...")
paragraph_embeddings_flat = model.encode(
    all_paragraphs_flat,
    show_progress_bar=True,
    batch_size=128,
)

# --- Step 4: Aggregate Embeddings (Paragraph -> Chapter -> Book) ---
# This section has been completely rewritten for clarity and correctness.
print("\nStep 4: Aggregating embeddings...")

# These lists will store the final aggregated embeddings for each level.
book_embeddings_aggregated = []
chapter_embeddings_aggregated_all_books = []
# This will be a nested list matching the original data structure.
# Note: We already have paragraph_embeddings_flat, which is more useful for the DataFrame.
paragraph_embeddings_hierarchical = [] 

current_paragraph_idx = 0
for i, book in enumerate(tqdm(data, desc='Aggregating embeddings')):
    # Embeddings for the current book
    book_chapter_embeddings = []
    book_paragraph_embeddings_hierarchical = []

    for j, chapter in enumerate(book['content']):
        num_paragraphs = paragraph_counts_per_chapter[i][j]
        
        if num_paragraphs > 0:
            # Slice the flat embeddings to get all paragraphs for this chapter
            chapter_p_embeddings = paragraph_embeddings_flat[current_paragraph_idx : current_paragraph_idx + num_paragraphs]
            
            # Aggregate paragraphs to get a single chapter embedding
            chapter_agg_embedding = np.mean(chapter_p_embeddings, axis=0)
            book_chapter_embeddings.append(chapter_agg_embedding)
            
            # Store the hierarchical paragraph embeddings for this chapter
            book_paragraph_embeddings_hierarchical.append(list(chapter_p_embeddings))
            
            current_paragraph_idx += num_paragraphs
        else:
            # If a chapter has no paragraphs, add an empty list
            book_paragraph_embeddings_hierarchical.append([])

    # Aggregate chapters to get a single book embedding
    if book_chapter_embeddings:
        book_agg_embedding = np.mean(book_chapter_embeddings, axis=0)
        book_embeddings_aggregated.append(book_agg_embedding)
    
    # Store the aggregated results for this book
    chapter_embeddings_aggregated_all_books.append(book_chapter_embeddings)
    paragraph_embeddings_hierarchical.append(book_paragraph_embeddings_hierarchical)


print("\nAggregation Complete!")
if book_embeddings_aggregated:
    print(f"Shape of one book embedding (aggregated): {book_embeddings_aggregated[0].shape}")
if chapter_embeddings_aggregated_all_books and chapter_embeddings_aggregated_all_books[0]:
    print(f"Shape of one chapter embedding (aggregated): {chapter_embeddings_aggregated_all_books[0][0].shape}")
if paragraph_embeddings_hierarchical and paragraph_embeddings_hierarchical[0] and paragraph_embeddings_hierarchical[0][0]:
     print(f"Shape of one paragraph embedding (original): {paragraph_embeddings_hierarchical[0][0][0].shape}")

# --- Step 5: Create Paragraph-level DataFrame ---
# This section has been modified to create one row per paragraph.
print("\nStep 5: Creating the Master DataFrame for paragraphs...")

records = []
for i, book in enumerate(tqdm(data, desc="Building DataFrame")):
    book_title = book['meta'].get('Original Title', f'book_{i}')
    book_author = book['meta'].get('Original Writer')
    year = book['meta'].get('Publication Year (Original)')
    for j, chapter in enumerate(book['content']):
        # We must iterate in the exact same way as in Step 2 to ensure alignment
        valid_paragraphs = [p for p in chapter if p.strip()]
        if valid_paragraphs:
            for k, paragraph in enumerate(valid_paragraphs):
                records.append({
                    "book_title": book_title,
                    "author": book_author,
                    "year": year,
                    "book_index": i,
                    "chapter_index": j,
                    "paragraph_index": k,
                    "text": paragraph
                })

df_paragraphs = pd.DataFrame(records)

# --- Step 6: Add Embeddings to DataFrame and Save ---
# This section now validates and saves the paragraph-level data.
print("\nStep 6: Adding embeddings and validating...")

print(f"Number of rows in DataFrame: {len(df_paragraphs)}")
print(f"Number of embeddings:        {len(paragraph_embeddings_flat)}")

assert len(df_paragraphs) == len(paragraph_embeddings_flat), "Mismatch between DataFrame rows and number of embeddings!"

df_paragraphs['embedding'] = list(paragraph_embeddings_flat)

print("\nDataFrame with paragraph embeddings created successfully.")
print(df_paragraphs.head())
print(f"\nShape of the DataFrame: {df_paragraphs.shape}")
print(f"Data type of the 'embedding' column: {type(df_paragraphs['embedding'].iloc[0])}")

print("\nStep 7: Saving the DataFrame to Parquet format...")

output_path = 'paragraph_embeddings_with_metadata.parquet'
df_paragraphs.to_parquet(output_path, index=False)

print(f"DataFrame successfully saved to {output_path}")
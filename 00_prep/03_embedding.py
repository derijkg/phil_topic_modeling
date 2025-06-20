from sentence_transformers import SentenceTransformer
import json
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    print("Downloading NLTK's 'punkt' model for sentence tokenization...")
    nltk.download('punkt')
import numpy as np
import pandas as pd
from tqdm import tqdm


# EMBEDDING SENTENCE LEVEL
model_name = 'all-mpnet-base-v2'
model = SentenceTransformer(model_name, device='cuda')

with open(r'cleaned_texts\all_processed.json', 'r',encoding='utf-8') as f:
    data = json.load(f)

# partitioning
all_sentences_flat = []
sentence_counts = []
for book in tqdm(data, desc='Processing books'):
    book_sentence_counts = []
    for chapter in book['content']:
        chapter_sentence_counts = []
        for paragraph in chapter:
            sentences = nltk.sent_tokenize(paragraph)
            if sentences:
                all_sentences_flat.extend(sentences)
                chapter_sentence_counts.append(len(sentences))
            else:
                chapter_sentence_counts.append(0)
        book_sentence_counts.append(chapter_sentence_counts)
    sentence_counts.append(book_sentence_counts)

print(f'total:{len(all_sentences_flat)}')

# tokenize sentences
sentence_embeddings_flat = model.encode(
    all_sentences_flat,
    show_progress_bar=True,
    batch_size=128,
    
)

sentence_embeddings_hierarchical = []
paragraph_embeddings_aggregated = []
chapter_embeddings_aggregated = []
book_embeddings_aggregated = []

current_sentence_idx = 0
for i, book in enumerate(tqdm(data,desc='agg embeddings')):
    book_chapter_embeddings = []
    book_paragraph_embeddings = []

    for j, chapter in enumerate(book['content']):
        chapter_paragraph_embeddings = []

        for k, paragraph in enumerate(chapter):
            num_sentences = sentence_counts[i][j][k]

            if num_sentences > 0:
                paragraph_s_embeddings = sentence_embeddings_flat[current_sentence_idx:current_sentence_idx+num_sentences]
                
                # paragraph
                paragraph_agg_embedding = np.mean(paragraph_s_embeddings, axis= 0)
                chapter_paragraph_embeddings.append(paragraph_agg_embedding)
                current_sentence_idx += num_sentences

        # chapter
        if chapter_paragraph_embeddings:
            chapter_agg_embedding = np.mean(chapter_paragraph_embeddings, axis=0)
            book_chapter_embeddings.append(chapter_agg_embedding)

        book_paragraph_embeddings.append(chapter_paragraph_embeddings)
# book
if book_chapter_embeddings:
    book_agg_embedding = np.mean(book_chapter_embeddings, axis=0)
    book_embeddings_aggregated.append(book_agg_embedding)

paragraph_embeddings_aggregated.append(book_paragraph_embeddings)
chapter_embeddings_aggregated.append(book_chapter_embeddings)

print("\nAggregation Complete!")
print(f"Shape of one book embedding (aggregated): {book_embeddings_aggregated[0].shape}")
print(f"Shape of one chapter embedding (aggregated): {chapter_embeddings_aggregated[0][0].shape}")
print(f"Shape of one paragraph embedding (aggregated): {paragraph_embeddings_aggregated[0][0][0].shape}")




#   save

# --- 1. Create the Master DataFrame ---
print("Step 1: Creating the Master DataFrame for sentences...")

# This process must EXACTLY mirror how you created `all_sentences_flat`
# We will iterate through the original data and tokenize it into sentences again.
records = []
for i, book in enumerate(tqdm(data, desc="Building DataFrame")):
    book_title = book['meta'].get('Original Title', f'book_{i}')
    book_author = book['meta'].get('Original Writer')
    year = book['meta'].get('Publication Year (Original)')
    for j, chapter in enumerate(book['content']):
        for k, paragraph in enumerate(chapter):
            # Tokenize the paragraph into sentences, just like before
            sentences = nltk.sent_tokenize(paragraph)
            
            if sentences:
                for s_idx, sentence in enumerate(sentences):
                    records.append({
                        "book_title": book_title,
                        "author": book_author,
                        "year": year,
                        "book_index": i,
                        "chapter_index": j,
                        "paragraph_index": k,
                        "sentence_index_in_para": s_idx, # Useful for context
                        "text": sentence
                    })

# Create the DataFrame from our list of records
df_sentences = pd.DataFrame(records)

# --- 2. Add Embeddings and Validate ---
print("\nStep 2: Adding embeddings and validating...")

# This is the most critical step. The length of the DataFrame must
# exactly match the number of embeddings you generated.
print(f"Number of rows in DataFrame: {len(df_sentences)}")
print(f"Number of embeddings:        {len(sentence_embeddings_flat)}")

assert len(df_sentences) == len(sentence_embeddings_flat), "Mismatch between DataFrame rows and number of embeddings!"

# Add the pre-computed embeddings as a new column.
# We convert the numpy array to a list so pandas can store it.
df_sentences['embedding'] = list(sentence_embeddings_flat)

print("\nDataFrame with sentence embeddings created successfully.")
print(df_sentences.head())
print(f"\nShape of the DataFrame: {df_sentences.shape}")
print(f"Data type of the 'embedding' column: {type(df_sentences['embedding'].iloc[0])}")


print("\nStep 3: Saving the DataFrame to Parquet format...")

# Define the output file path
output_path = 'sentence_embeddings_with_metadata.parquet'

# Save the DataFrame
df_sentences.to_parquet(output_path, index=False)

print(f"DataFrame successfully saved to {output_path}")
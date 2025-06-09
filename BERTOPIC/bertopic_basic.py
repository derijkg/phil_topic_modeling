import pandas as pd
from sentence_transformers import SentenceTransformer
from umap import UMAP
from sklearn.cluster import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import KeyBERTInspired # Example of alternative representation

# --- 0. Sample Data (Replace with your actual documents) ---
df = pd.read_csv(r'data\phil_nlp.csv')
df_works = df.groupby('title')['sentence_str'].agg(' '.join).reset_index()
docs = df_works['sentence_str']

# --- 1. Embedding Model ---
# Hyperparameters: Choice of model
# Popular choices: 'all-MiniLM-L6-v2', 'paraphrase-mpnet-base-v2', 'all-mpnet-base-v2'
# Consider multilingual models if needed: 'paraphrase-multilingual-MiniLM-L12-v2'
embedding_model_name = 'all-MiniLM-L6-v2'
sentence_model = SentenceTransformer(embedding_model_name)
# Pre-calculate embeddings if you plan to tune UMAP/HDBSCAN multiple times without re-embedding
# embeddings = sentence_model.encode(docs, show_progress_bar=True) # Do this once if tuning downstream

# --- 2. Dimensionality Reduction (UMAP) ---
# Key Hyperparameters:
# - n_neighbors: Controls local vs. global structure. Smaller = more local. (Default: 15)
# - n_components: Target dimensionality. (Default: 5)
# - min_dist: Controls how tightly UMAP packs points. (Default: 0.0)
# - metric: Distance metric. (Default: 'cosine')
umap_params = {
    'n_neighbors': 29,
    'n_components': 2,
    'min_dist': 0.16003130834582818,
    'metric': 'cosine',
    'random_state': 42 # For reproducibility
}
umap_model = UMAP(**umap_params)

# --- 3. Clustering (HDBSCAN) ---
# Key Hyperparameters:
# - min_cluster_size: Minimum number of documents to form a topic. (Default: 10, highly data-dependent)
# - min_samples: How conservative the clustering is. Higher = more points become noise/outliers. (Default: None, often set equal to min_cluster_size)
# - metric: Distance metric for clustering. (Default: 'euclidean')
# - cluster_selection_method: 'eom' (Excess of Mass) or 'leaf'. (Default: 'eom')
# - allow_single_cluster: (Default: False)
hdbscan_params = {
    'min_cluster_size': 2, # Adjust based on your dataset size and desired granularity
    'min_samples': 1, # Can be set to min_cluster_size or a bit lower
    'metric': 'euclidean',
    'cluster_selection_method': 'eom',
    'prediction_data': True # Important for calculating probabilities later if needed
}
hdbscan_model = HDBSCAN(**hdbscan_params)

# --- 4. Topic Representation (c-TF-IDF via CountVectorizer and ClassTfidfTransformer) ---
# Key Hyperparameters for CountVectorizer:
# - stop_words: Language for stop words (e.g., 'english') or a custom list.
# - ngram_range: (min_n, max_n) for n-grams. (e.g., (1, 1) for unigrams, (1, 2) for unigrams & bigrams)
# - min_df: Minimum document frequency for a word.
# - max_df: Maximum document frequency for a word.
vectorizer_params = {
    'stop_words': "english",
    'ngram_range': (1, 1), # Try (1, 2) for bigrams
    'min_df': 2,           # Word must appear in at least 2 documents
    # 'max_df': 0.95,      # Word must not appear in more than 95% of documents
}
vectorizer_model = CountVectorizer(**vectorizer_params)

# Key Hyperparameters for ClassTfidfTransformer:
# - reduce_frequent_words: If True, reduces importance of words appearing in many topics. (Default: False)
# - bm25_weighting: Use BM25 weighting instead of standard TF-IDF. (Default: False)
ctfidf_params = {
    'reduce_frequent_words': True,
    'bm25_weighting': False
}
ctfidf_model = ClassTfidfTransformer(**ctfidf_params)

# (Optional) Alternative Representation Models
# For example, using KeyBERT to refine topic representations:
# keybert_model = KeyBERTInspired()
# Or Maximal Marginal Relevance (MMR) for diversity:
# from bertopic.representation import MaximalMarginalRelevance
# mmr_model = MaximalMarginalRelevance(diversity=0.7)
# representation_model = {"KeyBERT": keybert_model, "MMR": mmr_model} # Can pass a dict for multiple representations

# --- 5. BERTopic Model Instantiation ---
# Key Hyperparameters for BERTopic itself:
# - nr_topics: 'auto' (HDBSCAN decides) or an integer (to reduce topics post-clustering) or None
# - top_n_words: Number of words to display per topic. (Default: 10)
# - calculate_probabilities: Whether to calculate document-topic probabilities (can be slow). (Default: False)
# - language: 'english' or 'multilingual' (if using multilingual sentence transformer)
# - verbose: True/False
topic_model = BERTopic(
    embedding_model=sentence_model,         # Pass the instantiated sentence model
    umap_model=umap_model,                  # Pass the instantiated UMAP model
    hdbscan_model=hdbscan_model,            # Pass the instantiated HDBSCAN model
    vectorizer_model=vectorizer_model,      # Pass the instantiated CountVectorizer
    ctfidf_model=ctfidf_model,              # Pass the instantiated c-TF-IDF transformer
    # representation_model=representation_model, # If using alternative representations
    language="english",
    top_n_words=10,
    nr_topics=None,  # Let HDBSCAN determine, or set to an int for reduction, or "auto"
    calculate_probabilities=True, # Set to True if you need probabilities
    verbose=True
)

# --- 6. Training the Model ---
# If you pre-calculated embeddings:
# topics, probabilities = topic_model.fit_transform(docs, embeddings=embeddings)
# If not, BERTopic will handle embedding internally:
topics, probabilities = topic_model.fit_transform(docs)

# --- 7. Inspecting Results ---
print("Topic Info:")
print(topic_model.get_topic_info())

print("\nTop words for topic 0:")
print(topic_model.get_topic(0))

# --- 8. (Optional) Topic Reduction / Further Tuning ---
# If you want a specific number of topics after initial clustering:
# desired_num_topics = 5
# topic_model.reduce_topics(docs, nr_topics=desired_num_topics)
# print(f"\nTopic Info after reducing to {desired_num_topics} topics:")
# print(topic_model.get_topic_info())

# If you want to update topic representations based on different parameters for CountVectorizer
# For example, to include bigrams after an initial run:
# new_vectorizer_params = {'stop_words': "english", 'ngram_range': (1, 2), 'min_df': 2}
# new_vectorizer_model = CountVectorizer(**new_vectorizer_params)
# topic_model.update_topics(docs, vectorizer_model=new_vectorizer_model)
# print("\nTopic Info after updating with bigrams:")
# print(topic_model.get_topic_info())


# --- How to Adapt Hyperparameters (Tuning Strategy) ---
# 1. Start with Defaults or Sensible Guesses:
#    - `min_cluster_size` (HDBSCAN): Highly dependent on dataset size. Try values like 5, 10, 20, 50.
#    - `n_neighbors` (UMAP): Default 15 is often good. Try 5-50.
#    - `ngram_range` (Vectorizer): Start with (1,1), then try (1,2).
#
# 2. Define an Evaluation Metric:
#    - Topic Coherence (e.g., NPMI, C_v). BERTopic doesn't have built-in coherence calculation as it's often corpus-specific.
#      You might need external libraries like Gensim or `tmtoolkit` and calculate it on the top N words of each topic.
#    - Number of topics vs. outlier count (topic -1).
#    - Qualitative assessment: Do the topics make sense?
#
# 3. Iterative Tuning:
#    - Change ONE hyperparameter (or a small related set) at a time.
#    - **Most impactful usually are:**
#        - `min_cluster_size` in `hdbscan_model` (controls number and granularity of topics)
#        - `n_neighbors` and `n_components` in `umap_model` (affects how clusters are formed)
#        - `ngram_range`, `min_df`, `max_df` in `vectorizer_model` (affects topic word quality)
#        - Choice of `embedding_model`
#
# 4. Example Loop for `min_cluster_size` (Conceptual - requires evaluation logic):
#    for mcs_val in [5, 10, 15, 20]:
#        temp_hdbscan_model = HDBSCAN(min_cluster_size=mcs_val, ...)
#        temp_topic_model = BERTopic(hdbscan_model=temp_hdbscan_model, ...)
#        topics, _ = temp_topic_model.fit_transform(docs)
#        # Evaluate (e.g., coherence, number of topics, qualitative)
#        # Store results and pick the best
#
# 5. Persistence:
#    topic_model.save("my_bertopic_model")
#    # loaded_model = BERTopic.load("my_bertopic_model")
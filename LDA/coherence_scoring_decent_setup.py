import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import spacy
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
import matplotlib.pyplot as plt
import numpy as np # For argmax

# --- 0. Setup spaCy Model (Load once) ---
try:
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
except OSError:
    print("Downloading 'en_core_web_sm' for spaCy...")
    print("If this fails, run: python -m spacy download en_core_web_sm")
    spacy.cli.download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])



# --- 1. Preprocessing and Helper Functions ---
DEFAULT_CHUNK_SIZE = 900_000
def lemmatize_spacy_chunked(text_data, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'], chunk_size=DEFAULT_CHUNK_SIZE):
    """
    Lemmatizes text using spaCy, processing in chunks if text is too long.
    Removes punctuation, spaces, and stopwords. Optionally filters by POS.
    """
    text_data_str = str(text_data).lower() # Ensure text is string and lowercased
    lemmas = []

    if len(text_data_str) <= chunk_size:
        doc = nlp(text_data_str)
        lemmas = [
            token.lemma_
            for token in doc
            if not token.is_punct
            and not token.is_space
            and not token.is_stop
            and (allowed_postags is None or token.pos_ in allowed_postags)
        ]
    else:
        # print(f"    Chunking document of length {len(text_data_str)} for lemmatization (chunk size: {chunk_size})...")
        for i in range(0, len(text_data_str), chunk_size):
            chunk_text = text_data_str[i : i + chunk_size]
            doc = nlp(chunk_text)
            chunk_lemmas = [
                token.lemma_
                for token in doc
                if not token.is_punct
                and not token.is_space
                and not token.is_stop
                and (allowed_postags is None or token.pos_ in allowed_postags)
            ]
            lemmas.extend(chunk_lemmas)
    return " ".join(lemmas)


def display_topics(model, feature_names, no_top_words):
    """Helper function to display topics nicely."""
    print("\n--- Top Words Per Topic ---")
    for topic_idx, topic_weights in enumerate(model.components_):
        top_keyword_locs = (-topic_weights).argsort()[:no_top_words]
        topic_str = f"Topic {topic_idx}: " + " ".join([feature_names[i] for i in top_keyword_locs])
        print(topic_str)
    print("-" * 30)

def tokenize_for_coherence_chunked(text_series, chunk_size=DEFAULT_CHUNK_SIZE):
    """
    Tokenizes and lemmatizes a series of texts for Gensim CoherenceModel, using chunking.
    """
    tokenized_texts = []
    for text_doc_original in text_series:
        text_doc_str = str(text_doc_original).lower() # Ensure string and lowercase
        current_doc_tokens = []
        if len(text_doc_str) <= chunk_size:
            doc = nlp(text_doc_str)
            current_doc_tokens = [
                token.lemma_ # Already lowercased from text_doc_str
                for token in doc
                if not token.is_punct and not token.is_space and not token.is_stop
            ]
        else:
            # print(f"    Chunking document of length {len(text_doc_str)} for coherence tokenization (chunk size: {chunk_size})...")
            for i in range(0, len(text_doc_str), chunk_size):
                chunk_text = text_doc_str[i : i + chunk_size]
                doc = nlp(chunk_text)
                chunk_tokens = [
                    token.lemma_
                    for token in doc
                    if not token.is_punct and not token.is_space and not token.is_stop
                ]
                current_doc_tokens.extend(chunk_tokens)
        tokenized_texts.append(current_doc_tokens)
    return tokenized_texts

def get_topics_from_sklearn_model(model, vocabulary, top_n=10):
    """Extracts top N words for each topic from an sklearn LDA model."""
    topics = []
    for topic_weights in model.components_:
        top_keyword_locs = (-topic_weights).argsort()[:top_n]
        topics.append([vocabulary[i] for i in top_keyword_locs])
    return topics

# --- 2. Main Topic Modeling Function ---

def perform_lda_topic_modeling(
    series_data,
    n_topics=10,
    use_lemmatization=True,
    lemmatizer_pos_tags=['NOUN', 'ADJ', 'VERB'],
    custom_stop_words=None,
    min_df_val=5,
    max_df_val=0.90,
    ngram_r=(1, 1),
    random_seed=42,
    no_top_words_display=10,
    count_vectorizer_stop_words='english'
    ):
    print(f"\n--- Starting LDA Topic Modeling for {n_topics} Topics ---")
    processed_series = series_data.copy()
    if use_lemmatization:
        print("Step 1: Lemmatizing text (with chunking if necessary)...")
        # MODIFIED HERE:
        processed_series = processed_series.apply(
            lambda x: lemmatize_spacy_chunked(x, allowed_postags=lemmatizer_pos_tags)
        )

    current_stop_words = []
    if count_vectorizer_stop_words == 'english':
        current_stop_words = list(CountVectorizer(stop_words='english').get_stop_words())
    if custom_stop_words:
        current_stop_words = list(set(current_stop_words).union(set(custom_stop_words)))
    if not current_stop_words:
        current_stop_words = None

    print(f"Step 2: Vectorizing with min_df={min_df_val}, max_df={max_df_val}, ngrams={ngram_r}...")
    vec = CountVectorizer(
        stop_words=current_stop_words,
        lowercase=True, # lemmatize_spacy_chunked already lowercases
        min_df=min_df_val,
        max_df=max_df_val,
        ngram_range=ngram_r
    )
    dtm = vec.fit_transform(processed_series)
    vocabulary = vec.get_feature_names_out()

    if dtm.shape[0] == 0 or dtm.shape[1] == 0:
        print("Warning: Document-Term Matrix is empty...")
        return None, None, None
    print(f"  Vocabulary size: {len(vocabulary)}")
    if len(vocabulary) < n_topics:
        print(f"Warning: Vocabulary size ({len(vocabulary)}) is less than n_topics ({n_topics}).")

    print(f"Step 3: Fitting LDA model with {n_topics} topics...")
    lda_model = LatentDirichletAllocation(
        n_components=n_topics, learning_method='online', random_state=random_seed, n_jobs=-1
    )
    document_topic_distributions_raw = lda_model.fit_transform(dtm)

    print("Step 4: Formatting results...")
    topic_names_list = [f'Topic {k}' for k in range(n_topics)]
    topic_word_distributions_df = pd.DataFrame(
        lda_model.components_, columns=vocabulary, index=topic_names_list
    )
    topic_word_distributions_df = topic_word_distributions_df.apply(lambda x: x / np.sum(x), axis=1)
    document_topic_distributions_df = pd.DataFrame(
        document_topic_distributions_raw, columns=topic_names_list, index=series_data.index
    )
    display_topics(lda_model, vocabulary, no_top_words_display)
    print("--- LDA Topic Modeling Complete ---")
    return topic_word_distributions_df, document_topic_distributions_df, lda_model

def display_topics(model, feature_names, no_top_words):
    """Helper function to display topics nicely."""
    print("\n--- Top Words Per Topic ---")
    for topic_idx, topic_weights in enumerate(model.components_):
        top_keyword_locs = (-topic_weights).argsort()[:no_top_words]
        topic_str = f"Topic {topic_idx}: " + " ".join([feature_names[i] for i in top_keyword_locs])
        print(topic_str)
    print("-" * 30)


# --- 3. Function to Tune Number of Topics using Coherence ---

def tune_lda_hyperparameters(
    series_data_for_tuning,
    topic_range,
    use_lemmatization_tuning=True,
    lemmatizer_pos_tags_tuning=['NOUN', 'ADJ', 'VERB'],
    custom_stop_words_tuning=None,
    min_df_tuning=5,
    max_df_tuning=0.90,
    ngram_r_tuning=(1,1),
    random_seed_tuning=42,
    coherence_metric='c_v',
    coherence_top_n_words=10,
    count_vectorizer_stop_words_tuning='english'
):
    print("\n--- Starting LDA Hyperparameter Tuning for Number of Topics ---")

    print("Step 1: Tokenizing texts for coherence calculation (with chunking if necessary)...")
    # MODIFIED HERE:
    tokenized_texts_for_coherence = tokenize_for_coherence_chunked(series_data_for_tuning)
    if not any(tokenized_texts_for_coherence):
        print("Error: Tokenization for coherence resulted in all empty documents.")
        return pd.DataFrame({'n_topics': [], f'coherence_{coherence_metric}': []})
    gensim_dictionary = Dictionary(tokenized_texts_for_coherence)

    processed_series_for_dtm = series_data_for_tuning.copy()
    if use_lemmatization_tuning:
        print("Step 2: Lemmatizing text for DTM creation (with chunking if necessary)...")
        # MODIFIED HERE:
        processed_series_for_dtm = processed_series_for_dtm.apply(
            lambda x: lemmatize_spacy_chunked(x, allowed_postags=lemmatizer_pos_tags_tuning)
        )

    current_stop_words_tuning = []
    if count_vectorizer_stop_words_tuning == 'english':
        current_stop_words_tuning = list(CountVectorizer(stop_words='english').get_stop_words())
    if custom_stop_words_tuning:
        current_stop_words_tuning = list(set(current_stop_words_tuning).union(set(custom_stop_words_tuning)))
    if not current_stop_words_tuning:
        current_stop_words_tuning = None

    print("Step 3: Creating Document-Term Matrix (once)...")
    vec_tuning = CountVectorizer(
        stop_words=current_stop_words_tuning, lowercase=True, min_df=min_df_tuning,
        max_df=max_df_tuning, ngram_range=ngram_r_tuning
    )
    dtm_tuning = vec_tuning.fit_transform(processed_series_for_dtm)
    vocabulary_tuning = vec_tuning.get_feature_names_out()

    if dtm_tuning.shape[0] == 0 or dtm_tuning.shape[1] == 0:
        print("Warning: DTM for tuning is empty. Aborting tuning.")
        return pd.DataFrame({'n_topics': [], f'coherence_{coherence_metric}': []})
    print(f"  DTM for tuning: {dtm_tuning.shape[0]} documents, {dtm_tuning.shape[1]} features.")

    coherence_values = []
    print(f"Step 4: Iterating through n_topics from {min(topic_range)} to {max(topic_range)}...")
    for n_topics_val in topic_range:
        print(f"  Training LDA with {n_topics_val} topics...")
        if len(vocabulary_tuning) < n_topics_val:
            print(f"    Skipping {n_topics_val} topics: vocabulary size ({len(vocabulary_tuning)}) is smaller.")
            coherence_values.append(np.nan)
            continue

        lda_tuning_model = LatentDirichletAllocation(
            n_components=n_topics_val, learning_method='online',
            random_state=random_seed_tuning, n_jobs=-1
        )
        lda_tuning_model.fit(dtm_tuning)
        topics_sklearn = get_topics_from_sklearn_model(lda_tuning_model, vocabulary_tuning, top_n=coherence_top_n_words)

        if not topics_sklearn or not any(topics_sklearn):
             print(f"    Warning: No topics extracted for n_topics={n_topics_val}. Skipping coherence calculation.")
             coherence_values.append(np.nan)
             continue
        try:
            cm = CoherenceModel(
                topics=topics_sklearn, texts=tokenized_texts_for_coherence,
                dictionary=gensim_dictionary, coherence=coherence_metric
            )
            coherence = cm.get_coherence()
        except Exception as e:
            print(f"    Error calculating coherence for {n_topics_val} topics: {e}")
            coherence = np.nan
        coherence_values.append(coherence)
        print(f"    Coherence ({coherence_metric}) for {n_topics_val} topics: {coherence:.4f}")

    results_df = pd.DataFrame({
        'n_topics': list(topic_range),
        f'coherence_{coherence_metric}': coherence_values
    }).dropna()

    if not results_df.empty:
        plt.figure(figsize=(10, 6))
        plt.plot(results_df['n_topics'], results_df[f'coherence_{coherence_metric}'], marker='o')
        plt.title(f'Topic Coherence ({coherence_metric}) vs. Number of Topics')
        plt.xlabel('Number of Topics'); plt.ylabel(f'Coherence Score ({coherence_metric})')
        plt.xticks(list(topic_range)); plt.grid(True); plt.show()
    else:
        print("No valid coherence scores to plot.")
    print("--- LDA Hyperparameter Tuning Complete ---")
    return results_df


if __name__ == '__main__':
    # Create Sample Data
    # To test chunking, make at least one document very long:
    df = pd.read_csv(r'data\phil_nlp.csv')
    df_works = df.groupby('title')['sentence_str'].agg(' '.join).reset_index()

    sample_series = df_works['sentence_str']
    sample_series.index = [f"Doc_{i}" for i in range(len(sample_series))]

    # Optional: If you want to see the chunking messages, uncomment the print statements
    # inside lemmatize_spacy_chunked and tokenize_for_coherence_chunked.
    
    print(f"Processing {len(sample_series)} documents.")
    longest_doc_len = sample_series.astype(str).map(len).max()
    print(f"Length of longest document: {longest_doc_len} characters.")
    if longest_doc_len > DEFAULT_CHUNK_SIZE:
        print(f"Chunking will be active for documents longer than {DEFAULT_CHUNK_SIZE} characters.")


    # --- Part 1: Tune Number of Topics ---
    print("### PART 1: TUNING NUMBER OF TOPICS ###")
    min_docs_for_word = 10
    topic_numbers_to_try = range(12, 26, 1) #current best 13

    coherence_results = tune_lda_hyperparameters(
        series_data_for_tuning=sample_series,
        topic_range=topic_numbers_to_try,
        use_lemmatization_tuning=True,
        lemmatizer_pos_tags_tuning=['NOUN', 'ADJ', 'VERB'],
        min_df_tuning=min_docs_for_word,
        max_df_tuning=0.95,
        random_seed_tuning=42,
        coherence_metric='c_v'
    )
    print("\nCoherence Score Results:"); print(coherence_results)

    optimal_n_topics = 13 # Default
    if not coherence_results.empty and f'coherence_c_v' in coherence_results.columns:
        valid_coherence = coherence_results.dropna(subset=[f'coherence_c_v'])
        if not valid_coherence.empty:
            optimal_n_topics = int(valid_coherence.loc[valid_coherence[f'coherence_c_v'].idxmax(), 'n_topics'])
            print(f"\nOptimal number of topics based on C_v coherence: {optimal_n_topics}")
        else: print("\nCould not determine optimal number of topics. Using default.")
    else: print(f"\nCoherence results empty or 'coherence_c_v' missing. Using default: {optimal_n_topics}")

    # --- Part 2: Run LDA with Chosen Number of Topics ---
    print(f"\n\n### PART 2: RUNNING LDA WITH {optimal_n_topics} TOPICS ###")
    my_custom_stops = None # TUNE after corpus analysis, ['abc', 'def']

    topic_word_dist_df, doc_topic_dist_df, final_lda_model = perform_lda_topic_modeling(
        series_data=sample_series,
        n_topics=optimal_n_topics,
        use_lemmatization=True,
        lemmatizer_pos_tags=['NOUN', 'ADJ', 'VERB'],
        custom_stop_words=my_custom_stops,
        min_df_val=min_docs_for_word,
        max_df_val=0.95,
        random_seed=42
    )

    if topic_word_dist_df is not None:
        print("\n--- Topic-Word Distributions (Sample) ---")
        print(topic_word_dist_df.head().iloc[:, :5])
        print("\n--- Document-Topic Distributions (Sample) ---")
        print(doc_topic_dist_df.head())
        dominant_topic = doc_topic_dist_df.idxmax(axis=1)
        print("\n--- Dominant Topic per Document (Sample) ---")
        print(dominant_topic.head())

    # For further visualization, you can use pyLDAvis:
    # import pyLDAvis
    # import pyLDAvis.sklearn
    #
    # if final_lda_model and dtm_tuning is not None: # Assuming dtm_tuning was used for the final model run
    #     # To use pyLDAvis, you need the LDA model, DTM, and CountVectorizer
    #     # Re-run vectorizer and LDA if you changed settings between tuning and final run
    #     # For simplicity, let's assume parameters were consistent or re-run:
    #     # vec_final = CountVectorizer(...)
    #     # dtm_final = vec_final.fit_transform(processed_series) # processed_series from perform_lda_topic_modeling
    #     # lda_final = LatentDirichletAllocation(n_components=optimal_n_topics, ...).fit(dtm_final)
    #
    #     # If the DTM used for the final model is available (e.g. dtm from perform_lda_topic_modeling),
    #     # and the vectorizer 'vec' from that function.
    #
    #     # To get the DTM and vectorizer from the perform_lda_topic_modeling function, you'd need to return them.
    #     # For now, let's assume you'd re-create them if necessary for pyLDAvis
    #     # vis_data = pyLDAvis.sklearn.prepare(final_lda_model, dtm_tuning, vec_tuning, mds='tsne') # use dtm & vec from final model run
    #     # pyLDAvis.display(vis_data)
    #     # pyLDAvis.save_html(vis_data, 'lda_visualization.html')
    #     # print("\nSaved pyLDAvis visualization to lda_visualization.html")
    # else:
    #     print("\nSkipping pyLDAvis visualization as model or DTM is not available.")
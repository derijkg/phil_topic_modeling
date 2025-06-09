import pandas as pd
from sklearn.feature_extraction import text # TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import spacy # For lemmatization
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary

# Load spacy model (once)
try:
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
except OSError:
    print("Downloading en_core_web_sm for spaCy...")
    spacy.cli.download('en_core_web_sm')
    nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

def lemmatize_spacy(text_data):
    """Lemmatizes text using spaCy, removing punctuation and spaces."""
    doc = nlp(text_data)
    return " ".join([token.lemma_ for token in doc if not token.is_punct and not token.is_space and not token.is_stop])


def display_topics(model, feature_names, no_top_words):
    """Helper function to display topics nicely."""
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic {topic_idx}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[:-no_top_words - 1:-1]]))
    print("\n")


def tm_class_LDA(series,
                 n_topics=10, # Start with fewer, then tune
                 min_df_val=5,
                 max_df_val=0.90,
                 ngram_r=(1, 1),
                 use_lemmatization=True,
                 custom_stop_words=None,
                 random_seed=42,
                 no_top_words_display=15):

    # 1. Preprocessing
    processed_series = series
    if use_lemmatization:
        print("Lemmatizing text...")
        processed_series = series.apply(lemmatize_spacy)
        # Note: spaCy's default lemmatizer might remove its own stop words.
        # If you want to control stop words strictly with CountVectorizer,
        # you might adjust lemmatize_spacy or apply stop word removal after lemmatization.

    # 2. Count Vectorizer
    stop_list = list(text.ENGLISH_STOP_WORDS) # Ensure it's a list
    if custom_stop_words:
        stop_list = list(set(stop_list).union(set(custom_stop_words))) # Combine and ensure uniqueness

    print(f"Vectorizing with min_df={min_df_val}, max_df={max_df_val}, ngrams={ngram_r}...")
    vec = text.CountVectorizer(
        stop_words=stop_list,
        lowercase=True, # Lemmatization usually handles case, but good to keep
        min_df=min_df_val,
        max_df=max_df_val,
        ngram_range=ngram_r
    )
    dtm = vec.fit_transform(processed_series)
    vocabulary = vec.get_feature_names_out()

    if dtm.shape[1] == 0:
        print("Warning: Vocabulary is empty after vectorization. Check your min_df/max_df settings and data.")
        return None, None

    print(f"Vocabulary size: {len(vocabulary)}")

    # 3. LDA Model
    print(f"Fitting LDA model with {n_topics} topics...")
    model = LatentDirichletAllocation(
        n_components=n_topics,
        learning_method='online', # 'batch' can be better for smaller datasets
        random_state=random_seed,
        # You might want to tune these:
        # learning_decay=0.7,
        # doc_topic_prior=None, # Defaults to 1/n_components
        # topic_word_prior=None, # Defaults to 1/n_components
        # max_iter=10 (if using 'batch')
    )

    document_topic_distributions_raw = model.fit_transform(dtm)

    # 4. Results Formatting
    topic_names = [f'Topic {k}' for k in range(n_topics)]
    topic_word_distributions = pd.DataFrame(
        model.components_, columns=vocabulary, index=topic_names)

    # Normalize topic_word_distributions to sum to 1 if you want probabilities
    # topic_word_distributions = topic_word_distributions.apply(lambda x: x / x.sum(), axis=1)

    document_topic_distributions = pd.DataFrame(
        document_topic_distributions_raw, columns=topic_names, index=series.index) # Use original series index

    print("\nTop words per topic:")
    display_topics(model, vocabulary, no_top_words_display)

    # Example: Print top words for a specific topic (e.g., Topic 0 if n_topics > 0)
    if n_topics > 0 and 'Topic 0' in topic_word_distributions.index:
         print(f"\nDetailed view of Topic 0:")
         print(topic_word_distributions.loc['Topic 0'].sort_values(ascending=False).head(18))


    # Consider adding perplexity / coherence calculation here
    # perplexity = model.perplexity(dtm)
    # print(f"Model Perplexity: {perplexity}")
    # For coherence, you'd typically use gensim or octis

    return topic_word_distributions, document_topic_distributions

# --- Example Usage ---
if __name__ == '__main__':
    df = pd.read_csv('data\phil_nlp.csv')
    df_works = df.groupby('title')['sentence_str'].agg(' '.join).reset_index()

    sample_series = df_works['sentence_str']
    r_seed = 42 # Your random seed

    custom_stops = ["fox", "dog"] # Example custom stop words

    topic_word_dist, doc_topic_dist = tm_class_LDA(
        sample_series,
        n_topics=3,        # Try different numbers
        min_df_val=1,      # For small corpus, min_df might need to be 1
        max_df_val=0.95,
        ngram_r=(1,2),     # Try with and without bigrams
        use_lemmatization=True,
        custom_stop_words=custom_stops,
        random_seed=r_seed,
        no_top_words_display=5
    )

    if topic_word_dist is not None:
        print("\nTopic Word Distributions (Head):")
        print(topic_word_dist.head())
        print("\nDocument Topic Distributions (Head):")
        print(doc_topic_dist.head())
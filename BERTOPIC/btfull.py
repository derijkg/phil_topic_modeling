import optuna
import pandas as pd
import numpy as np # For subsampling and float32
from sentence_transformers import SentenceTransformer
from umap import UMAP
from sklearn.cluster import HDBSCAN # Using sklearn's version
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import silhouette_score
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance

from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel
import nltk
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')
from nltk.corpus import stopwords
import logging
from functools import partial
import random # For subsampling

import json
# --- 0. Global Settings ---
optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2' # Consider 'all-MiniLM-L6-v2' for speed/less memory if quality is acceptable
TOP_N_WORDS_FOR_COHERENCE = 10
FINAL_MODEL_TOP_N_WORDS = 10
STOP_WORDS_GNSM = stopwords.words('english')

logger.info("--- Starting Data Setup ---")
try:
    # Corrected path for example, adjust to your actual path
    with open(r'..\data_processing\data.json') as f:
        data = json.load(f)
    end_data = []
    for item in data:
        end_data.extend(item.get('paragraphs'))
except FileNotFoundError:
    logger.error(r"Error: phil_nlp.csv not found. Please check the path.")
    exit()


DOCS_FULL = end_data

storage_name = "sqlite:///bertopic_optuna_sentence_tuned.db"
study_name = "bertopic_sentence_tuned_study"
study = optuna.create_study(study_name=study_name, storage=storage_name,
                            direction="maximize", load_if_exists=True)

sentence_model_local = SentenceTransformer(EMBEDDING_MODEL_NAME)
MAX_DOCS_FOR_OPTUNA_TRIALS = 100_000 # Example: 100,000 sentences


if study.best_trial:
    logger.info("Best trial overall in study (on subset):")
    logger.info(f"  Value (Combined Score): {study.best_trial.value:.4f}")
    best_params = study.best_trial.params
    logger.info("  Best Params (from Optuna on subset): ")
    for key, value in best_params.items(): logger.info(f"    {key}: {value}")

    logger.info("\n--- Training Final BERTopic Model with Best Parameters ON FULL DATASET ---")
    # Re-encode full dataset for final model
    logger.info("Encoding FULL dataset for final model...")
    EMBEDDINGS_FULL = sentence_model_local.encode(DOCS_FULL, show_progress_bar=True).astype(np.float32)

    final_umap_metric = best_params.get('umap_metric', 'cosine')
    final_umap_model = UMAP(n_neighbors=best_params['umap_n_neighbors'],
                            n_components=best_params['umap_n_components'],
                            min_dist=best_params['umap_min_dist'],
                            metric=final_umap_metric, random_state=42,
                            low_memory=True, verbose=True) # low_memory=True and verbose for final model on full data

    final_hdbscan_min_cluster_size = best_params['hdbscan_min_cluster_size']
    # Recalculate min_samples based on the logic in objective function
    if final_hdbscan_min_cluster_size <= 5:
        final_hdbscan_min_samples = best_params.get('hdbscan_min_samples', 1) # Fallback if not tuned that way
    else:
        final_hdbscan_min_samples = best_params.get('hdbscan_min_samples', max(2, final_hdbscan_min_cluster_size // 2))


    final_hdbscan_metric = best_params.get('hdbscan_metric', 'euclidean')
    final_hdbscan_model = HDBSCAN(min_cluster_size=final_hdbscan_min_cluster_size,
                                    min_samples=final_hdbscan_min_samples, # Use recalculated
                                    metric=final_hdbscan_metric,
                                    cluster_selection_method=best_params['hdbscan_cluster_selection_method'],
                                    # prediction_data=True, # Not a param for sklearn.cluster.HDBSCAN
                                    # gen_min_span_tree=True # Not a param for sklearn.cluster.HDBSCAN
                                    ) # Use multiple cores for final model if possible


    final_vectorizer_model = CountVectorizer(stop_words="english",
                                                ngram_range=(1, best_params['ngram_max']),
                                                min_df=best_params['vectorizer_min_df'],
                                                max_df=best_params.get('vectorizer_max_df', 0.95))

    final_ctfidf_model_instance = ClassTfidfTransformer(
        reduce_frequent_words=best_params.get('ctfidf_reduce_frequent_words', True),
        bm25_weighting=best_params.get('ctfidf_bm25_weighting', False))

    final_nr_topics_strategy = best_params.get('nr_topics_strategy', 'auto')
    final_nr_topics_suggestion = None
    if final_nr_topics_strategy == 'target_range':
        final_nr_topics_suggestion = best_params.get('nr_topics_target', None)
        # Potentially re-scale nr_topics_target if full dataset is much larger than subset
        # This is heuristic:
        if len(DOCS_FULL) > 2 * MAX_DOCS_FOR_OPTUNA_TRIALS and final_nr_topics_suggestion is not None:
            scale_factor = len(DOCS_FULL) / MAX_DOCS_FOR_OPTUNA_TRIALS
            final_nr_topics_suggestion = int(final_nr_topics_suggestion * (scale_factor**0.5)) # Non-linear scaling
            final_nr_topics_suggestion = max(5, min(100, final_nr_topics_suggestion)) # Bounds
            logger.info(f"Adjusted nr_topics_target for full dataset to: {final_nr_topics_suggestion}")


    final_representation_choice = best_params.get('representation_model_type', 'default_ctfidf')
    final_current_representation_model = None
    if final_representation_choice == 'keybert':
        final_current_representation_model = KeyBERTInspired()
    elif final_representation_choice == 'mmr':
        final_mmr_diversity = best_params.get('mmr_diversity', 0.5)
        final_current_representation_model = MaximalMarginalRelevance(diversity=final_mmr_diversity)

    final_topic_model = BERTopic(
        embedding_model=None, umap_model=final_umap_model, hdbscan_model=final_hdbscan_model,
        vectorizer_model=final_vectorizer_model, ctfidf_model=final_ctfidf_model_instance,
        representation_model=final_current_representation_model,
        language="english", top_n_words=FINAL_MODEL_TOP_N_WORDS,
        calculate_probabilities=True, verbose=True, nr_topics=final_nr_topics_suggestion)

    logger.info("Fitting the final BERTopic model ON FULL DATASET...")
    try:
        final_topics, final_probabilities = final_topic_model.fit_transform(DOCS_FULL, embeddings=EMBEDDINGS_FULL)

        logger.info("\n--- Final Model Results (on Full Dataset) ---")
        final_topic_info = final_topic_model.get_topic_info()
        print("Topic Info (Sample):")
        print(final_topic_info.head(min(15, len(final_topic_info))))

        logger.info(f"\nKeywords for each topic (Top {min(10, len(final_topic_info)-1 if -1 in final_topic_info.Topic.values else len(final_topic_info) )} topics, Top 7 words):")
        topics_to_show_ids = [tid for tid in final_topic_info.sort_values(by="Count", ascending=False).Topic if tid != -1][:10]

        for topic_id in topics_to_show_ids:
            topic_name = final_topic_info[final_topic_info.Topic == topic_id]['Name'].values[0]
            topic_count = final_topic_info[final_topic_info.Topic == topic_id]['Count'].values[0]
            print(f"\nTopic {topic_id} ({topic_name}): Count={topic_count}")
            topic_words = final_topic_model.get_topic(topic_id)
            if topic_words: print(topic_words[:7])
            else: print("  (No words found for this topic)")

        final_model_path = "final_bertopic_model_optimized_full_data"
        # When saving, if you used a specific sentence_model instance for embeddings,
        # you can pass its name or the instance itself.
        # Here, we pass the name used for loading, assuming it can be loaded by SentenceTransformer
        final_topic_model.save(final_model_path, serialization="pickle", save_ctfidf=True, save_embedding_model=EMBEDDING_MODEL_NAME)
        logger.info(f"\nFinal model saved to '{final_model_path}'")
        # For loading: BERTopic.load(final_model_path, embedding_model=EMBEDDING_MODEL_NAME) # or the actual model instance if not a standard name

    except MemoryError as e:
        logger.error(f"MemoryError during FINAL model fitting on FULL dataset: {e}. Try reducing sentence count further or using a machine with more RAM.", exc_info=True)
    except Exception as e:
        logger.error(f"Unexpected error during FINAL model fitting: {e}", exc_info=True)
else:
    logger.info("No successful trials completed in the study, cannot train final model.")
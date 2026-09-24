"""TF-IDF features for email text."""

from sklearn.feature_extraction.text import TfidfVectorizer

from src.preprocess import normalize_email_text


def build_vectorizer() -> TfidfVectorizer:
    """Word unigrams and bigrams over normalized email text."""
    return TfidfVectorizer(
        preprocessor=normalize_email_text,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        max_features=30000,
        sublinear_tf=True,
    )

import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from gensim.utils import simple_preprocess
import joblib

# Define stopwords
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "else", "for", "on", "in", "at",
    "of", "to", "from", "by", "with", "about", "as", "into", "like", "through", "after",
    "over", "between", "out", "against", "during", "without", "before", "under", "around",
    "among", "is", "am", "are", "was", "were", "be", "being", "been", "this", "that", "these",
    "those", "it", "its", "i", "you", "he", "she", "they", "we", "him", "her", "them", "my",
    "your", "our", "their", "so", "not", "no", "yes", "can", "could", "should", "would",
    "will", "just", "than", "too", "very", "also", "such", "has", "have", "had", "one",
    "two", "three", "more", "most", "many", "much", "any", "some", "each", "other",
    "another", "within", "across", "against", "while", "where", "when", "which",
    "whose", "what", "who", "whom", "why", "how", "into", "onto", "ever", "said",
    "made", "make", "makes", "doing", "done", "seen", "used", "using", "based",
    "according", "including", "include", "includes", "may", "might"
}

# Load keyphrase classification model
try:
    keyphrase_model = joblib.load("keyphrase_model.joblib")
except Exception:
    keyphrase_model = None


# Text Preprocessing
def clean_tokens(text):
    tokens = simple_preprocess(text)
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) >= 4]
    return tokens


# Rule-Based Keyword Extraction
def rule_based_keywords(text, top_n=20):
    tokens = clean_tokens(text)
    if not tokens:
        return []
    freq = Counter(tokens)
    return [w for w, _ in freq.most_common(top_n)]


# TF-IDF Based Keyword Extraction
def ml_keywords(text, top_n=20):
    tokens = clean_tokens(text)
    if not tokens:
        return []
    doc = " ".join(tokens)
    docs = [doc]
    vectorizer = TfidfVectorizer(ngram_range=(1, 1), stop_words="english")
    try:
        tfidf_matrix = vectorizer.fit_transform(docs)
    except ValueError:
        return []
    scores = tfidf_matrix.toarray()[0]
    feature_names = vectorizer.get_feature_names_out()
    scored = sorted(zip(feature_names, scores), key=lambda x: x[1], reverse=True)
    return [term for term, _ in scored[:top_n]]


# Combine Rule-Based + ML
def hybrid_keywords(text, top_n=20):
    rule = rule_based_keywords(text, top_n)
    ml = ml_keywords(text, top_n)
    combined = []
    seen = set()

    for w in rule + ml:
        if w not in seen:
            combined.append(w)
            seen.add(w)

    return combined[:top_n]


# Generate bigram/trigram candidates using CountVectorizer
def generate_candidate_phrases(text, ngram_range=(2, 3), top_n=50):
    tokens = simple_preprocess(text)
    clean_text = " ".join([t for t in tokens if t not in STOPWORDS])
    vectorizer = CountVectorizer(ngram_range=ngram_range).fit([clean_text])
    candidates = vectorizer.get_feature_names_out()
    return list(candidates)


# Remove redundant overlaps
def reduce_redundant_phrases(phrases):
    unique = list(dict.fromkeys(phrases))
    sorted_p = sorted(unique, key=lambda p: len(p.split()), reverse=True)
    kept = []
    for p in sorted_p:
        if not any(p in k for k in kept):
            kept.append(p)
    return kept


# Predict Top Keyphrases from model
def model_keyphrases(text, top_n=20, min_prob=0.55):
    if keyphrase_model is None:
        return []

    candidates = generate_candidate_phrases(text)
    if not candidates:
        return []

    try:
        probs = keyphrase_model.predict_proba(candidates)
    except Exception:
        return []

    scored = sorted(zip(candidates, probs[:, 1]), key=lambda x: x[1], reverse=True)

    filtered = []
    for phrase, prob in scored:
        if prob < min_prob:
            continue
        filtered.append(phrase)
        if len(filtered) >= top_n * 2:
            break

    if not filtered:
        filtered = [p for p, _ in scored[:top_n * 2]]

    filtered = reduce_redundant_phrases(filtered)
    return filtered[:top_n]


# Public Keyphrase Extractor
def hybrid_keyphrases(text, top_n=20):
    return model_keyphrases(text, top_n)

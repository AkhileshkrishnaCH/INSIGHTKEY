import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.utils import simple_preprocess
import joblib
import spacy

# Load English model for POS filtering
nlp = spacy.load("en_core_web_sm")

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

try:
    keyphrase_model = joblib.load("keyphrase_model.joblib")
except Exception:
    keyphrase_model = None


def clean_tokens(text):
    tokens = simple_preprocess(text)
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) >= 4]
    return tokens


def rule_based_keywords(text, top_n=20):
    tokens = clean_tokens(text)
    if not tokens:
        return []
    freq = Counter(tokens)
    return [w for w, _ in freq.most_common(top_n)]


def ml_keywords(text, top_n=20):
    tokens = clean_tokens(text)
    if not tokens:
        return []
    doc = " ".join(tokens)
    vectorizer = TfidfVectorizer(ngram_range=(1, 1), stop_words="english")
    try:
        tfidf_matrix = vectorizer.fit_transform([doc])
    except ValueError:
        return []
    scores = tfidf_matrix.toarray()[0]
    feature_names = vectorizer.get_feature_names_out()
    scored = sorted(zip(feature_names, scores), key=lambda x: x[1], reverse=True)
    return [term for term, _ in scored[:top_n]]


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


def generate_ngram_candidates(text, ngram_range=(2, 3)):
    doc = nlp(text.lower())
    phrases = []
    for chunk in doc.noun_chunks:
        phrase = chunk.text.strip()
        if len(phrase.split()) >= 2 and not any(w in STOPWORDS for w in phrase.split()):
            phrases.append(phrase)
    # Additional bigrams/trigrams using TF-IDF
    clean_text = " ".join([t.text for t in doc if not t.is_stop and not t.is_punct])
    vectorizer = TfidfVectorizer(ngram_range=ngram_range, stop_words="english").fit([clean_text])
    tfidf_phrases = vectorizer.get_feature_names_out()
    phrases.extend(tfidf_phrases)
    # Remove duplicates
    unique_phrases = list(dict.fromkeys(phrases))
    return unique_phrases


def reduce_redundant_phrases(phrases):
    sorted_p = sorted(phrases, key=lambda p: len(p.split()), reverse=True)
    kept = []
    for p in sorted_p:
        if not any(p in k for k in kept):
            kept.append(p)
    return kept


def model_keyphrases(text, top_n=20, min_prob=0.5):
    if keyphrase_model is None:
        return []

    candidates = generate_ngram_candidates(text)
    if not candidates:
        return []

    try:
        probs = keyphrase_model.predict_proba(candidates)
    except Exception:
        return []

    scored = sorted(zip(candidates, probs[:, 1]), key=lambda x: x[1], reverse=True)
    filtered = [p for p, prob in scored if prob >= min_prob]
    filtered = reduce_redundant_phrases(filtered)
    return filtered[:top_n]


def hybrid_keyphrases(text, top_n=20):
    return model_keyphrases(text, top_n)

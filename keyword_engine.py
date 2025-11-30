import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.utils import simple_preprocess
import joblib

STOPWORDS = {
"the","a","an","and","or","but","if","then","else","for","on","in","at",
"of","to","from","by","with","about","as","into","like","through","after",
"over","between","out","against","during","without","before","under","around",
"among","is","am","are","was","were","be","being","been","this","that","these",
"those","it","its","i","you","he","she","they","we","him","her","them","my",
"your","our","their","so","not","no","yes","can","could","should","would",
"will","just","than","too","very","also","such","has","have","had","one",
"two","three","more","most","many","much","any","some","each","other",
"another","within","across","against","while","where","when","which",
"whose","what","who","whom","why","how","into","onto","ever","ever",
"said","made","make","makes","doing","done","seen","used","using",
"based","according","including","include","includes","may","might"
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
    return [w for w,_ in freq.most_common(top_n)]

def ml_keywords(text, top_n=20):
    tokens = clean_tokens(text)
    if not tokens:
        return []
    doc = " ".join(tokens)
    docs = [doc]
    vectorizer = TfidfVectorizer(ngram_range=(1,1),stop_words="english")
    try:
        tfidf_matrix = vectorizer.fit_transform(docs)
    except ValueError:
        return []
    scores = tfidf_matrix.toarray()[0]
    feature_names = vectorizer.get_feature_names_out()
    scored = sorted(zip(feature_names,scores),key=lambda x:x[1],reverse=True)
    return [term for term,score in scored[:top_n]]

def hybrid_keywords(text, top_n=20):
    rule = rule_based_keywords(text,top_n)
    ml = ml_keywords(text,top_n)
    combined = []
    seen = set()
    for w in rule:
        if w in ml and w not in seen:
            combined.append(w)
            seen.add(w)
    for w in rule:
        if w not in seen:
            combined.append(w)
            seen.add(w)
    for w in ml:
        if w not in seen:
            combined.append(w)
            seen.add(w)
    return combined[:top_n]

def generate_candidate_phrases(text):
    tokens = simple_preprocess(text)
    tokens = [t for t in tokens if t not in STOPWORDS]
    phrases = []
    for i in range(len(tokens)-1):
        bigram = tokens[i] + " " + tokens[i+1]
        phrases.append(bigram)
    for i in range(len(tokens)-2):
        trigram = tokens[i] + " " + tokens[i+1] + " " + tokens[i+2]
        phrases.append(trigram)
    cleaned = []
    for p in phrases:
        p2 = re.sub(r"\s+"," ",p).strip().lower()
        if len(p2.split())<2:
            continue
        cleaned.append(p2)
    unique = list(dict.fromkeys(cleaned))
    return unique

def reduce_redundant_phrases(phrases):
    unique = list(dict.fromkeys(phrases))
    sorted_p = sorted(unique,key=lambda p:len(p.split()),reverse=True)
    kept = []
    for p in sorted_p:
        if not any(p in k for k in kept):
            kept.append(p)
    return kept

def model_keyphrases(text,top_n=20,min_prob=0.55):
    if keyphrase_model is None:
        return []
    candidates = generate_candidate_phrases(text)
    if not candidates:
        return []
    try:
        probs = keyphrase_model.predict_proba(candidates)
    except Exception:
        return []
    scored = list(zip(candidates,probs[:,1]))
    scored.sort(key=lambda x:x[1],reverse=True)
    filtered = []
    for phrase,prob in scored:
        if prob<min_prob:
            continue
        filtered.append(phrase)
        if len(filtered)>=top_n*2:
            break
    if not filtered:
        filtered = [p for p,_ in scored[:top_n*2]]
    filtered = reduce_redundant_phrases(filtered)
    return filtered[:top_n]

def hybrid_keyphrases(text,top_n=20):
    return model_keyphrases(text,top_n)

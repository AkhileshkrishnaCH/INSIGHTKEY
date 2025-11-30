import joblib

model = joblib.load("keyphrase_model.joblib")

test_phrases = [
"offshore bank accounts",
"click here to continue",
"confidential witness statement",
"privacy policy",
"illegal campaign donations",
"thank you for reading",
"forensic accounting review",
"homepage banner text",
"money laundering scheme",
"sample demo text"
]

probs = model.predict_proba(test_phrases)
preds = model.predict(test_phrases)

for phrase, pred, prob in zip(test_phrases, preds, probs[:,1]):
    print("Phrase:", phrase)
    print("Predicted label:", pred, "(1=keyphrase, 0=not keyphrase)")
    print("Keyphrase probability:", round(prob,4))
    print("-"*40)

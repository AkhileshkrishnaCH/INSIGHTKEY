# INSIGHTKEY
Keyword and Key-phrase Extractor using Multiple Machine learning models in form of a 3 page mini website
INSIGHTKEY – Keyword & Keyphrase Extraction Web Application
📌 Project Overview
INSIGHTKEY is a web-based Natural Language Processing (NLP) application designed to extract keywords and keyphrases from long-form textual documents such as investigative articles, reports, and research content.

The system uses a hybrid approach that combines:

Rule-based text processing

Machine Learning techniques (TF-IDF, Logistic Regression)



The application provides a clean web interface with user authentication, allowing users to securely log in, submit text, and view extracted keywords and keyphrases.

🎯 Objectives
Extract meaningful keywords and keyphrases from large textual inputs

Reduce noise from common or irrelevant terms

Demonstrate practical use of Machine Learning in NLP

Build a full-stack application integrating ML models with a web interface

Store user data securely using a database

🧠 Machine Learning Approach
1. TF-IDF (Term Frequency–Inverse Document Frequency)
Used for statistical keyword and keyphrase weighting

Unsupervised learning technique

Dynamically fitted on input text or training data

2. Logistic Regression Classifier
Used for document-level relevance classification

Trained using TF-IDF feature vectors

Helps differentiate meaningful investigative content

3. spaCy Pre-trained NLP Model
Used for noun-phrase extraction

Performs linguistic analysis such as:

Part-of-Speech tagging

Dependency parsing

Phrase chunking

4. Hybrid Model Design
Combines:

Rule-based filtering

TF-IDF statistical scores

ML-based phrase extraction

Removes duplicates and redundant phrases

Produces concise and meaningful output

🏗️ System Architecture
User → Web Interface → Flask Backend
     → Keyword Engine (Hybrid NLP Model)
     → TF-IDF / Logistic Model
     → Results Display
     → SQLite Database (User Data)
📂 Project Structure
INSIGHTKEY/
│
├── static/                     # CSS & JavaScript files
├── templates/                  # HTML templates
├── venv/                       # Python virtual environment
├── app.py                      # Main Flask application
├── keyword_engine.py           # Hybrid keyword & keyphrase extraction logic
├── train_keyphrase_model.py    # ML training script
├── test_keyphrase_model.py     # Model testing script
├── keyphrase_model.joblib      # Trained ML model
├── keyphrase_training_large.csv# Training dataset
├── users.db                    # SQLite database for users
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
🔐 Authentication System
User Sign Up and Login

Passwords stored securely using hashing

Session-based authentication

User data stored in SQLite database

🧪 Model Training Process
TF-IDF vectorizer is trained on a labeled dataset

Logistic Regression classifier is trained using TF-IDF features

Model is saved using joblib for reuse

spaCy model is pre-trained and loaded during runtime

No manual neural network training is required.

🛠️ Technologies Used
Backend: Python, Flask

Frontend: HTML, CSS, JavaScript

Machine Learning: scikit-learn

NLP: spaCy, TF-IDF

Database: SQLite

Model Storage: joblib

🚀 How to Run the Project
1. Clone the repository
git clone <repository-url>
cd INSIGHTKEY
2. Activate virtual environment
venv\Scripts\activate
3. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm
4. Run the application
python app.py
5. Open in browser
http://127.0.0.1:5000

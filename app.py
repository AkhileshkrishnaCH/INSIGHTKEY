from flask import Flask, render_template, request, redirect, url_for, session, jsonify
from dotenv import load_dotenv
from keyword_engine import hybrid_keywords, hybrid_keyphrases
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import os

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key")

DB_PATH = os.path.join(os.path.dirname(__file__), "users.db")

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()

init_db()

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "").strip()
        if not username or not email or not password:
            return render_template("signup.html", error="All fields are required.")
        if len(password) < 6:
            return render_template("signup.html", error="Password must be at least 6 characters.")
        conn = get_db()
        cur = conn.execute("SELECT id FROM users WHERE username = ? OR email = ?", (username, email))
        existing = cur.fetchone()
        if existing:
            conn.close()
            return render_template("signup.html", error="Username or email already exists.")
        password_hash = generate_password_hash(password)
        conn.execute(
            "INSERT INTO users (username, email, password_hash) VALUES (?, ?, ?)",
            (username, email, password_hash),
        )
        conn.commit()
        conn.close()
        return redirect(url_for("login"))
    return render_template("signup.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "").strip()
        if not email or not password:
            return render_template("login.html", error="Email and password are required.")
        conn = get_db()
        cur = conn.execute("SELECT * FROM users WHERE email = ?", (email,))
        user = cur.fetchone()
        conn.close()
        if not user:
            return render_template("login.html", error="Invalid credentials.")
        if not check_password_hash(user["password_hash"], password):
            return render_template("login.html", error="Invalid credentials.")
        session["username"] = user["username"]
        return redirect(url_for("dashboard"))
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.pop("username", None)
    return redirect(url_for("home"))

@app.route("/dashboard")
def dashboard():
    if "username" not in session:
        return redirect(url_for("login"))
    return render_template("dashboard.html", username=session["username"])

@app.route("/api/extract", methods=["POST"])
def extract():
    if "username" not in session:
        return jsonify({"error": "Unauthorized"}), 401
    data = request.get_json()
    text = data.get("text", "")
    if not text.strip():
        return jsonify({"error": "No text provided"}), 400
    keywords = hybrid_keywords(text)
    keyphrases = hybrid_keyphrases(text)
    return jsonify({"keywords": keywords, "keyphrases": keyphrases})

if __name__ == "__main__":
    app.run(debug=True)

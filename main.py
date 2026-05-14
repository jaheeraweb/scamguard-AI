# ============================================================
# FLASK APP - SCAM DETECTION + LOGIN + SQLITE + EMAIL + GEMINI CHATBOT
# ============================================================

from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import pandas as pd
import joblib
import sqlite3
import smtplib
import os
import logging
import google.generativeai as genai
from dotenv import load_dotenv
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from werkzeug.security import generate_password_hash, check_password_hash

# ============================================================
# INITIAL CONFIG
# ============================================================

app = Flask(__name__)
app.secret_key = "supersecretkey"
logging.basicConfig(level=logging.DEBUG)

# Load ML Model
try:
    model = joblib.load("multimodal_scam_model.pkl")
    print(f"Model loaded: {type(model)}")
except Exception as e:
    print(f"Error loading model: {e}")
    # Create a dummy model for testing if file is missing
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier()
    print("Using dummy model for testing")

# Gemini Setup
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("Warning: GEMINI_API_KEY not set. Gemini requests will fail until configured.")
else:
    print(f"Debug: API Key loaded, length={len(GEMINI_API_KEY)}")
genai.configure(api_key=GEMINI_API_KEY)

# List available models and find one that supports generateContent
gemini_model = None
try:
    print("Available models:")
    for available_model in genai.list_models():
        print(f"  - {available_model.name}: {available_model.supported_generation_methods}")
        if 'generateContent' in available_model.supported_generation_methods and gemini_model is None:
            gemini_model = genai.GenerativeModel(available_model.name)
            print(f"Selected model: {available_model.name}")
except Exception as e:
    print(f"Error listing models: {e}")
    # Fallback to a stable model name format
    gemini_model = genai.GenerativeModel("gemini-2.5-flash-lite")

# Email Config
SENDER_EMAIL = "prosdgunal@gmail.com"
SENDER_PASSWORD = "wyfc rppx gkml thfz"
RECEIVER_EMAIL = "jaheerahaajimohammed@gmail.com"

# ============================================================
# DATABASE INIT
# ============================================================

def init_db():
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    """)
    conn.commit()
    conn.close()

init_db()

# ============================================================
# EMAIL FUNCTION
# ============================================================

def send_email_alert(probability):
    subject = "🚨 Scam Detection Alert!"
    body = f"""
ALERT!

A suspicious scam activity has been detected.

Confidence Level: {probability:.2f}

Please review immediately.
"""
    msg = MIMEMultipart()
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECEIVER_EMAIL
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.send_message(msg)
        server.quit()
    except Exception as e:
        print("Email Error:", e)

# ============================================================
# ROUTES
# ============================================================

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form["name"]
        email = request.form["email"]
        password = generate_password_hash(request.form["password"])

        try:
            conn = sqlite3.connect("users.db")
            cursor = conn.cursor()
            cursor.execute("INSERT INTO users (name, email, password) VALUES (?, ?, ?)",
                           (name, email, password))
            conn.commit()
            conn.close()
            flash("Registration Successful! Please Login.")
            return redirect(url_for("login"))
        except:
            flash("Email already exists!")

    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]

        conn = sqlite3.connect("users.db")
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users WHERE email=?", (email,))
        user = cursor.fetchone()
        conn.close()

        if user and check_password_hash(user[3], password):
            session["user"] = user[1]
            return redirect(url_for("index"))
        else:
            flash("Invalid Email or Password!")

    return render_template("login.html")

@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("home"))

# ============================================================
# SCAM DETECTION (PROTECTED)
# ============================================================

@app.route("/index", methods=["GET", "POST"])
def index():

    if "user" not in session:
        return redirect(url_for("login"))

    prediction = None
    probability = None

    if request.method == "POST":

        input_data = pd.DataFrame([{
            "account_age_days": int(request.form["account_age_days"]),
            "num_followers": int(request.form["num_followers"]),
            "num_following": int(request.form["num_following"]),
            "num_posts": int(request.form["num_posts"]),
            "num_posts_last_7_days": int(request.form["num_posts_last_7_days"]),
            "avg_likes_per_post": int(request.form["avg_likes_per_post"]),
            "avg_comments_per_post": int(request.form["avg_comments_per_post"]),
            "num_shares_per_post": int(request.form["num_shares_per_post"]),
            "num_hashtags_per_post": int(request.form["num_hashtags_per_post"]),
            "num_messages_sent": int(request.form["num_messages_sent"]),
            "num_suspicious_links_shared": int(request.form["num_suspicious_links_shared"]),
            "num_reports_received": int(request.form["num_reports_received"]),
            "transaction_amount": float(request.form["transaction_amount"]),
            "purchase_time": int(request.form["purchase_time"]),
            "sentiment": request.form["sentiment"],
            "payment_method": request.form["payment_method"],
            "text_message": request.form["text_message"]
        }])

        try:
            prediction = model.predict(input_data)[0]
            probability = model.predict_proba(input_data)[0][1]
        except AttributeError as e:
            logging.error(f"Model error: {e}. Model type: {type(model)}")
            return render_template("index.html",
                                   prediction=None,
                                   probability=None,
                                   user=session["user"],
                                   error=f"Model error: {str(e)}")

        if prediction == 1:
            send_email_alert(probability)

    return render_template("index.html",
                           prediction=prediction,
                           probability=probability,
                           user=session["user"],
                           error=None)

# ============================================================
# GEMINI CHATBOT API
# ============================================================

@app.route("/chat", methods=["POST"])
def chat():

    if "user" not in session:
        return jsonify({"response": "Please login first."})

    user_message = request.json.get("message")

    try:
        response = gemini_model.generate_content(
            f"You are a cybersecurity expert. Answer clearly.\nUser: {user_message}"
        )
        return jsonify({"response": response.text})
    except Exception as e:
        logging.error(f"Gemini API Error: {str(e)}")
        return jsonify({"response": f"Error: {str(e)}"})

# ============================================================
# RUN APP
# ============================================================

if __name__ == "__main__":
    app.run(debug=True)
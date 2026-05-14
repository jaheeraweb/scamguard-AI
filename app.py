# ============================================================
# STREAMLIT APP - MULTI MODAL SCAM DETECTION
# ============================================================

import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ============================================================
# 1. LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():
    return joblib.load("multimodal_scam_model.pkl")

model = load_model()

st.set_page_config(page_title="AI Scam Detection", layout="wide")

st.title("🚨 AI Powered Multi-Modal Scam Detection System")
st.markdown("Detect Scam Posts & Fraud Transactions in Real-Time")

# ============================================================
# 2. USER INPUT FORM
# ============================================================

st.subheader("🔹 User Behaviour Information")

col1, col2, col3 = st.columns(3)

with col1:
    account_age_days = st.number_input("Account Age (days)", 0, 5000, 365)
    num_followers = st.number_input("Number of Followers", 0, 1000000, 500)
    num_following = st.number_input("Number Following", 0, 1000000, 300)
    num_posts = st.number_input("Total Posts", 0, 100000, 50)

with col2:
    num_posts_last_7_days = st.number_input("Posts Last 7 Days", 0, 1000, 5)
    avg_likes_per_post = st.number_input("Average Likes", 0, 100000, 100)
    avg_comments_per_post = st.number_input("Average Comments", 0, 10000, 10)
    num_shares_per_post = st.number_input("Shares Per Post", 0, 10000, 5)

with col3:
    num_hashtags_per_post = st.number_input("Hashtags Per Post", 0, 100, 3)
    num_messages_sent = st.number_input("Messages Sent", 0, 10000, 50)
    num_suspicious_links_shared = st.number_input("Suspicious Links Shared", 0, 100, 0)
    num_reports_received = st.number_input("Reports Received", 0, 1000, 0)

st.subheader("🔹 Transaction & Content Info")

col4, col5 = st.columns(2)

with col4:
    transaction_amount = st.number_input("Transaction Amount", 0.0, 1000000.0, 1000.0)
    purchase_time = st.number_input("Purchase Time (24hr format)", 0, 23, 12)
    sentiment = st.selectbox("Sentiment", ["positive", "neutral", "negative"])

with col5:
    payment_method = st.selectbox("Payment Method",
                                  ["credit_card", "debit_card", "upi", "paypal", "crypto"])
    text_message = st.text_area("Post / Message Text")

# ============================================================
# 3. PREDICTION
# ============================================================

if st.button("🔍 Predict Scam"):

    input_data = pd.DataFrame([{
        "account_age_days": account_age_days,
        "num_followers": num_followers,
        "num_following": num_following,
        "num_posts": num_posts,
        "num_posts_last_7_days": num_posts_last_7_days,
        "avg_likes_per_post": avg_likes_per_post,
        "avg_comments_per_post": avg_comments_per_post,
        "num_shares_per_post": num_shares_per_post,
        "num_hashtags_per_post": num_hashtags_per_post,
        "num_messages_sent": num_messages_sent,
        "num_suspicious_links_shared": num_suspicious_links_shared,
        "num_reports_received": num_reports_received,
        "transaction_amount": transaction_amount,
        "purchase_time": purchase_time,
        "sentiment": sentiment,
        "payment_method": payment_method,
        "text_message": text_message
    }])

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.subheader("📊 Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ Scam Detected! (Confidence: {probability:.2f})")
    else:
        st.success(f"✅ Legitimate Activity (Confidence: {1 - probability:.2f})")

    st.progress(float(probability))

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.markdown("Developed for AI-Based Online Scam Detection System")
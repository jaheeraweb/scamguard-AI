import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import traceback

class ScamDetector:
    """Scam detection model handler"""
    
    def __init__(self, dataset_path, model_path, expected_features=83, text_features=72):
        self.dataset_path = dataset_path
        self.model_path = model_path
        self.expected_features = expected_features
        self.text_features = text_features
        self.numeric_features = expected_features - text_features
        self.tfidf = None
        self.model = None
        self.is_loaded = False
        self.error_message = ""
        self.load_models()
    
    def load_models(self):
        """Load the TF-IDF vectorizer and the trained model"""
        try:
            print(f"\n🔍 Loading models...")
            print(f"   Dataset path: {self.dataset_path}")
            print(f"   Model path: {self.model_path}")
            
            # Check if files exist
            if not os.path.exists(self.dataset_path):
                self.error_message = f"Dataset not found at {self.dataset_path}"
                print(f"   ❌ {self.error_message}")
                return False
                
            if not os.path.exists(self.model_path):
                self.error_message = f"Model not found at {self.model_path}"
                print(f"   ❌ {self.error_message}")
                return False
            
            # Load dataset
            print(f"   ✅ Dataset file found")
            try:
                data = pd.read_csv(self.dataset_path)
                print(f"   ✅ Dataset loaded: {data.shape[0]} rows, {data.shape[1]} columns")
            except Exception as e:
                self.error_message = f"Error loading dataset: {str(e)}"
                print(f"   ❌ {self.error_message}")
                return False
            
            # Check if required column exists
            if 'text_message' not in data.columns:
                self.error_message = "Dataset must contain 'text_message' column"
                print(f"   ❌ {self.error_message}")
                print(f"   Available columns: {list(data.columns)}")
                return False
            
            # Initialize TF-IDF
            print(f"   ✅ Initializing TF-IDF vectorizer with {self.text_features} features...")
            self.tfidf = TfidfVectorizer(max_features=self.text_features)
            self.tfidf.fit(data['text_message'].astype(str))
            print(f"   ✅ TF-IDF vectorizer fitted successfully")
            
            # Load model
            print(f"   ✅ Model file found")
            try:
                self.model = joblib.load(self.model_path)
                print(f"   ✅ Model loaded successfully")
            except Exception as e:
                self.error_message = f"Error loading model: {str(e)}"
                print(f"   ❌ {self.error_message}")
                return False
            
            # Verify model
            if hasattr(self.model, 'n_features_in_'):
                model_features = self.model.n_features_in_
                print(f"   📊 Model expects {model_features} features")
                if model_features != self.expected_features:
                    print(f"   ⚠️ Warning: Model expects {model_features} features, but configured for {self.expected_features}")
                    # Update expected features to match model
                    self.expected_features = model_features
                    self.text_features = model_features - self.numeric_features
                    print(f"   📊 Adjusted: text_features={self.text_features}, total={self.expected_features}")
            
            self.is_loaded = True
            print(f"   ✅ Models loaded successfully!\n")
            return True
            
        except Exception as e:
            self.error_message = f"Unexpected error: {str(e)}\n{traceback.format_exc()}"
            print(f"   ❌ {self.error_message}")
            self.is_loaded = False
            return False
    
    def encode_features(self, account_age_days, num_followers, num_following,
                        avg_post_sequence, avg_transaction_sequence,
                        contains_link, contains_media, num_hashtags_per_post,
                        sentiment, product_category, payment_method, text_message):
        """Encode all features for prediction with proper dimension handling"""
        
        if not self.is_loaded:
            raise Exception(f"Models not loaded properly: {self.error_message}")
        
        try:
            # Encode categorical features
            sentiment_map = {"positive": 0, "neutral": 1, "negative": 2}
            sentiment_code = sentiment_map.get(sentiment.lower(), 1)
            
            product_map = {
                "Electronics": 0, "Clothing": 1, "Beauty": 2, 
                "Home Appliances": 3, "Toys": 4
            }
            product_category_code = product_map.get(product_category, 0)
            
            payment_map = {
                "Credit Card": 0, "UPI": 1, "PayPal": 2
            }
            payment_method_code = payment_map.get(payment_method, 0)

            # TF-IDF for text
            text_features = self.tfidf.transform([text_message]).toarray()
            
            # Ensure text features have correct dimension
            actual_text_features = text_features.shape[1]
            if actual_text_features < self.text_features:
                # Pad with zeros if we have fewer features
                padding = np.zeros((1, self.text_features - actual_text_features))
                text_features = np.hstack([text_features, padding])
                print(f"   ℹ️ Padded text features from {actual_text_features} to {self.text_features}")
            elif actual_text_features > self.text_features:
                # Truncate if we have more features
                text_features = text_features[:, :self.text_features]
                print(f"   ℹ️ Truncated text features from {actual_text_features} to {self.text_features}")

            # Create numeric features array
            numeric_features = np.array([
                float(account_age_days), 
                float(num_followers), 
                float(num_following),
                float(avg_post_sequence), 
                float(avg_transaction_sequence),
                float(contains_link), 
                float(contains_media), 
                float(num_hashtags_per_post),
                float(sentiment_code), 
                float(product_category_code), 
                float(payment_method_code)
            ]).reshape(1, -1)
            
            # Concatenate numeric and text features
            features = np.hstack([numeric_features, text_features])
            
            # Final dimension check
            if features.shape[1] != self.expected_features:
                print(f"   ⚠️ Feature shape mismatch. Got {features.shape[1]}, expected {self.expected_features}")
                # Adjust to match expected features
                if features.shape[1] < self.expected_features:
                    padding = np.zeros((1, self.expected_features - features.shape[1]))
                    features = np.hstack([features, padding])
                else:
                    features = features[:, :self.expected_features]
            
            print(f"   ✅ Encoded features shape: {features.shape}")
            return features
            
        except Exception as e:
            raise Exception(f"Error encoding features: {str(e)}")
    
    def predict(self, features):
        """Make prediction on encoded features"""
        try:
            # Validate feature shape
            if features.shape[1] != self.expected_features:
                raise ValueError(f"Feature shape mismatch: expected {self.expected_features}, got {features.shape[1]}")
            
            # Make prediction
            prediction = self.model.predict(features)[0]
            
            # Get probabilities if available
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(features)[0]
                
                # Handle different probability array lengths
                if len(probabilities) >= 2:
                    prob_scam = float(probabilities[1])
                    prob_legit = float(probabilities[0])
                    confidence = float(max(probabilities))
                else:
                    prob_scam = float(probabilities[0]) if prediction == 1 else 0
                    prob_legit = float(probabilities[0]) if prediction == 0 else 0
                    confidence = float(probabilities[0])
            else:
                # If no predict_proba, use default values
                prob_scam = 1.0 if prediction == 1 else 0.0
                prob_legit = 1.0 if prediction == 0 else 0.0
                confidence = 1.0
            
            result = {
                'is_scam': bool(prediction == 1),
                'confidence': confidence,
                'probability_scam': prob_scam,
                'probability_legit': prob_legit,
                'prediction_label': 'SCAM' if prediction == 1 else 'LEGITIMATE'
            }
            
            print(f"   ✅ Prediction: {result['prediction_label']} (confidence: {confidence:.2f})")
            return result
            
        except Exception as e:
            raise Exception(f"Prediction error: {str(e)}")
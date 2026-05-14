import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import joblib
import logging
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BERTScamDetector:
    """
    BERT-based model for text classification of scam posts.
    """
    
    def __init__(self, config: Any):
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased', 
            num_labels=2  # Binary classification: scam vs legitimate
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def prepare_data(self, texts: List[str], labels: List[int]) -> torch.utils.data.Dataset:
        """Prepare text data for BERT."""
        
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=self.config.MAX_SEQUENCE_LENGTH,
            return_tensors='pt'
        )
        
        dataset = TensorDataset(
            encodings['input_ids'],
            encodings['attention_mask'],
            torch.tensor(labels)
        )
        
        return dataset
    
    def train(self, train_texts: List[str], train_labels: List[int], 
              val_texts: List[str], val_labels: List[int]):
        """Train the BERT model."""
        
        train_dataset = self.prepare_data(train_texts, train_labels)
        val_dataset = self.prepare_data(val_texts, val_labels)
        
        train_loader = DataLoader(train_dataset, batch_size=self.config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.BATCH_SIZE)
        
        optimizer = AdamW(self.model.parameters(), lr=self.config.LEARNING_RATE)
        
        logger.info("Starting BERT training...")
        
        for epoch in range(self.config.EPOCHS):
            self.model.train()
            total_loss = 0
            
            for batch in train_loader:
                input_ids, attention_mask, labels = [b.to(self.device) for b in batch]
                
                optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()
                
                loss.backward()
                optimizer.step()
            
            # Validation
            self.model.eval()
            val_predictions = []
            val_true = []
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids, attention_mask, labels = [b.to(self.device) for b in batch]
                    outputs = self.model(input_ids, attention_mask=attention_mask)
                    predictions = torch.argmax(outputs.logits, dim=-1)
                    
                    val_predictions.extend(predictions.cpu().numpy())
                    val_true.extend(labels.cpu().numpy())
            
            accuracy = accuracy_score(val_true, val_predictions)
            logger.info(f"Epoch {epoch+1}/{self.config.EPOCHS} - Loss: {total_loss:.4f}, Val Accuracy: {accuracy:.4f}")
        
        # Save model
        self.model.save_pretrained(f"{self.config.MODEL_SAVE_PATH}bert_scam_detector")
        self.tokenizer.save_pretrained(f"{self.config.MODEL_SAVE_PATH}bert_tokenizer")
        
    def predict(self, texts: List[str]) -> np.ndarray:
        """Predict scam probability for new texts."""
        
        self.model.eval()
        dataset = self.prepare_data(texts, [0] * len(texts))  # Dummy labels
        loader = DataLoader(dataset, batch_size=self.config.BATCH_SIZE)
        
        predictions = []
        probabilities = []
        
        with torch.no_grad():
            for batch in loader:
                input_ids, attention_mask, _ = [b.to(self.device) for b in batch]
                outputs = self.model(input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=-1)
                
                predictions.extend(torch.argmax(outputs.logits, dim=-1).cpu().numpy())
                probabilities.extend(probs.cpu().numpy())
        
        return np.array(predictions), np.array(probabilities)


class EnsembleScamDetector:
    """
    Ensemble of traditional ML models for scam detection.
    """
    
    def __init__(self, config: Any):
        self.config = config
        self.models = {}
        self.ensemble_model = None
        
    def build_models(self):
        """Initialize individual models."""
        
        # Random Forest
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        # Gradient Boosting
        self.models['gradient_boosting'] = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
        # Neural Network
        self.models['neural_network'] = MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            max_iter=500,
            random_state=42
        )
        
        # Create voting ensemble
        self.ensemble_model = VotingClassifier(
            estimators=[(name, model) for name, model in self.models.items()],
            voting='soft'  # Use probability voting
        )
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
        """Train all models and the ensemble."""
        
        logger.info("Training ensemble models...")
        
        results = {}
        
        # Train individual models
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            model.fit(X_train, y_train)
            
            # Validate
            y_pred = model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            precision = precision_score(y_val, y_pred)
            recall = recall_score(y_val, y_pred)
            f1 = f1_score(y_val, y_pred)
            
            results[name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
            
            logger.info(f"{name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
            
            # Save individual model
            joblib.dump(model, f"{self.config.MODEL_SAVE_PATH}{name}.pkl")
        
        # Train ensemble
        logger.info("Training ensemble model...")
        self.ensemble_model.fit(X_train, y_train)
        
        y_pred = self.ensemble_model.predict(X_val)
        results['ensemble'] = {
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred),
            'recall': recall_score(y_val, y_pred),
            'f1': f1_score(y_val, y_pred)
        }
        
        logger.info(f"Ensemble - Accuracy: {results['ensemble']['accuracy']:.4f}, "
                   f"F1: {results['ensemble']['f1']:.4f}")
        
        # Save ensemble
        joblib.dump(self.ensemble_model, f"{self.config.MODEL_SAVE_PATH}ensemble_model.pkl")
        
        return results
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict using ensemble model."""
        
        if self.ensemble_model is None:
            # Load saved model
            self.ensemble_model = joblib.load(f"{self.config.MODEL_SAVE_PATH}ensemble_model.pkl")
        
        predictions = self.ensemble_model.predict(X)
        probabilities = self.ensemble_model.predict_proba(X)
        
        return predictions, probabilities
    
    def get_feature_importance(self, feature_names: List[str]) -> pd.DataFrame:
        """Get feature importance from Random Forest model."""
        
        rf_model = self.models.get('random_forest')
        if rf_model is None:
            rf_model = joblib.load(f"{self.config.MODEL_SAVE_PATH}random_forest.pkl")
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return importance_df


class ScamDetectionSystem:
    """
    Complete scam detection system combining BERT and ensemble models.
    """
    
    def __init__(self, config: Any):
        self.config = config
        self.bert_detector = BERTScamDetector(config)
        self.ensemble_detector = EnsembleScamDetector(config)
        self.feature_extractor = None  # Will be set later
        
    def train(self, X_features: np.ndarray, texts: List[str], 
              y: np.ndarray, feature_names: List[str]):
        """Train both detection models."""
        
        # Split data
        X_train_f, X_val_f, X_train_t, X_val_t, y_train, y_val = train_test_split(
            X_features, texts, y, 
            test_size=self.config.TRAIN_TEST_SPLIT,
            random_state=42,
            stratify=y
        )
        
        # Train BERT on text data
        logger.info("Training BERT detector...")
        self.bert_detector.train(X_train_t, y_train, X_val_t, y_val)
        
        # Train ensemble on features
        logger.info("Training ensemble detector...")
        self.ensemble_detector.build_models()
        results = self.ensemble_detector.train(X_train_f, y_train, X_val_f, y_val)
        
        # Get feature importance
        importance_df = self.ensemble_detector.get_feature_importance(feature_names)
        logger.info("\nTop 10 Most Important Features:")
        logger.info(importance_df.head(10))
        
        return results
    
    def predict(self, features: np.ndarray, texts: List[str]) -> Dict[str, Any]:
        """Make predictions using both models."""
        
        # Get predictions from both models
        bert_pred, bert_probs = self.bert_detector.predict(texts)
        ensemble_pred, ensemble_probs = self.ensemble_detector.predict(features)
        
        # Combine predictions (weighted average)
        combined_probs = (0.4 * bert_probs + 0.6 * ensemble_probs)
        combined_pred = np.argmax(combined_probs, axis=1)
        
        # Calculate risk scores
        risk_scores = combined_probs[:, 1]  # Probability of being scam
        
        # Categorize risk
        risk_categories = []
        for score in risk_scores:
            if score >= self.config.HIGH_RISK_THRESHOLD:
                risk_categories.append('HIGH_RISK')
            elif score >= self.config.SCAM_PROBABILITY_THRESHOLD:
                risk_categories.append('MEDIUM_RISK')
            else:
                risk_categories.append('LOW_RISK')
        
        return {
            'predictions': combined_pred,
            'risk_scores': risk_scores,
            'risk_categories': risk_categories,
            'bert_probabilities': bert_probs,
            'ensemble_probabilities': ensemble_probs
        }
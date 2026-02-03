"""
ml_models.py - Machine Learning Models and Explainable AI
=========================================================
Implements unsupervised learning models for anomaly detection:
- Autoencoder (neural network based)
- Isolation Forest (tree based)
- Ensemble scoring
- Explainability (XAI) for alerts
"""

import numpy as np                                                            # for matrix multiplications and numerical operations(faster)
import logging                                                                # for logging info and debug messages
from typing import Dict, List, Tuple, Any                                     # for type hinting                            
from abc import ABC, abstractmethod                                           # for abstract base classes( can't be instantiated and enforce method implementation in subclasses)

# ML libraries with pretrained models and utilities
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import warnings                                                               # to manage warning messages
warnings.filterwarnings('ignore')                                             # ignore warnings for cleaner output                                                       

logger = logging.getLogger('IDS.Models')                                      # create logger for this module : a logger is an object that logs messages for a specific system component


# ============================================================================
# BASE ANOMALY DETECTOR INTERFACE (Open/Closed Principle)
# ============================================================================

class AnomalyDetector(ABC):
    """
    Abstract base class for anomaly detectors.
    Allows easy extension with new detection algorithms.
    """
    
    @abstractmethod
    def train(self, X: np.ndarray):
        """Train the model on normal traffic data."""
        pass
    
    @abstractmethod
    def predict_anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomaly scores for samples.
        Returns: Array of scores in range [0, 1], where 1 = most anomalous
        """
        pass
    
    @abstractmethod
    def explain(self, X: np.ndarray) -> List[Dict[str, Any]]:
        """
        Explain why samples are anomalous.
        Returns: List of explanations, one per sample.
        """
        pass


# ============================================================================
# AUTOENCODER MODEL
# ============================================================================

class SimpleAutoencoder:
    """
    Simple feedforward autoencoder for anomaly detection.
    
    How it works:
    1. Learns to compress normal data into lower dimensions (encoding)
    2. Then reconstructs the original data (decoding)
    3. If reconstruction error is high -> anomaly
    
    This works because the model learns patterns in normal traffic.
    Abnormal traffic doesn't follow these patterns, so it reconstructs poorly.
    """
    
    def __init__(self, input_dim: int, latent_dim: int = 5):
        self.input_dim = input_dim                                          # input feature dimension
        self.latent_dim = latent_dim                                        # dimension of latent space                  
        self.scaler = StandardScaler()                                      # for feature normalization              
        
        # Simple architecture: input -> latent -> output (11 -> 5 -> 11)
        # set up weights and biases to None, will be initialized during
        self.encoder_weights = None
        self.encoder_bias = None
        self.decoder_weights = None
        self.decoder_bias = None

        '''
        ENCODER:
        Input (11) * encoder_weights (11*5) + encoder_bias (5) = Latent (5)

        DECODER:
        Latent (5) * decoder_weights (5*11) + decoder_bias (11) = Output (11)
        '''
        
        self.is_trained = False
    
    def _sigmoid(self, x):                                            #fonction d'activation sigmoide
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))               # clip to avoid overflow: limits x to range [-500, 500]
    
    def _sigmoid_derivative(self, x):                                   # we need it for backpropagation in training
        s = self._sigmoid(x)                                            # application of sigmoide function to s 
        return s * (1 - s)                                              # derivative formula of sigmoide function
    
    def _forward(self, X):
        # Encode : compress input to latent space
        self.z1 = np.dot(X, self.encoder_weights) + self.encoder_bias      # calculate weighted sum + bias (linear transformation) (why : to learn relationships between input features)
        self.a1 = self._sigmoid(self.z1)                                   # apply activation function (non-linear transformation)  (why : to introduce non-linearity in the model)              
        
        # Decode : reconstruct input from latent space
        self.z2 = np.dot(self.a1, self.decoder_weights) + self.decoder_bias             # weighted sum + bias 
        self.a2 = self._sigmoid(self.z2)                                                # activation function                    
        
        return self.a2
    
    def _backward(self, X, output, learning_rate):     # retropropagation : calculate how much each weight contributed to the error then update weights accordingly (repeate until error is minimized)
        
        m = X.shape[0]                                              # number of samples in batch 
        
        # Output layer gradients
        dz2 = output - X                                            # calculate error at output layer (predicted - real) : Math derivation leads to: dL/dz2 = output - X( loss function : Mean Squared Error)
        dw2 = np.dot(self.a1.T, dz2) / m                            # gradient w.r.t. decoder weights --  devide by m to get average gradient over batch
        db2 = np.sum(dz2, axis=0, keepdims=True) / m                # gradient w.r.t. decoder bias (keepdims means keep the same number of dimensions otherwise sum would reduce dimensions)
        

        # Hidden layer gradients
        da1 = np.dot(dz2, self.decoder_weights.T)                   # backpropagate error to hidden layer
        dz1 = da1 * self._sigmoid_derivative(self.z1)               # apply derivative of activation function
        dw1 = np.dot(X.T, dz1) / m                                  # gradient w.r.t. encoder weights
        db1 = np.sum(dz1, axis=0, keepdims=True) / m                # gradient w.r.t. encoder bias
        
        # Update weights
        self.encoder_weights -= learning_rate * dw1
        self.encoder_bias -= learning_rate * db1
        self.decoder_weights -= learning_rate * dw2
        self.decoder_bias -= learning_rate * db2
    
    def train(self, X: np.ndarray, epochs: int = 50, learning_rate: float = 0.1):
        """
        Train the autoencoder on normal traffic.
        
        Args:
            X: Training data (normal traffic only)
            epochs: Number of training iterations
            learning_rate: Learning rate for gradient descent
        """
        # Normalize data
        X_scaled = self.scaler.fit_transform(X)                       # normalize features after learning mean and std from training data
        
        # Initialize weights randomly
        np.random.seed(42)                                            # Set random seed for reproducibility 
        self.encoder_weights = np.random.randn(self.input_dim, self.latent_dim) * 0.1        # creates random array of shape (11,5) scaled by 0.1 to keep initial weights small
        self.encoder_bias = np.zeros((1, self.latent_dim))                                   # initializes bias to zeros of shape (1,5)
        self.decoder_weights = np.random.randn(self.latent_dim, self.input_dim) * 0.1        # creates random array of shape (5,11) scaled by 0.1
        self.decoder_bias = np.zeros((1, self.input_dim))                                    # initializes bias to zeros of shape (1,11)
        
        # Training loop
        for epoch in range(epochs):
            # Forward pass
            output = self._forward(X_scaled)                 # get reconstructed output
            
            # Calculate loss (Mean Squared Error)
            loss = np.mean((output - X_scaled) ** 2)
            
            # Backward pass
            self._backward(X_scaled, output, learning_rate)            # update weights based on error
            
            if (epoch + 1) % 10 == 0:              # log every 10 epochs 
                logger.info(f"Autoencoder epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")
        
        self.is_trained = True
        logger.info("Autoencoder training complete")           # Mark as trained and log completion 
    
    def reconstruction_error(self, X: np.ndarray) -> np.ndarray:     # Calculate error for each sample
        """
        Calculate reconstruction error for each sample.
        Higher error = more anomalous.
        """
        if not self.is_trained:
            raise ValueError("Model must be trained first")
        
        X_scaled = self.scaler.transform(X)           # normalize input data using learned scaler
        reconstructed = self._forward(X_scaled)
        
        # Mean squared error per sample
        errors = np.mean((X_scaled - reconstructed) ** 2, axis=1)
        return errors
    
    def get_feature_contributions(self, X: np.ndarray) -> np.ndarray:          # Calculate feature-wise contributions to reconstruction error
        """
        Calculate how much each feature contributes to the reconstruction error.
        This is used for explainability.
        """
        X_scaled = self.scaler.transform(X)
        reconstructed = self._forward(X_scaled)
        
        # Squared error per feature
        feature_errors = (X_scaled - reconstructed) ** 2
        return feature_errors


class AutoencoderDetector(AnomalyDetector):
    """
    Autoencoder-based anomaly detector with explainability.
    Single Responsibility: Autoencoder detection logic.
    """
    
    def __init__(self, input_dim: int, latent_dim: int = 5):       # initialize autoencoder model
        self.model = SimpleAutoencoder(input_dim, latent_dim)      # create autoencoder instance from SimpleAutoencoder class
        self.threshold = None                                      # anomaly detection threshold : will be set after training                   
        self.feature_names = None                                  # to store feature names for explanations
    
    def train(self, X: np.ndarray, feature_names: List[str] = None):
        """Train on normal traffic and set detection threshold."""
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
        
        logger.info("Training autoencoder...")         # log training start
        self.model.train(X, epochs=50)                 # train autoencoder model : .train is method of SimpleAutoencoder class
        
        # Set threshold at 95th percentile of reconstruction errors
        errors = self.model.reconstruction_error(X)
        self.threshold = np.percentile(errors, 95)    #percentile function returns the value below which a given percentage of observations fall
        logger.info(f"Autoencoder threshold set to: {self.threshold:.4f}")
    
    def predict_anomaly_score(self, X: np.ndarray) -> np.ndarray:          # predict anomaly scores : higher score = more anomalous
        """
        Predict anomaly scores normalized to [0, 1].
        """
        errors = self.model.reconstruction_error(X)                      # reconstruction_error is method of SimpleAutoencoder class
        # Normalize scores: anything above threshold gets higher scores
        scores = np.clip(errors / (self.threshold * 2), 0, 1)            # multiply by 2 to allow some margin above threshold - clip to set values > than 1 to 1 
        return scores
    
    def explain(self, X: np.ndarray) -> List[Dict[str, Any]]:
        """
        Explain which features contributed most to anomaly detection.
        """
        feature_errors = self.model.get_feature_contributions(X)       # get feature-wise reconstruction errors
        explanations = []                                              # set an empty list to store explanations
        
        for i in range(X.shape[0]):                                    # iterate over each sample
            # Get feature contributions for this sample
            contributions = feature_errors[i]                          # feature-wise errors for sample i
            
            # Rank features by contribution
            feature_importance = [
                {'feature': self.feature_names[j], 'contribution': float(contributions[j])}
                for j in range(len(contributions))
            ]
            feature_importance.sort(key=lambda x: x['contribution'], reverse=True)   # sort features by contribution descending
            
            # Generate explanation text
            top_features = feature_importance[:3]
            explanation_text = "Anomaly detected due to unusual values in: " + \
                             ", ".join([f"{f['feature']} ({f['contribution']:.3f})" 
                                       for f in top_features])
            
            explanations.append({
                'method': 'autoencoder',
                'feature_importance': feature_importance,
                'explanation': explanation_text
            })
        
        return explanations


# ============================================================================
# ISOLATION FOREST MODEL
# ============================================================================

class IsolationForestDetector(AnomalyDetector):
    """
    Isolation Forest based anomaly detector.
    
    How it works:
    - Randomly creates decision trees
    - Anomalies are isolated (separated) in fewer splits
    - Normal points require more splits to isolate
    
    Complementary to autoencoder: good at detecting global outliers.
    """
    
    def __init__(self, contamination: float = 0.1):
        self.model = IsolationForest(contamination=contamination, random_state=42, n_estimators=100)
        self.scaler = StandardScaler()
        self.feature_names = None
    
    def train(self, X: np.ndarray, feature_names: List[str] = None):
        """Train the isolation forest on normal traffic."""
        self.feature_names = feature_names or [f"feature_{i}" for i in range(X.shape[1])]
        
        logger.info("Training isolation forest...")
        X_scaled = self.scaler.fit_transform(X)            # normalize features
        self.model.fit(X_scaled)                           # fit isolation forest model
        logger.info("Isolation forest training complete")
    
    def predict_anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomaly scores in [0, 1].
        Isolation Forest returns scores in [-1, 1], we convert to [0, 1].
        """
        X_scaled = self.scaler.transform(X)
        # score_samples returns negative scores for anomalies
        scores = self.model.score_samples(X_scaled)         # score_samples is method of IsolationForest class
        # Convert to [0, 1] where 1 is most anomalous
        normalized_scores = 1 - (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)
        return np.clip(normalized_scores, 0, 1)
    
    def explain(self, X: np.ndarray) -> List[Dict[str, Any]]:
        """
        Explain anomalies based on feature values.
        Uses decision path length as a proxy for feature importance.
        """
        X_scaled = self.scaler.transform(X)
        explanations = []
        
        for i in range(X.shape[0]):
            sample = X_scaled[i:i+1]
            
            # Get average path length (lower = more anomalous)
            path_length = self.model.decision_function(sample)[0]
            
            # Simple heuristic: features with extreme values are suspicious
            feature_importance = []
            for j, feature_name in enumerate(self.feature_names):
                # How far from mean (in standard deviations)
                deviation = abs(sample[0, j])
                feature_importance.append({
                    'feature': feature_name,
                    'contribution': float(deviation)
                })
            
            feature_importance.sort(key=lambda x: x['contribution'], reverse=True)
            
            top_features = feature_importance[:3]
            explanation_text = f"Isolation Forest detected outlier. " + \
                             f"Most deviant features: " + \
                             ", ".join([f"{f['feature']}" for f in top_features])
            
            explanations.append({
                'method': 'isolation_forest',
                'feature_importance': feature_importance,
                'explanation': explanation_text
            })
        
        return explanations


# ============================================================================
# ENSEMBLE DETECTOR
# ============================================================================

class EnsembleDetector:
    """
    Combines multiple detectors for robust anomaly detection.
    Uses weighted average of scores.
    
    Single Responsibility: Coordinate multiple detectors.
    """
    
    def __init__(self, input_dim: int, feature_names: List[str]):
        self.feature_names = feature_names
        
        # Initialize detectors
        self.autoencoder = AutoencoderDetector(input_dim, latent_dim=5)
        self.isolation_forest = IsolationForestDetector(contamination=0.1)
        
        # Weights for ensemble
        self.weights = {
            'autoencoder': 0.6,
            'isolation_forest': 0.4
        }
    
    def train(self, X: np.ndarray):
        """Train all detectors."""
        logger.info("Training ensemble...")
        self.autoencoder.train(X, self.feature_names)
        self.isolation_forest.train(X, self.feature_names)
        logger.info("Ensemble training complete")
    
    def predict_anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """
        Get ensemble anomaly score (weighted average).
        """
        ae_scores = self.autoencoder.predict_anomaly_score(X)
        if_scores = self.isolation_forest.predict_anomaly_score(X)
        
        # Weighted average
        ensemble_scores = (
            self.weights['autoencoder'] * ae_scores +
            self.weights['isolation_forest'] * if_scores
        )
        
        return ensemble_scores
    
    def explain(self, X: np.ndarray) -> List[Dict[str, Any]]:
        """
        Get explanations from all detectors and combine them.
        """
        ae_explanations = self.autoencoder.explain(X)
        if_explanations = self.isolation_forest.explain(X)
        
        combined_explanations = []
        for i in range(X.shape[0]):
            # Merge feature importance from both methods
            all_features = {}
            
            for feat in ae_explanations[i]['feature_importance']:
                name = feat['feature']
                all_features[name] = all_features.get(name, 0) + \
                                    feat['contribution'] * self.weights['autoencoder']
            
            for feat in if_explanations[i]['feature_importance']:
                name = feat['feature']
                all_features[name] = all_features.get(name, 0) + \
                                    feat['contribution'] * self.weights['isolation_forest']
            
            # Sort by combined importance
            feature_importance = [
                {'feature': k, 'contribution': v}
                for k, v in all_features.items()
            ]
            feature_importance.sort(key=lambda x: x['contribution'], reverse=True)
            
            # Create combined explanation
            top_features = feature_importance[:3]
            explanation_text = "ENSEMBLE DETECTION: Anomaly detected. " + \
                             "Top suspicious features: " + \
                             ", ".join([f"{f['feature']} (score: {f['contribution']:.3f})" 
                                       for f in top_features])
            
            combined_explanations.append({
                'method': 'ensemble',
                'autoencoder_explanation': ae_explanations[i]['explanation'],
                'isolation_forest_explanation': if_explanations[i]['explanation'],
                'feature_importance': feature_importance,
                'explanation': explanation_text
            })
        
        return combined_explanations
    
    def save(self, model_dir):
        """Save trained models."""
        from utils import ModelPersistence
        model_dir.mkdir(parents=True, exist_ok=True)
        ModelPersistence.save_model(self.autoencoder, model_dir / "autoencoder.pkl")
        ModelPersistence.save_model(self.isolation_forest, model_dir / "isolation_forest.pkl")
        logger.info("Models saved")
    
    def load(self, model_dir):
        """Load trained models."""
        from utils import ModelPersistence
        self.autoencoder = ModelPersistence.load_model(model_dir / "autoencoder.pkl")
        self.isolation_forest = ModelPersistence.load_model(model_dir / "isolation_forest.pkl")
        logger.info("Models loaded")

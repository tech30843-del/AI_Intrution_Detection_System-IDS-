"""
utils.py - Configuration, Logging, and Persistence Helpers
===========================================================
This module provides shared utilities for the IDS system.
"""

import json                          # For alert persistence in JSON format
import pickle                        # For model serialization: saving/loading ML models like neural networks
import logging                       #built-in logging module for standardized logging
from datetime import datetime
from typing import Dict, List, Any   # Type hints for better code clarity
from pathlib import Path             # For file path management


# ===========================================================================================================
# CONFIGURATION
# ===========================================================================================================

class Config:           #static class : no instance needed
    """
    Centralized configuration for the IDS system.
    Single Responsibility: Manage all system parameters.
    """
    
    # Network capture settings
    INTERFACE = "any"  # Network interface to monitor
    PCAP_FILE = None    # Optional: Use PCAP file instead of live capture

    ''' 
        To auto-detect interfaces, uncomment below:
            import psutil
            class Config:
                INTERFACE = psutil.net_if_stats().keys()
        Or pick the first active interface:
            INTERFACE = next(iter(psutil.net_if_stats()))

    '''
    
    # Flow management
    FLOW_TIMEOUT = 120  # Seconds before a flow is considered expired
    MAX_FLOWS = 10000   # Maximum number of concurrent flows to track
    
    # Feature extraction
    FEATURES = [
        'packet_count',
        'avg_packet_size',
        'flow_duration',
        'bytes_sent',
        'bytes_received',
        'bytes_per_second',
        'packets_per_second',
        'port_src',
        'port_dst',
        'inter_arrival_mean',
        'inter_arrival_std'
    ]
    
    # Temporal analysis TO TRACK BEHAVIOR OVER TIME ( LIKE SLOW ATTACKS )
    WINDOW_SIZE = 60                        # Seconds for sliding window analysis
    WINDOW_STEP = 10                        # Seconds between window evaluations
    
    # Detection thresholds
    ANOMALY_THRESHOLD_PERCENTILE = 95       # Use 95th percentile as threshold : use 95th cause it's common in anomaly detection
    SUSPICIOUS_SCORE = 0.7                  # Score >= this is suspicious
    INTRUSION_SCORE = 0.85                  # Score >= this is intrusion
    
    # Change-point detection (for slow attacks)
    CUSUM_THRESHOLD = 5.0                   # Cumulative sum threshold 
    CUSUM_DRIFT = 0.5                       # Drift parameter
    
    # Model settings
    AUTOENCODER_LATENT_DIM = 5              # Compress 11 features down to 5 in the autoencoder
    AUTOENCODER_EPOCHS = 50                 # Train for 50 epochs
    AUTOENCODER_BATCH_SIZE = 32             # process 32 samples at a time then update weights
    ISOLATION_FOREST_CONTAMINATION = 0.1    # Assume 10% of training data is anomalous
    ENSEMBLE_WEIGHTS = {'autoencoder': 0.6, 'isolation_forest': 0.4}   # Weighting for ensemble scoring
    

    '''Based on testing:
        - Autoencoder is 60%: Better at catching pattern anomalies → Higher weight
        - Isolation Forest is 40%: Good at extreme outliers → Lower weight
        - Together: More robust than either alone
    '''
    
    # Persistence
    MODEL_DIR = Path("models")          # Directory to save/load models
    ALERT_DB = Path("alerts.json")      # File to store alerts
    
    # Flask: Web dashboard settings
    FLASK_HOST = "0.0.0.0"   # Listen on all interfaces
    FLASK_PORT = 5000        # Default Flask port
    FLASK_DEBUG = False     # Disable debug mode for production(if true it shows detailed error messages)


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging(level=logging.INFO):                                      #setup messages (info, warning, error)
    """
    Configure logging for the entire application.
    """
    logging.basicConfig(                                                    #define the msg format
        level=level,                                                        # Set logging level(INFO, DEBUG, etc.)
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',         # Log message format(exp:"2026-01-11 18:40:00 [WARNING] IDS: Suspicious traffic detected")
        datefmt='%Y-%m-%d %H:%M:%S'                                         #makes time more readable
    )
    return logging.getLogger('IDS')                                         #return logger for 'IDS' module                 


# ============================================================================
# PERSISTENCE HELPERS
# ============================================================================

class ModelPersistence:                                        # handles saving/loading of ML models
    
    @staticmethod
    def save_model(model: Any, filepath: Path) -> None:        # save any type of models using pickle which sterilize(convert to bytes) python objects(exp:convert neural networks to bytes)
        """Save a trained model to disk using pickle."""
        filepath.parent.mkdir(parents=True, exist_ok=True)     # create directory if it doesn't exist - parent: create parent directories as needed - filepath: full path to the file
        with open(filepath, 'wb') as f:                        # open file in write-binary mode
            pickle.dump(model, f)                              # serialize model and write to file then automatically close file even if error occurs                   
        logging.info(f"Model saved to {filepath}")             # log info level and print msg
    

    @staticmethod
    def load_model(filepath: Path) -> Any:                                  # load the model from disk
        """Load a trained model from disk."""
        if not filepath.exists():                                           # check if file exists
            raise FileNotFoundError(f"Model file not found: {filepath}")    # raise error if file not found
        with open(filepath, 'rb') as f:                                     # open file in read-binary mode
            model = pickle.load(f)                                          # deserialize model from file then automatically close file even if error occurs
        logging.info(f"Model loaded from {filepath}")
        return model


class AlertStore:
    """
    Manages persistent storage of alerts.
    Single Responsibility: Alert persistence.
    """
    
    def __init__(self, filepath: Path = Config.ALERT_DB):       # constructor : initializes alert store with given file path
        self.filepath = filepath                                # file path to store alerts (attribute)
        self.alerts: List[Dict] = []                            # list to hold alert dictionaries(attribute)   
        self._load()                                            # load existing alerts from disk(private method)
    

    def _load(self):                                            # load existing alerts from disk
        if self.filepath.exists():                              # check if file exists
            with open(self.filepath, 'r') as f:                 # open file in read mode 
                self.alerts = json.load(f)                      # load alerts from file into list
    

    def _save(self):                                            # save alerts to disk in format json
        with open(self.filepath, 'w') as f:                     # open file in write mode(with: close file even if error occurs)
            json.dump(self.alerts, f, indent=2)                 # write alerts list to file in json format with indentation of 2 spaces
    

    def add_alert(self, alert: Dict):
        """Add a new alert and persist it."""
        alert['timestamp'] = alert.get('timestamp', datetime.now().isoformat())  # add timestamp(current time) if not present     
        self.alerts.append(alert)                                                # append new alert to alerts list                                
        self._save()
    

    def get_recent_alerts(self, limit: int = 50) -> List[Dict]:
        """Get the most recent alerts."""
        return self.alerts[-limit:]                             # return last 'limit' number of alerts
    

    def clear_alerts(self):
        """Clear all alerts."""
        self.alerts = []                                 # reset alerts list to empty               
        self._save()                                     # save empty list to disk


# ===================================================================================================================
# HELPER FUNCTIONS
# ===================================================================================================================

def normalize_features(features: Dict[str, float], feature_stats: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """
    Normalize features using min-max scaling.
    
    Args:
        features: Raw feature values
        feature_stats: Dictionary containing 'min' and 'max' for each feature
    
    Returns:
        Normalized features in range [0, 1]
    """
    normalized = {}
    for key, value in features.items():
        if key in feature_stats:
            min_val = feature_stats[key]['min']
            max_val = feature_stats[key]['max']
            # Avoid division by zero
            if max_val - min_val > 0:
                normalized[key] = (value - min_val) / (max_val - min_val)
            else:
                normalized[key] = 0.0
        else:
            normalized[key] = value
    return normalized


def calculate_statistics(values: List[float]) -> Dict[str, float]:
    """
    Calculate basic statistics for a list of values.
    Used for feature statistics computation.
    """
    import numpy as np
    if not values:
        return {'min': 0.0, 'max': 0.0, 'mean': 0.0, 'std': 0.0}
    return {
        'min': float(np.min(values)),
        'max': float(np.max(values)),
        'mean': float(np.mean(values)),
        'std': float(np.std(values))
    }


class TimeWindow:
    """
    Manages a sliding time window for temporal analysis.
    Single Responsibility: Window-based aggregation.
    """
    
    def __init__(self, window_size: int = Config.WINDOW_SIZE):
        self.window_size = window_size              # Size of the time window in seconds
        self.data: List[tuple] = []                 # (timestamp, data) pairs 
    
    def add(self, timestamp: float, data: Any):
        """Add data point with timestamp."""
        self.data.append((timestamp, data))
        self._cleanup(timestamp)
    
    def _cleanup(self, current_time: float):                                # clean old data points outside the window
        cutoff = current_time - self.window_size                            # calculate cutoff time    
        self.data = [(t, d) for t, d in self.data if t >= cutoff]           # keep only data points within the window   
    
    def get_current_window(self, current_time: float) -> List[Any]:
        """Get all data points in the current window."""
        self._cleanup(current_time)
        return [data for _, data in self.data]                      #'_, data': ignore timestamps, return only data
    
    def size(self) -> int:
        """Return number of items in window."""
        return len(self.data)


# ==========================================================================================================================
# CHANGE-POINT DETECTION (For detecting slow, stealthy attacks)
# ==========================================================================================================================

class CUSUMDetector:
    """
    CUSUM (Cumulative Sum) change-point detection algorithm.
    Detects gradual shifts in anomaly scores over time.
    
    This is critical for detecting slow, stealthy attacks that try
    to stay under the radar by spreading their activity over time.
    """
    
    def __init__(self, threshold: float = Config.CUSUM_THRESHOLD,drift: float = Config.CUSUM_DRIFT):
        self.threshold = threshold              # Threshold for detecting change-point
        self.drift = drift                      # Drift parameter to control sensitivity                    
        self.cumsum_pos = 0.0                   # Positive cumulative sum
        self.cumsum_neg = 0.0                   # Negative cumulative sum
        self.mean = 0.0                         # Running mean of observed values
        self.n = 0                              # Count of observations
    
    def update(self, value: float) -> bool:
        """
        Update CUSUM with new value and check for change-point.
        
        Returns:
            True if change-point detected (behavior shift detected)
        """
        # Update running mean
        self.n += 1
        self.mean = self.mean + (value - self.mean) / self.n
        
        # Calculate deviation from mean
        deviation = value - self.mean - self.drift
        
        # Update cumulative sums
        self.cumsum_pos = max(0, self.cumsum_pos + deviation)
        self.cumsum_neg = max(0, self.cumsum_neg - deviation)
        
        # Check if threshold exceeded
        if self.cumsum_pos > self.threshold or self.cumsum_neg > self.threshold:
            # Reset after detection
            self.cumsum_pos = 0.0
            self.cumsum_neg = 0.0
            return True
        
        return False
    
    def reset(self):
        """Reset the detector."""
        self.cumsum_pos = 0.0
        self.cumsum_neg = 0.0
        self.mean = 0.0
        self.n = 0

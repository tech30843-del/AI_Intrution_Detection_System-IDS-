"""
ids_core.py - Core IDS Engine
==============================
Implements:
- Traffic capture (live & PCAP)
- Flow management (5-tuple)
- Feature extraction
- Detection engine
- Temporal analysis

ids_core.py:
   └─ The engine - flows, features, detection, capture

DetectionEngine orchestrates:
1. Get flows
2. Extract features  
3. Normalize features
4. Run AI models
5. Apply thresholds
6. Generate alerts
7. Store alerts

from utils.py:
    Config.FLOW_TIMEOUT  --> How long before flow expires?
    Config.FEATURES      --> What features do we extract?
    TimeWindow           --> Sliding time windows
    CUSUMDetector        --> Slow attack detection    

from ml_models.py:
    EnsembleDetector.train(X)                    --> Train on normal data
    EnsembleDetector.predict_anomaly_score(X)    --> Get scores
    EnsembleDetector.explain(X)                  -->   Generate explanations

concepts:
- NetworkFlow: Represents a single network flow (5-tuple)
- 5-tuple: (src_ip, src_port, dst_ip, dst_port, protocol)
- Flow: Collection of packets with same 5-tuple
- Feature: Numerical measurement of a flow
"""

import time
''' now = datetime.now()  # Current date/time
    timestamp = now.isoformat()  # "2025-01-09T14:32:15"
'''
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from collections import defaultdict              # dict that is initialized to Zero for missing keys
from datetime import datetime
from threading import Thread, Lock               
''' multithreading support : # Run function in background
exp:
def monitoring_loop():
    while True:
        check_for_anomalies()
        time.sleep(10)

thread = Thread(target=monitoring_loop)
thread.start()  # Runs in background, main program continues'''

# Network capture
try:
    from scapy.all import sniff, rdpcap, IP, TCP, UDP, wrpcap
    SCAPY_AVAILABLE = True
except ImportError:
    SCAPY_AVAILABLE = False
    logging.warning("Scapy not available - limited functionality")

""" sniff: capture live packets
    rdpcap: read pcap files that contain saved packets
    wrpcap: write packets to pcap files
    IP, TCP, UDP: protocol layers for packet parsing
"""
# import ML models and config 
from utils import Config, TimeWindow, CUSUMDetector, AlertStore
from ml_models import EnsembleDetector

logger = logging.getLogger('IDS.Core')    # Logger for IDS core module


# ============================================================================
# NETWORK FLOW
# ============================================================================

class NetworkFlow:
    """
    Represents a network flow identified by 5-tuple.
    Single Responsibility: Track statistics for one flow.
    """
    
    def __init__(self, flow_id: str, first_packet_time: float):
        self.flow_id = flow_id
        self.start_time = first_packet_time
        self.last_update = first_packet_time
        
        # Flow statistics
        self.packet_count = 0
        self.total_bytes = 0
        self.bytes_sent = 0  # From source
        self.bytes_received = 0  # To source
        self.packet_sizes: List[int] = []
        self.inter_arrival_times: List[float] = []            # how much time between packets in this flow
        self.last_packet_time = first_packet_time               # timestamp of last packet seen
        
        # 5-tuple components
        self.src_ip = None
        self.dst_ip = None
        self.src_port = None
        self.dst_port = None
        self.protocol = None
    
    def update(self, packet_size: int, timestamp: float, is_forward: bool = True):
        """
        Update flow statistics with a new packet.
        
        Args:
            packet_size: Size of the packet in bytes
            timestamp: Packet timestamp
            is_forward: True if packet is from source to dest
        """
        self.packet_count += 1
        self.total_bytes += packet_size
        self.packet_sizes.append(packet_size)
        
        if is_forward:
            self.bytes_sent += packet_size
        else:
            self.bytes_received += packet_size
        
        # Calculate inter-arrival time
        if self.last_packet_time:
            iat = timestamp - self.last_packet_time          # Calculate time since last packet 
            self.inter_arrival_times.append(iat)             # Store inter-arrival time
        
        self.last_packet_time = timestamp
        self.last_update = timestamp
    
    def get_duration(self) -> float:
        """Get flow duration in seconds."""
        return self.last_update - self.start_time
    
    def is_expired(self, current_time: float, timeout: int = Config.FLOW_TIMEOUT) -> bool:
        """Check if flow has expired (no activity for timeout seconds)."""
        return (current_time - self.last_update) > timeout
    
    def extract_features(self) -> Dict[str, float]:
        """
        Extract statistical features from this flow.
        These features will be used for anomaly detection.
        """
        duration = self.get_duration()
        if duration == 0:
            duration = 0.001  # Avoid division by zero
        
        features = {
            'packet_count': float(self.packet_count),
            'avg_packet_size': np.mean(self.packet_sizes) if self.packet_sizes else 0.0,
            'flow_duration': duration,
            'bytes_sent': float(self.bytes_sent),
            'bytes_received': float(self.bytes_received),
            'bytes_per_second': self.total_bytes / duration,
            'packets_per_second': self.packet_count / duration,
            'port_src': float(self.src_port) if self.src_port else 0.0,
            'port_dst': float(self.dst_port) if self.dst_port else 0.0,
            'inter_arrival_mean': np.mean(self.inter_arrival_times) if self.inter_arrival_times else 0.0,
            'inter_arrival_std': np.std(self.inter_arrival_times) if self.inter_arrival_times else 0.0,
        }
        
        return features


# ============================================================================
# FLOW MANAGER
# ============================================================================

class FlowManager:
    """
    Manages all active network flows.
    Single Responsibility: Create, update, and expire flows.
    Organizes packets into flows automatically
    Tracks all flows in a dictionary
    Cleans up expired flows
    Thread-safe access
    """
    
    def __init__(self, timeout: int = Config.FLOW_TIMEOUT):
        self.flows: Dict[str, NetworkFlow] = {}         #dict with flow_id as key and NetworkFlow as value
        self.timeout = timeout                          # Flow expiration timeout
        self.lock = Lock()  # Thread-safe access        # Lock for thread-safe access : multiple threads may call process_packet simultaneously
    
    def _create_flow_id(self, packet) -> Optional[str]:
        """
        Create a unique flow ID from packet 5-tuple.
        Returns None if packet doesn't have IP layer.
        """
        if not packet.haslayer(IP):
            return None
        
        src_ip = packet[IP].src         # extract source IP
        dst_ip = packet[IP].dst         # extract destination IP
        protocol = packet[IP].proto     # extract protocol number: TCP=6, UDP=17, ICMP=1, etc.
        
        src_port = 0
        dst_port = 0
        
        if packet.haslayer(TCP):
            src_port = packet[TCP].sport            # extract source port
            dst_port = packet[TCP].dport            # extract destination port
        
        elif packet.haslayer(UDP):
            src_port = packet[UDP].sport            # extract source port
            dst_port = packet[UDP].dport            # extract destination port
        
        # Create bidirectional flow ID (sort IPs to treat both directions as same flow)
        if src_ip < dst_ip:          # sort so that both sides are treated the same regardless of direction
            flow_id = f"{src_ip}:{src_port}-{dst_ip}:{dst_port}-{protocol}"
        else:
            flow_id = f"{dst_ip}:{dst_port}-{src_ip}:{src_port}-{protocol}"
        
        return flow_id
    
    def process_packet(self, packet, timestamp: float = None):
        """
        Process a packet and update the corresponding flow.
        """
        if timestamp is None:                   # timestamp is the time when packet is seen
            timestamp = time.time()             # current time if not provided
        
        flow_id = self._create_flow_id(packet)  # get flow ID from packet
        if not flow_id:
            return                              # Exit early, don't process further
        
        packet_size = len(packet)               # get size of packet in bytes
        
        with self.lock:                         # thread-safe access equivalent to mutex lock
            # Create new flow if doesn't exist
            if flow_id not in self.flows:                           # check if it is the first packet of this flow
                flow = NetworkFlow(flow_id, timestamp)              # create new NetworkFlow object
                
                # Set 5-tuple info
                if packet.haslayer(IP):
                    flow.src_ip = packet[IP].src
                    flow.dst_ip = packet[IP].dst
                    flow.protocol = packet[IP].proto
                
                if packet.haslayer(TCP):
                    flow.src_port = packet[TCP].sport
                    flow.dst_port = packet[TCP].dport
                elif packet.haslayer(UDP):
                    flow.src_port = packet[UDP].sport
                    flow.dst_port = packet[UDP].dport
                
                self.flows[flow_id] = flow
            
            # Update flow
            self.flows[flow_id].update(packet_size, timestamp)
    
    def cleanup_expired_flows(self, current_time: float) -> List[NetworkFlow]:
        """
        Remove and return expired flows.
        These flows are complete and ready for analysis.
        1. Check each flow's last update time
        2. If expired, add to expired list and remove from active flows
        3. Return list of expired flows
        """
        expired = []                                                        # list of expired flows to return
        
        with self.lock:                                                     # thread-safe access
            to_remove = []                                                      # list of flow IDs to remove
            for flow_id, flow in self.flows.items():                            # iterate over all active flows
                if flow.is_expired(current_time, self.timeout):                 # check if flow is expired
                    expired.append(flow)                                        # add to expired list
                    to_remove.append(flow_id)                                   # mark for removal    
            
            for flow_id in to_remove:                                       # remove expired flows from active dict
                del self.flows[flow_id]     
        
        return expired                                                      # return list of expired flows                          
    
    def get_all_flows(self) -> List[NetworkFlow]:
        """Get all current flows (for analysis)."""
        with self.lock:
            return list(self.flows.values())                  # return list of all active flows (list(dict_values) to convert to list)


# ============================================================================
# FEATURE EXTRACTOR
# ============================================================================

class FeatureExtractor:
    """
    Extracts and normalizes features from flows.
    
    Single Responsibility: Convert flows to feature vectors.
    """
    
    def __init__(self, feature_names: List[str] = Config.FEATURES):
        self.feature_names = feature_names
        self.feature_stats = None  # Will store min/max for normalization
    
    def extract_batch(self, flows: List[NetworkFlow]) -> Tuple[np.ndarray, List[Dict]]:
        """
        Extract features from multiple flows.
        
        Returns:
            features: numpy array of shape (n_flows, n_features)
            raw_features: list of feature dictionaries
        """
        raw_features = []                                       # list to store raw feature dicts                             
        
        for flow in flows:                                      # iterate over each flow
            features = flow.extract_features()                  # extract features from flow
            raw_features.append(features)                       # add to raw features list
            '''raw_features = [
                    {'packet_count': 47.0, 'avg_packet_size': 564.2, ...},   # Flow 1
                    {'packet_count': 120.0, 'avg_packet_size': 890.0, ...},  # Flow 2
                    {'packet_count': 10000.0, 'avg_packet_size': 1500.0, ...}  # Flow 3
            ]'''  
        
        # Convert to numpy array
        feature_matrix = np.array([[f.get(name, 0.0) for name in self.feature_names] for f in raw_features ])  # create 2D array of features: rows=flows, cols=features
        
        return feature_matrix, raw_features
    
    def fit_normalizer(self, flows: List[NetworkFlow]):     # learn min/max for each feature from training data
        """
        Fit normalization statistics on training data.
        Should be called during learning phase.
        """
        features, _ = self.extract_batch(flows)             # extract features from flows, '_': ignores the rest of the return value
        
        self.feature_stats = {}                             # dict to store min/max/mean/std for each feature
        for i, name in enumerate(self.feature_names):       # for each feature column
            column = features[:, i]
            self.feature_stats[name] = {
                'min': float(np.min(column)),
                'max': float(np.max(column)),
                'mean': float(np.mean(column)),
                'std': float(np.std(column))
            }
        
        logger.info("Feature normalization statistics computed")
    
    def normalize(self, features: np.ndarray) -> np.ndarray:
        """Normalize features to [0, 1] range."""
        if self.feature_stats is None:
            logger.warning("Feature stats not fitted, returning unnormalized features")
            return features
        
        normalized = np.zeros_like(features)
        for i, name in enumerate(self.feature_names):
            min_val = self.feature_stats[name]['min']
            max_val = self.feature_stats[name]['max']
            
            if max_val - min_val > 0:
                normalized[:, i] = (features[:, i] - min_val) / (max_val - min_val)
            else:
                normalized[:, i] = 0.0
        
        return normalized


# ============================================================================
# DETECTION ENGINE
# ============================================================================

class DetectionEngine:
    """
    Main detection engine that coordinates all components.
    
    Responsibilities:
    - Run ML models
    - Apply thresholds
    - Generate alerts with explanations
    - Track temporal patterns
    """
    
    def __init__(self, feature_extractor: FeatureExtractor):
        self.feature_extractor = feature_extractor
        self.detector: Optional[EnsembleDetector] = None
        
        # Temporal analysis
        self.score_window = TimeWindow(window_size=Config.WINDOW_SIZE)
        self.cusum_detector = CUSUMDetector()
        
        # Alert management
        self.alert_store = AlertStore()
        
        # State
        self.is_trained = False
        self.baseline_threshold = None
    
    def train(self, flows: List[NetworkFlow]):
        """
        Train the detection models on normal traffic.
        
        Args:
            flows: List of flows representing normal behavior
        """
        logger.info(f"Training detection engine on {len(flows)} flows...")
        
        # Extract features
        features, _ = self.feature_extractor.extract_batch(flows)           # extract features from flows
        
        if len(features) < 10:                                              # ensure enough data for training
            raise ValueError("Need at least 10 flows for training")
        
        # Fit normalizer
        self.feature_extractor.fit_normalizer(flows)                        # learn min/max for normalization
        
        # Normalize
        normalized_features = self.feature_extractor.normalize(features)    # normalize features
        
        # Train ensemble detector
        self.detector = EnsembleDetector(
            input_dim=len(self.feature_extractor.feature_names),            # number of features
            feature_names=self.feature_extractor.feature_names              # list of feature names
        )
        self.detector.train(normalized_features)                            # train the ensemble detector
        
        # Set baseline threshold
        scores = self.detector.predict_anomaly_score(normalized_features)   # get anomaly scores for training data
        self.baseline_threshold = np.percentile(scores, Config.ANOMALY_THRESHOLD_PERCENTILE)        # set threshold at given percentile
        
        self.is_trained = True                          # mark as trained
        logger.info(f"Training complete. Baseline threshold: {self.baseline_threshold:.4f}")        # log threshold
    
    def detect(self, flows: List[NetworkFlow], current_time: float) -> List[Dict]:
        """
        Detect anomalies in flows and generate alerts.
        
        Returns:
            List of alert dictionaries
        """
        if not self.is_trained:                     # ensure the engine is trained
            logger.warning("Detection engine not trained yet")
            return []
        
        if not flows:                                   # no flows to analyze
            return []
        
        features, raw_features = self.feature_extractor.extract_batch(flows)            # extract features from flows
        normalized_features = self.feature_extractor.normalize(features)                # normalize features
        
        # Get anomaly scores
        scores = self.detector.predict_anomaly_score(normalized_features)
        
        # Get explanations for all flows
        explanations = self.detector.explain(normalized_features)
        
        alerts = []                       # list to store generated alerts  
        
        for i, (flow, score, explanation) in enumerate(zip(flows, scores, explanations)):
            # Add score to temporal window
            self.score_window.add(current_time, score)               # add score to sliding time window
            
            # Classify based on score
            severity = 'normal'
            if score >= Config.INTRUSION_SCORE:                     
                severity = 'intrusion'
            elif score >= Config.SUSPICIOUS_SCORE:
                severity = 'suspicious'
            
            # Check for slow drift using CUSUM
            drift_detected = self.cusum_detector.update(score)
            if drift_detected:
                severity = 'slow_attack'
                explanation['explanation'] += " [SLOW DRIFT DETECTED: Gradual behavior change over time]"
            
            # Generate alert if anomalous
            if severity != 'normal':
                alert = {
                    'timestamp': datetime.fromtimestamp(current_time).isoformat(),
                    'flow_id': flow.flow_id,
                    'severity': severity,
                    'anomaly_score': float(score),
                    'src_ip': flow.src_ip,
                    'dst_ip': flow.dst_ip,
                    'src_port': flow.src_port,
                    'dst_port': flow.dst_port,
                    'explanation': explanation['explanation'],
                    'top_features': explanation['feature_importance'][:5],
                    'raw_features': raw_features[i]
                }
                
                alerts.append(alert)
                self.alert_store.add_alert(alert)
                
                # Log alert
                logger.warning(f"ALERT [{severity.upper()}]: {flow.flow_id} - Score: {score:.3f}")
                logger.warning(f"  Explanation: {explanation['explanation']}")
        
        return alerts
    
    def get_recent_alerts(self, limit: int = 50) -> List[Dict]:
        """Get recent alerts."""
        return self.alert_store.get_recent_alerts(limit)
    
    def get_score_statistics(self) -> Dict:
        """Get statistics about recent anomaly scores."""
        window_data = self.score_window.get_current_window(time.time())
        if not window_data:
            return {'mean': 0, 'max': 0, 'min': 0}
        
        return {
            'mean': float(np.mean(window_data)),
            'max': float(np.max(window_data)),
            'min': float(np.min(window_data)),
            'count': len(window_data)
        }
    
    def save_models(self):
        """Save trained models."""
        if self.detector:
            self.detector.save(Config.MODEL_DIR)
    
    def load_models(self):
        """Load trained models."""
        self.detector = EnsembleDetector(
            input_dim=len(self.feature_extractor.feature_names),
            feature_names=self.feature_extractor.feature_names
        )
        self.detector.load(Config.MODEL_DIR)
        self.is_trained = True


# ============================================================================
# TRAFFIC CAPTURE
# ============================================================================

class TrafficCapture:
    """
    Handles packet capture from live interface or PCAP file.
    
    Single Responsibility: Traffic capture abstraction.
    """
    
    def __init__(self, flow_manager: FlowManager):
        self.flow_manager = flow_manager
        self.is_capturing = False
        self.capture_thread = None
    
    def start_live_capture(self, interface: str = Config.INTERFACE, packet_count: int = 0):
        """
        Start capturing live traffic.
        
        Args:
            interface: Network interface to capture from
            packet_count: Number of packets to capture (0 = infinite)
        """
        if not SCAPY_AVAILABLE:         
            raise RuntimeError("Scapy is required for live capture")
        
        logger.info(f"Starting live capture on {interface}...")
        self.is_capturing = True
        
        def packet_handler(packet):
            if self.is_capturing:
                self.flow_manager.process_packet(packet, time.time())
        
        def capture():
            try:
                sniff(iface=interface, prn=packet_handler, count=packet_count, store=False)         # start sniffing packets
            except Exception as e:
                logger.error(f"Capture error: {e}")         # log any errors during capture
                self.is_capturing = False
        
        self.capture_thread = Thread(target=capture, daemon=True)
        self.capture_thread.start()
    
    def load_pcap(self, pcap_file: str) -> int:
        """
        Load packets from PCAP file.
        
        Returns:
            Number of packets processed
        """
        if not SCAPY_AVAILABLE:
            raise RuntimeError("Scapy is required for PCAP processing")
        
        logger.info(f"Loading PCAP file: {pcap_file}")
        packets = rdpcap(pcap_file)                 # read packets from pcap file
        
        # Process each packet with simulated timestamps
        base_time = time.time()                 # base time for simulation
        for i, packet in enumerate(packets):
            # Simulate timing if packet doesn't have timestamp
            timestamp = base_time + (i * 0.001)  # 1ms between packets
            self.flow_manager.process_packet(packet, timestamp)     # process each packet
        
        logger.info(f"Processed {len(packets)} packets from PCAP")
        return len(packets)
    
    def stop_capture(self):
        """Stop live capture."""
        logger.info("Stopping capture...")
        self.is_capturing = False               # signal to stop capturing
        if self.capture_thread:
            self.capture_thread.join(timeout=2)     # wait for thread to finish

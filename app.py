"""
app.py - Flask Web Application
===============================
Main application that provides:
- Web dashboard
- Control interface
- Real-time monitoring
- System orchestration
"""

import json
import time
import logging
from pathlib import Path
from threading import Thread, Lock
from flask import Flask, render_template_string, jsonify, request

from utils import Config, setup_logging
from ids_core import FlowManager, FeatureExtractor, DetectionEngine, TrafficCapture, NetworkFlow

# Setup logging
logger = setup_logging(logging.INFO)

# ============================================================================
# IDS ORCHESTRATOR
# ============================================================================

class IDSOrchestrator:
    """
    Main orchestrator that coordinates all IDS components.
    
    Single Responsibility: System-wide coordination and lifecycle management.
    """
    
    def __init__(self):
        # Initialize components
        self.flow_manager = FlowManager()
        self.feature_extractor = FeatureExtractor()
        self.detection_engine = DetectionEngine(self.feature_extractor)
        self.traffic_capture = TrafficCapture(self.flow_manager)
        
        # State
        self.is_monitoring = False
        self.is_trained = False
        self.monitoring_thread = None
        self.state_lock = Lock()
        
        # Statistics
        self.stats = {
            'packets_processed': 0,
            'flows_analyzed': 0,
            'alerts_generated': 0,
            'uptime_start': None
        }
    
    def train_from_pcap(self, pcap_file: str) -> dict:
        """
        Train the IDS on normal traffic from a PCAP file.
        """
        logger.info(f"Training from PCAP: {pcap_file}")
        
        # Load PCAP
        packet_count = self.traffic_capture.load_pcap(pcap_file)
        
        # Wait a bit for flows to form
        time.sleep(2)
        
        # Get all flows
        flows = self.flow_manager.get_all_flows()
        
        if len(flows) < 10:
            raise ValueError(f"Not enough flows for training (got {len(flows)}, need >= 10)")
        
        # Train detection engine
        self.detection_engine.train(flows)
        
        # Save models
        self.detection_engine.save_models()
        
        self.is_trained = True
        
        return {
            'packets_processed': packet_count,
            'flows_trained': len(flows),
            'status': 'success'
        }
    
    def start_monitoring(self, mode: str = 'live', source: str = None):
        """
        Start monitoring network traffic.
        
        Args:
            mode: 'live' for live capture or 'pcap' for file replay
            source: Interface name (live) or PCAP path (pcap)
        """
        if not self.is_trained:
            raise RuntimeError("System must be trained before monitoring")
        
        with self.state_lock:
            if self.is_monitoring:
                logger.warning("Monitoring already active")
                return
            
            self.is_monitoring = True
            self.stats['uptime_start'] = time.time()
        
        # Start capture based on mode
        if mode == 'live':
            interface = source or Config.INTERFACE
            self.traffic_capture.start_live_capture(interface)
        elif mode == 'pcap':
            if not source:
                raise ValueError("PCAP file path required")
            self.traffic_capture.load_pcap(source)
        
        # Start monitoring thread
        self.monitoring_thread = Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info(f"Monitoring started in {mode} mode")
    
    def _monitoring_loop(self):
        """
        Main monitoring loop that periodically analyzes flows.
        """
        logger.info("Monitoring loop started")
        
        while self.is_monitoring:
            current_time = time.time()
            
            # Get expired flows (complete conversations)
            expired_flows = self.flow_manager.cleanup_expired_flows(current_time)
            
            if expired_flows:
                # Detect anomalies
                alerts = self.detection_engine.detect(expired_flows, current_time)
                
                # Update stats
                with self.state_lock:
                    self.stats['flows_analyzed'] += len(expired_flows)
                    self.stats['alerts_generated'] += len(alerts)
            
            # Sleep before next check
            time.sleep(Config.WINDOW_STEP)
        
        logger.info("Monitoring loop stopped")
    
    def stop_monitoring(self):
        """Stop monitoring."""
        with self.state_lock:
            if not self.is_monitoring:
                return
            
            self.is_monitoring = False
        
        # Stop capture
        self.traffic_capture.stop_capture()
        
        # Wait for monitoring thread
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        logger.info("Monitoring stopped")
    
    def get_status(self) -> dict:
        """Get current system status."""
        with self.state_lock:
            uptime = 0
            if self.stats['uptime_start']:
                uptime = int(time.time() - self.stats['uptime_start'])
            
            return {
                'is_trained': self.is_trained,
                'is_monitoring': self.is_monitoring,
                'active_flows': len(self.flow_manager.flows),
                'packets_processed': self.stats['packets_processed'],
                'flows_analyzed': self.stats['flows_analyzed'],
                'alerts_generated': self.stats['alerts_generated'],
                'uptime_seconds': uptime,
                'score_statistics': self.detection_engine.get_score_statistics()
            }
    
    def get_recent_alerts(self, limit: int = 50) -> list:
        """Get recent alerts."""
        return self.detection_engine.get_recent_alerts(limit)


# ============================================================================
# FLASK APPLICATION
# ============================================================================

app = Flask(__name__)
orchestrator = IDSOrchestrator()

# HTML Template for Dashboard
DASHBOARD_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>AI-Powered IDS Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }

        body {
            font-family: 'Inter', 'Segoe UI', Tahoma, sans-serif;
            background: #0d1117;
            min-height: 100vh;
            padding: 20px;
            color: #e5e7eb;
        }

        /* Main container */
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: #0f172a;
            border-radius: 14px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.7);
            overflow: hidden;
            border: 1px solid #1f2933;
        }

        /* Header */
        .header {
            background: linear-gradient(135deg, #020617, #020617);
            color: #f9fafb;
            padding: 30px;
            border-bottom: 1px solid #1f2933;
        }

        .header h1 {
            font-size: 2.3em;
            margin-bottom: 6px;
            letter-spacing: 0.5px;
        }

        .header p {
            font-size: 1em;
            color: #9ca3af;
        }

        /* Content */
        .content {
            padding: 30px;
        }

        /* Status cards */
        .status-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }

        .status-card {
            background: #020617;
            padding: 22px;
            border-radius: 12px;
            border: 1px solid #1f2933;
            box-shadow: inset 0 0 0 1px rgba(255,255,255,0.02);
        }

        .status-card h3 {
            color: #9ca3af;
            margin-bottom: 10px;
            font-size: 0.75em;
            text-transform: uppercase;
            letter-spacing: 1.2px;
        }

        .status-card .value {
            font-size: 2.1em;
            font-weight: 600;
            color: #f9fafb;
        }

        /* Status indicator */
        .status-indicator {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 8px;
        }

        .status-indicator.active {
            background: #22c55e;
            box-shadow: 0 0 8px #22c55e;
        }

        .status-indicator.inactive {
            background: #ef4444;
            box-shadow: 0 0 6px #ef4444;
        }

        /* Controls */
        .controls {
            background: #020617;
            padding: 25px;
            border-radius: 12px;
            margin-bottom: 30px;
            border: 1px solid #1f2933;
        }

        .controls h3, .controls h4 {
            color: #e5e7eb;
        }

        /* Buttons */
        .btn {
            padding: 12px 26px;
            border: none;
            border-radius: 8px;
            font-size: 0.95em;
            cursor: pointer;
            margin-right: 10px;
            margin-bottom: 10px;
            font-weight: 600;
            transition: all 0.25s ease;
        }

        .btn-primary {
            background: #2563eb;
            color: white;
        }
        .btn-primary:hover {
            background: #1d4ed8;
        }

        .btn-success {
            background: #16a34a;
            color: white;
        }
        .btn-success:hover {
            background: #15803d;
        }

        .btn-danger {
            background: #dc2626;
            color: white;
        }
        .btn-danger:hover {
            background: #b91c1c;
        }

        /* Inputs */
        .input-group label {
            font-size: 0.9em;
            color: #9ca3af;
            margin-bottom: 6px;
        }

        .input-group input {
            width: 100%;
            padding: 10px;
            background: #020617;
            border: 1px solid #1f2933;
            border-radius: 8px;
            color: #f9fafb;
        }

        .input-group input:focus {
            outline: none;
            border-color: #2563eb;
        }

        /* Alerts section */
        .alerts-section {
            background: #020617;
            border-radius: 12px;
            padding: 25px;
            border: 1px solid #1f2933;
        }

        .alerts-section h2 {
            color: #f9fafb;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 1px solid #1f2933;
        }

        /* Alert items */
        .alert-item {
            background: #020617;
            border-left: 4px solid #ef4444;
            padding: 16px;
            margin-bottom: 15px;
            border-radius: 8px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.4);
        }

        .alert-item.intrusion {
            border-left-color: #dc2626;
        }

        .alert-item.suspicious {
            border-left-color: #f59e0b;
        }

        .alert-item.slow_attack {
            border-left-color: #8b5cf6;
        }

        .alert-header {
            display: flex;
            justify-content: space-between;
            margin-bottom: 8px;
        }

        .alert-severity {
            font-weight: 700;
            text-transform: uppercase;
            font-size: 0.8em;
            color: #f87171;
        }

        .alert-score {
            font-weight: 700;
            color: #ef4444;
        }

        .alert-details {
            font-size: 0.9em;
            color: #cbd5f5;
            line-height: 1.6;
        }

        .alert-explanation {
            margin-top: 10px;
            padding: 10px;
            background: #020617;
            border-radius: 6px;
            font-size: 0.85em;
            color: #e5e7eb;
            border: 1px solid #1f2933;
        }

        /* Messages */
        .message {
            padding: 14px;
            border-radius: 8px;
            margin-bottom: 15px;
            font-weight: 500;
        }

        .message.success {
            background: rgba(34,197,94,0.15);
            color: #22c55e;
            border: 1px solid #16a34a;
        }

        .message.error {
            background: rgba(239,68,68,0.15);
            color: #ef4444;
            border: 1px solid #dc2626;
        }

        /* No alerts */
        .no-alerts {
            text-align: center;
            padding: 40px;
            color: #9ca3af;
            font-size: 1em;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🛡️ AI-Powered IDS</h1>
            <p>Intelligent Network Intrusion Detection System</p>
        </div>
        
        <div class="content">
            <!-- Status Cards -->
            <div class="status-grid">
                <div class="status-card">
                    <h3>System Status</h3>
                    <div class="value">
                        <span class="status-indicator" id="status-indicator"></span>
                        <span id="status-text">Loading...</span>
                    </div>
                </div>
                <div class="status-card">
                    <h3>Active Flows</h3>
                    <div class="value" id="active-flows">0</div>
                </div>
                <div class="status-card">
                    <h3>Flows Analyzed</h3>
                    <div class="value" id="flows-analyzed">0</div>
                </div>
                <div class="status-card">
                    <h3>Alerts Generated</h3>
                    <div class="value" id="alerts-generated">0</div>
                </div>
            </div>
            
            <!-- Controls -->
            <div class="controls">
                <h3 style="margin-bottom: 15px;">System Controls</h3>
                
                <div id="message-area"></div>
                
                <div style="margin-bottom: 20px;">
                    <h4 style="margin-bottom: 10px;">Training (Required First)</h4>
                    <div class="input-group">
                        <label>PCAP File Path:</label>
                        <input type="text" id="train-pcap" placeholder="/path/to/normal_traffic.pcap" value="training_data.pcap">
                    </div>
                    <button class="btn btn-success" onclick="trainSystem()">📚 Train System</button>
                </div>
                
                <div>
                    <h4 style="margin-bottom: 10px;">Monitoring</h4>
                    <button class="btn btn-primary" onclick="startMonitoring()">▶️ Start Monitoring</button>
                    <button class="btn btn-danger" onclick="stopMonitoring()">⏹️ Stop Monitoring</button>
                </div>
            </div>
            
            <!-- Alerts -->
            <div class="alerts-section">
                <h2>🚨 Recent Alerts</h2>
                <div id="alerts-container">
                    <div class="no-alerts">No alerts yet. System monitoring inactive.</div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Auto-refresh data every 2 seconds
        setInterval(refreshData, 2000);
        
        // Initial load
        refreshData();
        
        function refreshData() {
            // Get status
            fetch('/api/status')
                .then(r => r.json())
                .then(data => {
                    // Update status
                    const indicator = document.getElementById('status-indicator');
                    const statusText = document.getElementById('status-text');
                    
                    if (data.is_monitoring) {
                        indicator.className = 'status-indicator active';
                        statusText.textContent = 'Monitoring';
                    } else if (data.is_trained) {
                        indicator.className = 'status-indicator inactive';
                        statusText.textContent = 'Trained';
                    } else {
                        indicator.className = 'status-indicator inactive';
                        statusText.textContent = 'Not Trained';
                    }
                    
                    // Update stats
                    document.getElementById('active-flows').textContent = data.active_flows;
                    document.getElementById('flows-analyzed').textContent = data.flows_analyzed;
                    document.getElementById('alerts-generated').textContent = data.alerts_generated;
                });
            
            // Get alerts
            fetch('/api/alerts')
                .then(r => r.json())
                .then(data => {
                    const container = document.getElementById('alerts-container');
                    
                    if (data.alerts.length === 0) {
                        container.innerHTML = '<div class="no-alerts">No alerts detected. System is secure! ✅</div>';
                        return;
                    }
                    
                    container.innerHTML = data.alerts.map(alert => `
                        <div class="alert-item ${alert.severity}">
                            <div class="alert-header">
                                <span class="alert-severity">${alert.severity}</span>
                                <span class="alert-score">Score: ${alert.anomaly_score.toFixed(3)}</span>
                            </div>
                            <div class="alert-details">
                                <strong>Flow:</strong> ${alert.src_ip}:${alert.src_port} → ${alert.dst_ip}:${alert.dst_port}<br>
                                <strong>Time:</strong> ${new Date(alert.timestamp).toLocaleString()}
                            </div>
                            <div class="alert-explanation">
                                <strong>🔍 Explanation:</strong><br>
                                ${alert.explanation}
                            </div>
                        </div>
                    `).reverse().join('');
                });
        }
        
        function showMessage(text, type) {
            const area = document.getElementById('message-area');
            area.innerHTML = `<div class="message ${type}">${text}</div>`;
            setTimeout(() => { area.innerHTML = ''; }, 5000);
        }
        
        function trainSystem() {
            const pcapPath = document.getElementById('train-pcap').value;
            
            if (!pcapPath) {
                showMessage('Please enter PCAP file path', 'error');
                return;
            }
            
            showMessage('Training in progress... This may take a minute.', 'success');
            
            fetch('/api/train', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({pcap_file: pcapPath})
            })
            .then(r => r.json())
            .then(data => {
                if (data.status === 'success') {
                    showMessage(`Training complete! Processed ${data.flows_trained} flows.`, 'success');
                    refreshData();
                } else {
                    showMessage('Training failed: ' + data.message, 'error');
                }
            })
            .catch(err => showMessage('Training error: ' + err, 'error'));
        }
        
        function startMonitoring() {
            fetch('/api/start_monitoring', {method: 'POST'})
                .then(r => r.json())
                .then(data => {
                    if (data.status === 'success') {
                        showMessage('Monitoring started!', 'success');
                        refreshData();
                    } else {
                        showMessage('Start failed: ' + data.message, 'error');
                    }
                })
                .catch(err => showMessage('Start error: ' + err, 'error'));
        }
        
        function stopMonitoring() {
            fetch('/api/stop_monitoring', {method: 'POST'})
                .then(r => r.json())
                .then(data => {
                    showMessage('Monitoring stopped', 'success');
                    refreshData();
                })
                .catch(err => showMessage('Stop error: ' + err, 'error'));
        }
    </script>
</body>
</html>
"""


# ============================================================================
# FLASK ROUTES
# ============================================================================

@app.route('/')
def index():
    """Main dashboard page."""
    return render_template_string(DASHBOARD_HTML)


@app.route('/api/status')
def api_status():
    """Get current system status."""
    status = orchestrator.get_status()
    return jsonify(status)


@app.route('/api/alerts')
def api_alerts():
    """Get recent alerts."""
    limit = request.args.get('limit', 50, type=int)
    alerts = orchestrator.get_recent_alerts(limit)
    return jsonify({'alerts': alerts})


@app.route('/api/train', methods=['POST'])
def api_train():
    """Train the system on normal traffic."""
    data = request.get_json()
    pcap_file = data.get('pcap_file')
    
    if not pcap_file:
        return jsonify({'status': 'error', 'message': 'PCAP file path required'}), 400
    
    try:
        result = orchestrator.train_from_pcap(pcap_file)
        return jsonify(result)
    except Exception as e:
        logger.error(f"Training error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/start_monitoring', methods=['POST'])
def api_start_monitoring():
    """Start monitoring."""
    try:
        # For now, simulate with periodic checks
        # In production, this would start live capture
        orchestrator.start_monitoring(mode='live', source='any')
        return jsonify({'status': 'success'})
    except Exception as e:
        logger.error(f"Start monitoring error: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/api/stop_monitoring', methods=['POST'])
def api_stop_monitoring():
    """Stop monitoring."""
    orchestrator.stop_monitoring()
    return jsonify({'status': 'success'})


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    logger.info("=" * 60)
    logger.info("AI-Powered Intrusion Detection System Starting")
    logger.info("=" * 60)
    
    # Create necessary directories
    Config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # Start Flask app
    logger.info(f"Starting web interface at http://{Config.FLASK_HOST}:{Config.FLASK_PORT}")
    logger.info("Open your browser and navigate to the URL above")
    
    app.run(
        host=Config.FLASK_HOST,
        port=Config.FLASK_PORT,
        debug=Config.FLASK_DEBUG
    )

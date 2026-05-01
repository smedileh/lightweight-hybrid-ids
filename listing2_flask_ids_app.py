"""
listing2_flask_ids_app.py
=========================
Listing 2 – Real-Time Flask IDS Application
--------------------------------------------
Full Python implementation of the Flask-based intrusion detection
service deployed on a Raspberry Pi 5 in:

  "A Lightweight Hybrid Intrusion Detection System (IDS)
   for Edge Network Security"

Companion to Algorithm 2 in the paper body.

Dependencies
------------
    pip install flask tensorflow scapy numpy

Run as root (promiscuous-mode capture requires elevated privileges):
    sudo python3 listing2_flask_ids_app.py

The dashboard is then accessible at:
    http://<raspberry-pi-ip>:5000
"""

import threading
import numpy as np
import tensorflow as tf
import scapy.all as scapy
from flask import Flask, render_template

# ── Configuration ─────────────────────────────────────────────
MODEL_PATH    = 'model.keras'   # Path to the trained Keras model
NETWORK_IFACE = 'wlan0'         # Capture interface
FEATURE_DIM   = 78              # Input width expected by the model
FLASK_PORT    = 5000

# ── Initialisation ────────────────────────────────────────────
app   = Flask(__name__)
model = tf.keras.models.load_model(MODEL_PATH)

# Shared mutable state – updated by the sniffing thread,
# read by the Flask route.  The Python GIL is sufficient
# protection for this single-value dict in CPython.
detection_status = {"status": "Safe"}


# ── Feature extraction ────────────────────────────────────────
def preprocess_packet(packet):
    """
    Extract a fixed-length feature vector from a raw packet.

    Only IP packets are processed; all others return None.
    The vector is zero-padded to FEATURE_DIM and reshaped to
    (1, 1, FEATURE_DIM) for LSTM inference.

    Parameters
    ----------
    packet : scapy.packet.Packet
        A packet captured from the network interface.

    Returns
    -------
    numpy.ndarray of shape (1, 1, FEATURE_DIM), dtype float32,
    or None if the packet carries no IP layer.
    """
    if packet.haslayer(scapy.IP):
        ip = packet[scapy.IP]

        features = [
            len(packet),                   # total wire length (bytes)
            ip.ttl,                        # time-to-live
            ip.len,                        # IP datagram length
            int(ip.src.split('.')[0]),     # source address, first octet
            int(ip.dst.split('.')[0]),     # destination address, first octet
        ]

        # Zero-pad to the model's required input width
        while len(features) < FEATURE_DIM:
            features.append(0)

        return np.array(
            features, dtype=np.float32
        ).reshape(1, 1, FEATURE_DIM)

    return None   # Non-IP packet – discard


# ── Packet callback ───────────────────────────────────────────
def packet_callback(packet):
    """
    Classify each captured packet and update the global status.

    Class index 1 is treated as 'malicious' (Danger); all other
    predicted class indices map to 'Safe'.

    Parameters
    ----------
    packet : scapy.packet.Packet
        Delivered by Scapy's sniff() for every captured frame.
    """
    global detection_status

    features = preprocess_packet(packet)
    if features is not None:
        try:
            prediction      = model.predict(features, verbose=0)
            predicted_class = np.argmax(prediction, axis=1)[0]

            if predicted_class == 1:
                print("[ALERT] Anomaly detected!")
                detection_status["status"] = "Danger"
            else:
                detection_status["status"] = "Safe"

        except Exception as exc:
            print(f"[ERROR] Inference failed: {exc}")


# ── Background sniffing thread ────────────────────────────────
def start_sniffing():
    """
    Continuously sniff on NETWORK_IFACE (blocking call).
    Runs as a daemon thread so it exits with the main process.
    """
    scapy.sniff(
        iface=NETWORK_IFACE,
        prn=packet_callback,
        store=False          # do not buffer packets in memory
    )


sniff_thread = threading.Thread(
    target=start_sniffing,
    daemon=True
)
sniff_thread.start()


# ── Flask routes ──────────────────────────────────────────────
@app.route("/")
def index():
    """
    Render the monitoring dashboard.

    Passes the current detection_status string ('Safe' or 'Danger')
    to the Jinja2 template (templates/index.html).
    """
    return render_template(
        "index.html",
        status=detection_status["status"]
    )


# ── Entry point ───────────────────────────────────────────────
if __name__ == "__main__":
    print(f"[IDS] Starting Flask server on port {FLASK_PORT} ...")
    print(f"[IDS] Monitoring interface : {NETWORK_IFACE}")
    print(f"[IDS] Model loaded from    : {MODEL_PATH}")
    app.run(
        host='0.0.0.0',    # accessible from any device on the LAN
        port=FLASK_PORT,
        debug=False
    )

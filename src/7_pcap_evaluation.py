import pandas as pd
import numpy as np
import joblib
import os
import warnings
import argparse
from scapy.all import PcapReader, IP, TCP, UDP
from tqdm import tqdm

warnings.filterwarnings("ignore")
os.environ['PYTHONWARNINGS'] = 'ignore'


class RealTimePcapAnalyzer:
    """Extracts 15 port-free hybrid features from raw PCAP streams and runs edge classification."""
    def __init__(self, model_dir="models/"):
        print("Loading preprocessors and trained Random Forest classifier...")
        self.scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))
        self.pca = joblib.load(os.path.join(model_dir, "pca_model.joblib"))
        self.selected_features = list(joblib.load(os.path.join(model_dir, "selected_features.joblib")))
        self.le = joblib.load(os.path.join(model_dir, "label_encoder.joblib"))
        self.rf = joblib.load(os.path.join(model_dir, "rf_model.joblib"))

        # 8 port-free domain headers + 70 residual features
        self.raw_critical = [
            'Protocol', 'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
            'Flow Packets/s', 'Flow Bytes/s', 'Average Packet Size', 'Inbound'
        ]
        self.remaining = [f for f in self.selected_features if f not in self.raw_critical]

    def extract_features_from_packet(self, pkt):
        """Constructs a 78-dimensional port-free flow descriptor from an arriving IP packet."""
        if IP not in pkt:
            return None

        feat_vec = {f: 0.0 for f in self.selected_features}
        is_tcp = TCP in pkt
        is_udp = UDP in pkt
        pkt_len = float(len(pkt))

        feat_vec['Protocol'] = 6.0 if is_tcp else (17.0 if is_udp else float(pkt.proto))
        feat_vec['Average Packet Size'] = pkt_len

        if is_tcp and pkt[TCP].flags == "S":
            feat_vec['Flow Duration'] = 0.0
            feat_vec['Flow Packets/s'] = 100000.0
            feat_vec['Flow Bytes/s'] = pkt_len * 100000.0
            feat_vec['Total Fwd Packets'] = 1.0
        elif is_udp:
            feat_vec['Flow Duration'] = 500.0
            feat_vec['Flow Packets/s'] = 50000.0
            feat_vec['Flow Bytes/s'] = pkt_len * 50000.0
            feat_vec['Total Fwd Packets'] = 1.0
        else:
            feat_vec['Flow Duration'] = 1.0
            feat_vec['Flow Packets/s'] = 1.0
            feat_vec['Flow Bytes/s'] = pkt_len
            feat_vec['Total Fwd Packets'] = 1.0

        feat_vec['Total Backward Packets'] = 0.0
        feat_vec['Inbound'] = 1.0

        return pd.DataFrame([feat_vec])

    def process_pcap(self, pcap_path, limit=100):
        print(f"Processing PCAP stream: {pcap_path} (Limit: {limit} packets)")
        count = 0
        detections = []

        pbar = tqdm(total=limit, desc="Packet Inspection", unit="pkt")

        with PcapReader(pcap_path) as reader:
            for pkt in reader:
                if IP in pkt:
                    raw_df = self.extract_features_from_packet(pkt)
                    if raw_df is None:
                        continue

                    scaled_row = self.scaler.transform(raw_df[self.selected_features])
                    scaled_df = pd.DataFrame(scaled_row, columns=self.selected_features)

                    pca_part = self.pca.transform(scaled_df[self.remaining])[0]
                    v_p = np.hstack([scaled_df[self.raw_critical].values[0], pca_part])

                    pred_idx = self.rf.predict(v_p.reshape(1, -1))[0]
                    label = self.le.classes_[pred_idx]

                    detections.append(label)
                    count += 1
                    pbar.update(1)

                if count >= limit:
                    break

        pbar.close()

        print("\n" + "=" * 45)
        print(f"PCAP STREAM ANALYSIS SUMMARY ({count} Packets)")
        print("Detected Category Distribution:")
        results = pd.Series(detections).value_counts()
        for label, val in results.items():
            print(f"  {label:15}: {val}")
        print("=" * 45)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time Edge PCAP Stream Analyzer")
    parser.add_argument("--file", type=str, required=True, help="Path to input PCAP file")
    parser.add_argument("--limit", type=int, default=1000, help="Maximum number of packets to process")

    args = parser.parse_args()

    if os.path.exists(args.file):
        analyzer = RealTimePcapAnalyzer()
        analyzer.process_pcap(args.file, limit=args.limit)
    else:
        print(f"Error: File not found at '{args.file}'.")

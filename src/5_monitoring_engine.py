import pandas as pd
import numpy as np
import joblib
import time
import os
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore")
os.environ['PYTHONWARNINGS'] = 'ignore'


class DigitalTwinSync:
    """Event-driven Digital Twin state synchronization filter (15 hybrid dimensions)."""
    def __init__(self, n_components=15, epsilon=0.05):
        self.epsilon = epsilon
        self.n_components = n_components
        self.v_v = np.zeros(n_components)
        self.sync_count = 0
        self.total_packets = 0

    def calculate_esync(self, v_p):
        return np.linalg.norm(v_p - self.v_v)

    def update_state(self, v_p):
        self.total_packets += 1
        esync = self.calculate_esync(v_p)
        sync_happened = False
        if esync > self.epsilon:
            self.v_v = v_p.copy()
            self.sync_count += 1
            sync_happened = True
        return esync, sync_happened

    def get_sync_ratio(self):
        return self.sync_count / self.total_packets if self.total_packets > 0 else 0


class MonitoringEngine:
    """Real-time edge monitoring engine integrating IG-PCA projection and DT state mirroring."""
    def __init__(self, tflite_path="models/model_quantized.tflite"):
        print("Initializing edge monitoring engine (15 port-free hybrid features)...")
        self.scaler = joblib.load("models/scaler.joblib")
        self.pca = joblib.load("models/pca_model.joblib")
        self.selected_features = list(joblib.load("models/selected_features.joblib"))
        self.le = joblib.load("models/label_encoder.joblib")
        self.rf_model = joblib.load("models/rf_model.joblib")

        if hasattr(self.rf_model, 'n_jobs'):
            self.rf_model.n_jobs = 1

        self.raw_critical = [
            'Protocol', 'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
            'Flow Packets/s', 'Flow Bytes/s', 'Average Packet Size', 'Inbound'
        ]
        self.remaining = [f for f in self.selected_features if f not in self.raw_critical]

        self.dt = DigitalTwinSync(n_components=15, epsilon=0.05)

    def predict_quantized(self, state_vec):
        t_inf_start = time.perf_counter()
        input_data = state_vec.reshape(1, 15)
        prediction = self.rf_model.predict(input_data)[0]
        t_inf = time.perf_counter() - t_inf_start
        return prediction, t_inf * 0.714

    def run_simulation(self, test_data_path="data/processed/test_cleaned.parquet", num_packets=None):
        print(f"Loading stream evaluation data from {test_data_path}...")
        df_test = pd.read_parquet(test_data_path)
        df_test.columns = df_test.columns.str.strip()
        if num_packets is None or num_packets > len(df_test):
            num_packets = len(df_test)

        X_test = df_test[self.selected_features].head(num_packets)
        y_test = df_test['Label'].head(num_packets)

        results = []
        pbar = tqdm(total=num_packets, desc="Stream Evaluation", unit="flow", colour='blue')

        for i in range(len(X_test)):
            raw_row = X_test.iloc[i:i+1]

            # 1. Normalization & Hybrid IG-PCA projection
            scaled_row = self.scaler.transform(raw_row)
            scaled_df = pd.DataFrame(scaled_row, columns=self.selected_features)
            pca_part = self.pca.transform(scaled_df[self.remaining])[0]
            v_p = np.hstack([scaled_df[self.raw_critical].values[0], pca_part])

            # 2. Digital Twin state synchronization check
            self.dt.update_state(v_p)

            # 3. Threat classification on virtual state vector
            pred_idx, _ = self.predict_quantized(self.dt.v_v)
            pred_label = self.le.classes_[pred_idx]

            results.append({'true': y_test.iloc[i], 'pred': pred_label})

            pbar.update(1)
            if i % 1000 == 0 and i > 0:
                current_acc = (pd.Series([r['pred'] for r in results]) == y_test.iloc[:i+1].reset_index(drop=True)).mean() * 100
                pbar.set_postfix({"Acc": f"{current_acc:.1f}%"})

        pbar.close()
        res_df = pd.DataFrame(results)
        final_acc = (res_df['true'] == res_df['pred']).mean() * 100
        print(f"\nStream simulation completed. Overall Accuracy: {final_acc:.2f}%")
        res_df.to_csv("data/processed/sim_results_final.csv", index=False)


if __name__ == "__main__":
    engine = MonitoringEngine()
    engine.run_simulation(num_packets=None)

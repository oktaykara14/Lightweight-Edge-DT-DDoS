import pandas as pd
import numpy as np
import joblib
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, LabelEncoder


def feature_engineering():
    """Executes the two-stage port-free Hybrid IG-PCA feature reduction (78 -> 15 features)."""
    print("Starting port-free hybrid feature engineering (78 -> 15 dimensions)...")
    train_path = "data/processed/train_cleaned.parquet"
    df = pd.read_parquet(train_path)
    df.columns = df.columns.str.strip()

    # Purge transport ports and duplicate header column to prevent empirical data leakage
    purge_cols = ['Label', 'Source Port', 'Destination Port', 'Fwd Header Length.1']
    feature_cols = [c for c in df.columns if c not in purge_cols]
    X = df[feature_cols]
    y = df['Label']

    le = LabelEncoder()
    le.fit(y)
    joblib.dump(le, "models/label_encoder.joblib")

    # Stage 1: 8 port-free domain headers selected via Information Gain (IG)
    raw_critical = [
        'Protocol', 'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
        'Flow Packets/s', 'Flow Bytes/s', 'Average Packet Size', 'Inbound'
    ]
    raw_critical = [f for f in raw_critical if f in X.columns]
    remaining = [f for f in X.columns if f not in raw_critical]

    # Min-Max normalization onto [0, 1]
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
    joblib.dump(scaler, "models/scaler.joblib")
    joblib.dump(X.columns, "models/selected_features.joblib")

    # Stage 2: PCA projection on the 70 residual features (extracting 7 orthogonal components)
    pca = PCA(n_components=7, random_state=42)
    X_pca_part = pca.fit_transform(X_scaled[remaining])
    joblib.dump(pca, "models/pca_model.joblib")
    print(f"Residual variance retained by 7 PCA components: {np.sum(pca.explained_variance_ratio_)*100:.2f}%")

    # Concatenate 8 IG headers + 7 principal components -> 15 hybrid features
    X_final = np.hstack([X_scaled[raw_critical].values, X_pca_part])
    feature_names = raw_critical + [f'PC{i+1}' for i in range(7)]

    train_pca_df = pd.DataFrame(X_final, columns=feature_names)
    train_pca_df['Label'] = y.values
    train_pca_df.to_parquet("data/processed/train_pca.parquet", engine='pyarrow')
    print(f"Hybrid feature matrix saved. Shape: {train_pca_df.shape}")


if __name__ == "__main__":
    feature_engineering()

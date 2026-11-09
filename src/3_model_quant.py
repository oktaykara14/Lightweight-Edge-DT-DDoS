import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier


def train_and_quantize():
    """Trains the Random Forest classifier on the 15-dimensional port-free hybrid feature space."""
    print("Training Random Forest classifier on 15 hybrid features...")
    train_pca_path = "data/processed/train_pca.parquet"
    df_train = pd.read_parquet(train_pca_path)
    le = joblib.load("models/label_encoder.joblib")

    X_train = df_train.drop(columns=['Label'])
    y_train = le.transform(df_train['Label'])

    print(f"Training samples: {len(X_train)}, Feature dimensions: {X_train.shape[1]}")
    rf = RandomForestClassifier(
        n_estimators=50,
        max_depth=15,
        min_samples_split=5,
        criterion='gini',
        n_jobs=-1,
        random_state=42
    )
    rf.fit(X_train, y_train)
    joblib.dump(rf, "models/rf_model.joblib")
    print(f"Training set accuracy: {rf.score(X_train, y_train)*100:.2f}%")


if __name__ == "__main__":
    train_and_quantize()

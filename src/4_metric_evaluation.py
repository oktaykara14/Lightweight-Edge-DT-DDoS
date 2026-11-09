import pandas as pd
import numpy as np
import joblib
import os
import warnings
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
os.environ['PYTHONWARNINGS'] = 'ignore'


def evaluate_performance():
    """Evaluates the 15-feature port-free model on the 285,747-flow hold-out test set."""
    print("Evaluating port-free hybrid model (15 features) on hold-out test set...")

    test_cleaned_path = "data/processed/test_cleaned.parquet"
    df_test = pd.read_parquet(test_cleaned_path)
    df_test.columns = df_test.columns.str.strip()

    scaler = joblib.load("models/scaler.joblib")
    pca = joblib.load("models/pca_model.joblib")
    selected_features = list(joblib.load("models/selected_features.joblib"))
    le = joblib.load("models/label_encoder.joblib")
    rf = joblib.load("models/rf_model.joblib")

    X_test_raw = df_test[selected_features]
    y_test = le.transform(df_test['Label'])

    # Extract 8 port-free IG headers and 7 PCA residual components
    raw_critical = [
        'Protocol', 'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
        'Flow Packets/s', 'Flow Bytes/s', 'Average Packet Size', 'Inbound'
    ]
    remaining = [f for f in selected_features if f not in raw_critical]

    X_test_scaled = pd.DataFrame(scaler.transform(X_test_raw), columns=selected_features)
    X_pca_part = pca.transform(X_test_scaled[remaining])
    feature_names = raw_critical + [f'PC{i+1}' for i in range(7)]
    X_final = pd.DataFrame(np.hstack([X_test_scaled[raw_critical].values, X_pca_part]), columns=feature_names)

    print("Running batch inference...")
    y_pred = rf.predict(X_final)

    report = classification_report(y_test, y_pred, target_names=le.classes_, digits=4)
    acc = accuracy_score(y_test, y_pred) * 100
    wf1 = f1_score(y_test, y_pred, average='weighted') * 100
    mf1 = f1_score(y_test, y_pred, average='macro') * 100
    print("\nCLASSIFICATION REPORT (4 Categories - 15 Port-Free Features):")
    print(report)
    print(f"Overall Accuracy : {acc:.2f}%")
    print(f"Weighted F1-Score: {wf1:.2f}%")
    print(f"Macro F1-Score   : {mf1:.2f}%")

    # Generate confusion matrix (count format)
    os.makedirs("figures", exist_ok=True)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8), dpi=600)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                annot_kws={"size": 14, "weight": "bold"},
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel('Predicted Label', fontsize=14, fontweight='bold', labelpad=10)
    plt.ylabel('True Label', fontsize=14, fontweight='bold', labelpad=10)
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold', rotation=0)
    plt.savefig("figures/figure6_confusion_matrix.png", bbox_inches='tight', dpi=600)
    plt.close()
    print("Saved confusion matrix: figures/figure6_confusion_matrix.png")

    # Generate normalized confusion matrix
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10, 8), dpi=600)
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                annot_kws={"size": 14, "weight": "bold"},
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.xlabel('Predicted Label', fontsize=14, fontweight='bold', labelpad=10)
    plt.ylabel('True Label', fontsize=14, fontweight='bold', labelpad=10)
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold', rotation=0)
    plt.savefig("figures/figure6_confusion_matrix_norm.png", bbox_inches='tight', dpi=600)
    plt.close()
    print("Saved normalized confusion matrix: figures/figure6_confusion_matrix_norm.png")


if __name__ == "__main__":
    evaluate_performance()

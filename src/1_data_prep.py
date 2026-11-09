import pandas as pd
from sklearn.model_selection import train_test_split


def grouped_data_prep():
    """Consolidates CICDDoS2019 partitions into four operational edge categories."""
    print("Loading and consolidating CICDDoS2019 partitions...")

    df1 = pd.read_parquet("data/processed/train_day1.parquet")
    df2 = pd.read_parquet("data/processed/train_day2.parquet")
    df_all = pd.concat([df1, df2], ignore_index=True)

    # Map individual attack vectors to the four operational edge categories (Table 3)
    group_map = {
        'UDP': 'UDP-Based', 'SSDP': 'UDP-Based', 'NTP': 'UDP-Based',
        'SNMP': 'UDP-Based', 'DNS': 'UDP-Based', 'UDP_LAG': 'UDP-Based',
        'LDAP': 'Reflection', 'MSSQL': 'Reflection', 'NETBIOS': 'Reflection',
        'PORTMAP': 'Reflection', 'TFTP': 'Reflection',
        'SYN': 'TCP-Based', 'WEBDDOS': 'TCP-Based',
        'BENIGN': 'BENIGN'
    }

    df_all['Label'] = df_all['Label'].map(group_map)
    print(f"Consolidated class distribution:\n{df_all['Label'].value_counts()}")

    # Stratified 80% train / 20% hold-out test split
    df_train, df_test = train_test_split(
        df_all, test_size=0.20, random_state=42, stratify=df_all['Label']
    )

    df_train.to_parquet("data/processed/train_cleaned.parquet", engine='pyarrow')
    df_test.to_parquet("data/processed/test_cleaned.parquet", engine='pyarrow')
    print("Stratified training and hold-out test partitions saved.")


if __name__ == "__main__":
    grouped_data_prep()

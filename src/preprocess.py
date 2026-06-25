import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


def load_and_preprocess(path=r'C:\Users\ratho\FinSentinel\data\creditcard.csv'):
    df = pd.read_csv(path)
    df = df.dropna()

    df['Amount_log'] = np.log1p(df['Amount'])
    df['Hour'] = (df['Time'] % 86400) // 3600

    X = df.drop(['Class', 'Time', 'Amount'], axis=1)
    y = df['Class']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)

    joblib.dump(scaler, 'models/scaler.pkl')

    return (X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled)


if __name__ == "__main__":
    load_and_preprocess()
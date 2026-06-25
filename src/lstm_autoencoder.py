import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, RepeatVector, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, roc_auc_score

from preprocess import load_and_preprocess


def build_autoencoder(n_features):
    inputs  = Input(shape=(1, n_features))
    x       = LSTM(32, activation='relu', return_sequences=False)(inputs)
    x       = RepeatVector(1)(x)
    x       = LSTM(32, activation='relu', return_sequences=True)(x)
    outputs = TimeDistributed(Dense(n_features))(x)

    model = Model(inputs, outputs)
    model.compile(optimizer='adam', loss='mse')
    return model


def train_lstm(X_train_scaled, X_test_scaled, y_train, y_test):
    X_normal   = X_train_scaled[y_train == 0]
    X_normal_r = X_normal.reshape(X_normal.shape[0], 1, X_normal.shape[1])
    X_test_r   = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])

    model = build_autoencoder(X_normal.shape[1])

    model.fit(
        X_normal_r, X_normal_r,
        epochs=10,
        batch_size=1024,
        validation_split=0.1,
        callbacks=[EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)],
        shuffle=True,
        verbose=1
    )

    X_pred    = model.predict(X_test_r, verbose=0)
    mse       = np.mean(np.power(X_test_r - X_pred, 2), axis=(1, 2))
    threshold = np.percentile(mse[y_test == 0], 99.9)
    lstm_pred = (mse > threshold).astype(int)

    print(classification_report(y_test, lstm_pred, target_names=['Normal', 'Fraud']))
    print(f"AUROC: {roc_auc_score(y_test, mse):.4f}")

    plt.figure(figsize=(10, 4))
    plt.hist(mse[y_test == 0],  bins=50, alpha=0.7, label='Normal')
    plt.hist(mse[y_test == 1],  bins=50, alpha=0.7, label='Fraud')
    plt.axvline(threshold, color='red', linestyle='--', label='Threshold')
    plt.legend()
    plt.title('LSTM Reconstruction Error Distribution')
    plt.savefig('outputs/lstm_error.png', dpi=150)
    plt.close()

    model.save('models/lstm_autoencoder.keras')
    np.save('models/lstm_threshold.npy', threshold)

    return model, threshold, mse


if __name__ == "__main__":
    _, _, y_train, y_test, X_train_scaled, X_test_scaled = load_and_preprocess()
    train_lstm(X_train_scaled, X_test_scaled, y_train, y_test)
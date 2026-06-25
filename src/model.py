import xgboost as xgb
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from preprocess import load_and_preprocess


def add_anomaly_features(X_train, X_test):
    iso = IsolationForest(n_estimators=100, contamination=0.002, random_state=42, n_jobs=-1)
    iso.fit(X_train)

    X_train = X_train.copy()
    X_test  = X_test.copy()

    X_train["anomaly_score"] = iso.decision_function(X_train)
    X_test["anomaly_score"]  = iso.decision_function(X_test)

    joblib.dump(iso, "models/isolation_forest.pkl")

    return X_train, X_test


def train_xgboost(X_train, X_test, y_train, y_test):
    fraud_ratio = (y_train == 0).sum() / (y_train == 1).sum()

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=fraud_ratio,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        gamma=1,
        eval_metric="auc",
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    auroc  = roc_auc_score(y_test, y_prob)

    print(classification_report(y_test, y_pred, target_names=["Normal", "Fraud"]))
    print(f"AUROC: {auroc:.4f}")

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Fraud"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues")
    plt.title(f"Confusion Matrix — AUROC: {auroc:.4f}")
    plt.savefig("outputs/confusion_matrix.png", dpi=150)
    plt.close()

    xgb.plot_importance(model, max_num_features=15, importance_type="gain")
    plt.tight_layout()
    plt.savefig("outputs/feature_importance.png", dpi=150)
    plt.close()

    joblib.dump(model, "models/xgboost_model.pkl")

    return model, auroc


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, _, _ = load_and_preprocess()
    X_train, X_test = add_anomaly_features(X_train, X_test)
    train_xgboost(X_train, X_test, y_train, y_test)
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib


def generate_shap_plots(X_test, sample_size=300):
    model     = joblib.load('models/xgboost_model.pkl')
    iso_model = joblib.load('models/isolation_forest.pkl')

    X_test = X_test.copy()
    X_test['anomaly_score'] = iso_model.decision_function(X_test)

    X_sample    = X_test.sample(sample_size, random_state=42)
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    plt.figure()
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    plt.savefig('outputs/shap_bar.png', bbox_inches='tight')
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.savefig('outputs/shap_beeswarm.png', bbox_inches='tight')
    plt.close()

    return explainer, shap_values


if __name__ == "__main__":
    from preprocess import load_and_preprocess
    _, X_test, _, _, _, _ = load_and_preprocess()
    generate_shap_plots(X_test)
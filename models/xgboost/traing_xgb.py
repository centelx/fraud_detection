import numpy as np
import matplotlib.pyplot as plt
import joblib
import shap
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.preprocessing import get_processed_data
from src.utils import find_optimal_threshold, plot_cost_curve
from src.logger import get_logger

logger = get_logger(__name__)

logger.info("--- loading data ---")
X_train, X_test, y_train, y_test = get_processed_data()

logger.info("--- grid search for xgboost ---")
waga_klas = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [4, 6, 8],
    'learning_rate': [0.1, 0.2, 0.3],
    'subsample': [0.8, 1.0]
}

xgb_base = XGBClassifier(scale_pos_weight=waga_klas, random_state=42, n_jobs=-1, eval_metric='logloss')
grid_search_xgb = GridSearchCV(estimator=xgb_base, param_grid=param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=1)

logger.info("starting looking for the best parameters set")
grid_search_xgb.fit(X_train, y_train)

best_xgb = grid_search_xgb.best_estimator_
logger.info(f"best found set: {grid_search_xgb.best_params_}")

logger.info("--- looking for optimal sensitivity threshold ---")
y_probs = best_xgb.predict_proba(X_test)[:, 1]
best_thresh, min_cost, thresholds, costs = find_optimal_threshold(y_test, y_probs, C_FP=60.0, C_FN=122.21)

logger.info("--- drawing plot ---")
plot_cost_curve(thresholds, costs, best_thresh, min_cost, "XGBoost", "models/xgboost/xgb_costs_plot.png")

logger.info("--- SHAP ---")
logger.info("Calculating SHAP value for xgboost model...")
explainer = shap.TreeExplainer(best_xgb)
X_test_sample = X_test.sample(1000, random_state=42)
shap_values = explainer(X_test_sample)

plt.figure(figsize=(10, 8))
plt.title("Global Feature Importance for Fraud Detection", fontsize=14)
shap.summary_plot(shap_values, X_test_sample, show=False)
plt.tight_layout()
plt.savefig('models/xgboost/shap_summary.png', dpi=300, bbox_inches='tight')
plt.close()

logger.info("looking for confirmed fraud in testing set for local analyze")
fraud_indices = np.where(y_test == 1)[0]
first_fraud_idx = fraud_indices[0]
X_single_fraud = X_test.iloc[[first_fraud_idx]]
shap_values_single = explainer(X_single_fraud)

plt.figure(figsize=(10, 6))
shap.plots.waterfall(shap_values_single[0], show=False)
plt.title(f"Decision Breakdown for a Single Fraud Case (Index: {first_fraud_idx})", fontsize=14)
plt.tight_layout()
plt.savefig('models/xgboost/shap_waterfall.png', dpi=300, bbox_inches='tight')
plt.close()
logger.info("SHAP plot saved to dictionary models/xgboost/.")

logger.info("--- model export ---")
joblib.dump(best_xgb, 'models/xgboost/best_xgb_model.pkl')

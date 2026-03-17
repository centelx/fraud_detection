import numpy as np
from sklearn.ensemble import RandomForestClassifier
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

logger.info("--- grid search for random forest ---")
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

rf_base = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1)

grid_search = GridSearchCV(estimator=rf_base, param_grid=param_grid, cv=3, scoring='roc_auc', n_jobs=-1, verbose=1)

logger.info("starting looking for the best rf parameters set")
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_
logger.info(f"best found set: {grid_search.best_params_}")

logger.info("--- looking for optimal sensitivity threshold ---")
y_probs = best_rf.predict_proba(X_test)[:, 1]
best_thresh, min_cost, thresholds, costs = find_optimal_threshold(y_test, y_probs, C_FP=60.0, C_FN=122.21)

logger.info("--- drawing plot ---")
plot_cost_curve(thresholds, costs, best_thresh, min_cost, "Random Forest", "models/random_forest/rf_costs_plot.png")
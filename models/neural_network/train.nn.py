import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import roc_auc_score

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.preprocessing import get_processed_data
from src.utils import find_optimal_threshold, plot_cost_curve
from src.logger import get_logger

logger = get_logger(__name__)

logger.info("--- loading data ---")
X_train, X_test, y_train, y_test = get_processed_data()

logger.info("--- grid search for neural network) ---")
neg, pos = np.bincount(y_train)
total = neg + pos
weight_for_0 = (1 / neg) * (total / 2.0)
weight_for_1 = (1 / pos) * (total / 2.0)
class_weights = {0: weight_for_0, 1: weight_for_1}

param_grid = {
    'batch_size': [512, 1024],
    'learning_rate': [0.001, 0.01]
}

best_auc = 0
best_model = None
best_params = {}

logger.info("starting looking for best hiperparameters set")

for bs in param_grid['batch_size']:
    for lr in param_grid['learning_rate']:
        logger.info(f"Training model: batch_size={bs}, learning_rate={lr} ...")

        model = Sequential([
            Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
            BatchNormalization(),
            Dropout(0.3),
            Dense(32, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(1, activation='sigmoid')
        ])

        model.compile(optimizer=Adam(learning_rate=lr), loss='binary_crossentropy', metrics=['AUC'])

        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=0)

        model.fit(
            X_train, y_train,
            batch_size=bs,
            epochs=50,
            validation_split=0.2,
            class_weight=class_weights,
            callbacks=[early_stop],
            verbose=0
        )

        y_pred_prob = model.predict(X_test, verbose=0).ravel()
        auc = roc_auc_score(y_test, y_pred_prob)
        logger.info(f"-> Finished. ROC-AUC: {auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            best_model = model
            best_params = {'batch_size': bs, 'learning_rate': lr}

logger.info(f"best set: {best_params} (ROC-AUC: {best_auc:.4f})")

logger.info("--- looking for optimal sensitivity thereshold ---")
y_probs = best_model.predict(X_test, verbose=0).ravel()
best_thresh, min_cost, thresholds, costs = find_optimal_threshold(y_test, y_probs, C_FP=60.0, C_FN=122.21)

logger.info("--- drawing plot ---")
plot_cost_curve(thresholds, costs, best_thresh, min_cost, "Neural Network", "models/neural_network/nn_costs_plot.png")
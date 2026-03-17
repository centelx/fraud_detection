import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from src.logger import get_logger

logger = get_logger(__name__)

def find_optimal_threshold(y_true, y_probs, C_FP=60.0, C_FN=122.21):
    """
    scanning threshold from 0.01 to 0.99 and calculating buisnes costs for each.
    """
    thresholds = np.arange(0.01, 1.00, 0.01)
    costs = []

    logger.info(f"Calculating cost for 99 different thresholds (Cost FP: {C_FP}, Cost FN: {C_FN})")

    for thresh in thresholds:
        y_pred_thresh = (y_probs >= thresh).astype(int)
        cm = confusion_matrix(y_true, y_pred_thresh)

        FP = cm[0][1]
        FN = cm[1][0]

        cost = (FP * C_FP) + (FN * C_FN)
        costs.append(cost)

    min_cost = min(costs)
    best_thresh = thresholds[costs.index(min_cost)]

    logger.info("-" * 50)
    logger.info(f"The cheapest threshold: {best_thresh:.2f}")
    logger.info(f"minimal cost with this threshold: {min_cost:.2f} PLN")
    logger.info("-" * 50)

    return best_thresh, min_cost, thresholds, costs


def plot_cost_curve(thresholds, costs, best_thresh, min_cost, algorithm_name, save_path=None):
    """
    Plots the cost curve and optionally saves it as a high-resolution image.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, costs, label='cost curve', color='red', linewidth=2)
    plt.axvline(x=best_thresh, color='blue', linestyle='--', label=f'Optimal threshold ({best_thresh:.2f})')

    plt.scatter(best_thresh, min_cost, color='blue', s=100, zorder=5)

    plt.title(f'Total Business Cost vs. Decision Threshold ({algorithm_name})', fontsize=14)
    plt.xlabel('Threshold (from 0 to 1)', fontsize=12)
    plt.ylabel('Total cost [PLN]', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Plot saved to file: {save_path}")

    plt.show()
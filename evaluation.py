#!/user/bin/env python3
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import r_regression
from scipy.stats import spearmanr
from sklearn.metrics import cohen_kappa_score


def round_score(x):
    # Round scores for QWK calculation
    thresholds = [1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.25, 4.75]
    for i, threshold in enumerate(thresholds, start=1):
        if x < threshold:
            return i
    return len(thresholds) + 1  # Default value if x is greater than all thresholds


def get_metrics(task_type, y_trues, y_preds):
    # Initialize lists
    rmse = []
    pcc = []
    scc = []
    qwk = []

    # Get the number of categories (columns)
    num_categories = y_trues.shape[1]

    # Iterate over each category
    for i in range(num_categories):
        y_true = y_trues[:, i]
        y_pred = y_preds[:, i]

        # Calculate and append RMSE, PCC, SCC, and QWK for the current category
        rmse.append(mean_squared_error(y_true, y_pred, squared=False))
        pcc.append(r_regression(y_true.reshape(-1, 1), y_pred.reshape(-1, 1))[0])
        scc.append(spearmanr(y_true, y_pred)[0])
        qwk.append(
            cohen_kappa_score(
                pd.DataFrame(y_true).applymap(round_score),
                pd.DataFrame(y_pred).applymap(round_score),
                weights='quadratic'
            )
        )

    # Calculate mean values for RMSE, PCC, SCC, and QWK
    rmse.append(np.mean(rmse))
    pcc.append(np.mean(pcc))
    scc.append(np.mean(scc))
    qwk.append(np.mean(qwk))

    if task_type == 'ell':
        col_names = ['cohesion', 'syntax', 'vocabulary', 'phraseology', 'grammar', 'conventions', 'avg_val']
    elif task_type == 'asap_12':
        col_names = ['Content', 'Organization', 'Word Choice', 'Sentence Fluency', 'Conventions', 'avg_val']
    elif task_type == 'asap_36':
        col_names = ['Content', 'Prompt Adherence', 'Language', 'Narrativity', 'avg_val']
    else:
        raise ValueError('Invalid task type')
    idx_names = ['rmse', 'pcc', 'scc', 'qwk']

    return pd.DataFrame([rmse, pcc, scc, qwk], columns=col_names, index=idx_names)

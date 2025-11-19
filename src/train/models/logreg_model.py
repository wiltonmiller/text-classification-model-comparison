from __future__ import annotations

from typing import Any, Dict
import numpy as np
from sklearn.linear_model import LogisticRegression

def train_logreg(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    C: float = 1.0,
    max_iter: int = 1000,
    random_state: int = 0,
) -> LogisticRegression:
    """
    Train a multinomial logistic regression classifier using the given hyperparameters.
    Parameters:
    X_train, y_train : training data and labels
    C : inverse regularization strength
    max_iter : optimizer iteration limit
    random_state : reproducibility

    Returns:
    model : trained LogisticRegression instance
    """
    
    model = LogisticRegression(
        C=C,
        penalty="l2",
        solver="lbfgs",
        multi_class="multinomial",
        max_iter=max_iter,
        random_state=random_state,
    )

    model.fit(X_train, y_train)
    return model

def extract_parameters(model: LogisticRegression) -> Dict[str, Any]:
    """
    Extract the learned logistic regression parameters so that Part C can
    manually perform the forward pass without sklearn.

    Returns: 
    dict with fields:
        'weights': weight matrix (K x d)
        'bias': bias vector (K)
        'classes': class label order used during training
    """
    params = {
        'weights': model.coef_.tolist(), # shape (num_classes, num_features)
        'bias': model.intercept_.tolist(), # shape (num_classes,)
        'classes': model.classes_.tolist(), # shape (num_classes,)
    }
    return params
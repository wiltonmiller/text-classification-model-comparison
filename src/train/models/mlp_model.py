from __future__ import annotations
from typing import Any, Dict, Tuple
import numpy as np
from sklearn.neural_network import MLPClassifier

def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    hidden_layer_sizes: Tuple[int, ...] = (128,),
    alpha: float = 1e-4,
    batch_size: int = 64,
    max_iter: int = 200,
    random_state: int = 0,
) -> MLPClassifier:
    """
    Train a feed-forward neural network using the Adam optimizer and L2 weight decay.
    Parameters:
    X_train, y_train : training data and labels
    hidden_layer_sizes : tuple defining sizes of hidden layers
    alpha : L2 regularization strength
    batch_size : mini-batch size for Adam
    max_iter : training iteration limit
    random_state : reproducibility

    Returns:
    model : a trained MLPClassifier instance
    """

    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation='relu',
        solver='adam',
        alpha=alpha,
        batch_size=batch_size,
        max_iter=max_iter,
        random_state=random_state,
    )

    model.fit(X_train, y_train)

    return model

def extract_parameters(model: MLPClassifier) -> Dict[str, Any]:
    """
    Extract the learned MLP parameters so that Part C can manually perform inference. 

    Returns:
    dict with fields:
        'weights' : list of weight matrices
        'biases' : list of bias vectors
        'hidden_layers' : hidden-layer structure
        'activation' : hidden-layer activation
        'out_activation' : output-layer activation (softmax)
        'classes' : class ordering
    """
    params = {
        'weights': [w.tolist() for w in model.coefs_], # list of weight matrices
        'biases': [b.tolist() for b in model.intercepts_], # list of bias vectors
        "hidden_layers": list(model.hidden_layer_sizes), # hidden-layer structure
        "activation": model.activation, # hidden-layer activation
        "out_activation": model.out_activation_, # output-layer activation (softmax)
        'classes': model.classes_.tolist(), # class ordering
    }
    return params


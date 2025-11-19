from __future__ import annotations
from typing import Any, Dict, Optional
import numpy as np
from sklearn.tree import DecisionTreeClassifier

def train_tree(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    max_depth: Optional[int] = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    max_features: Optional[str] = None,
    random_state: int = 0,
) -> DecisionTreeClassifier:
    """
    Train a Decision Tree classifier using the given hyperparameters.
    Parameters:
    X_train, y_train : training data and labels
    max_depth : maximum depth of the tree
    min_samples_split : minimum samples required to split an internal node
    min_samples_leaf : minimum samples required in each leaf
    max_features : number of features to consider when looking for best split
    random_state : reproducibility

    Returns:
    model : trained DecisionTreeClassifier instance
    """

    model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        random_state=random_state,
    )

    model.fit(X_train, y_train)
    return model

def extract_parameters(model: DecisionTreeClassifier) -> Dict[str, Any]:
    """
    Decision Trees should NOT be implemented manually in Part C.
    The tree will instead be pickled. 

    This returns metadata for reporting and debugging.
    """
    tree = model.tree_

    return {
        "n_nodes": int(tree.node_count), # total number of nodes
        "max_depth": int(tree.max_depth), # maximum depth of the tree
        "classes": model.classes_.tolist(), # class label order used during training
    }
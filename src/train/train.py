from __future__ import annotations
import json
import pickle
import numpy as np
from pathlib import Path
from itertools import product
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# model families
from train.models.logreg_model import train_logreg, extract_parameters as extract_logreg
from train.models.tree_model import train_tree, extract_parameters as extract_tree
from train.models.mlp_model import train_mlp, extract_parameters as extract_mlp

# 1. load training and test arrays

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

X_train = np.load(DATA_DIR / "training_features.npy")
y_train = np.load(DATA_DIR / "training_labels.npy")
X_test = np.load(DATA_DIR / "testing_features.npy")
y_test = np.load(DATA_DIR / "testing_labels.npy")

# 2. define hyperparameter grids

# logistic regression hyperparameters 
logreg_grid = {"C": [0.01, 0.1, 1.0, 10.0, 100.0]}

# decision tree hyperparameters
tree_grid = {
    "max_depth": [5, 10, 15, None],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 5],
    "max_features": ["sqrt", "log2"],
}

# MLP hyperparameters
mlp_grid = {
    "hidden_layer_sizes": [(64,), (128,)],
    "alpha": [1e-4, 1e-3],
    "batch_size": [64],
    "max_iter": [200],
}

# 3. setup cross validation

kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

def fold_macro_f1(model, X_val, y_val):
    """compute the Macro-F1 for a validation fold."""
    return f1_score(y_val, model.predict(X_val), average="macro")

# 4. cross-validate grid search

def run_grid_search(name: str, train_fn, param_grid):
    """run grid search with cross-validation for a given model family."""

    print(f"Starting grid search for {name}\n")

    best_params = None
    best_score = -np.inf

    # create all combinations of hyperparameters
    keys = list(param_grid.keys())
    combos = list(product(*param_grid.values()))

    for combo in combos:
        params = dict(zip(keys, combo))
        cv_scores = []

        print(f" Testing {name} parameters: {params}")

        for train_idx, val_idx in kf.split(X_train, y_train):
            X_tr, X_val = X_train[train_idx], X_train[val_idx]
            y_tr, y_val = y_train[train_idx], y_train[val_idx]

            try:
                model = train_fn(X_tr, y_tr, **params)
                score = fold_macro_f1(model, X_val, y_val)
            except Exception as e:
                print(f"Error training {name} with params {params}: {e}")
                score = 0.0

            cv_scores.append(score)
        
        avg_score = float(np.mean(cv_scores))
        print(f"  Average Macro-F1: {avg_score:.4f}\n")

        if avg_score > best_score: 
            best_score = avg_score
            best_params = params

    print(f"Best parameters for {name}: {best_params} with Macro-F1: {best_score:.4f}\n")
    return best_params

# 5. run grid searches for each model family

best_logreg_params = run_grid_search("Logistic Regression", train_logreg, logreg_grid)
best_tree_params = run_grid_search("Decision Tree", train_tree, tree_grid)
best_mlp_params = run_grid_search("MLP", train_mlp, mlp_grid)

# 6. train final models on full training set

print("Training final models on full training set\n")

scores = {}

# logistic regression
logreg_model = train_logreg(X_train, y_train, **best_logreg_params)
logreg_pred = logreg_model.predict(X_test)
scores["logistic_regression"] = f1_score(y_test, logreg_pred, average="macro")

# decision tree
tree_model = train_tree(X_train, y_train, **best_tree_params)
tree_pred = tree_model.predict(X_test)
scores["decision_tree"] = f1_score(y_test, tree_pred, average="macro")
# MLP
mlp_model = train_mlp(X_train, y_train, **best_mlp_params)
mlp_pred = mlp_model.predict(X_test)
scores["mlp"] = f1_score(y_test, mlp_pred, average="macro")

best_model_type = max(scores, key=scores.get)
print(f"Best overall model: {best_model_type}\n")

if best_model_type == "logreg":
    best_model = logreg_model
    parameters = extract_logreg(best_model)

elif best_model_type == "tree":
    best_model = tree_model
    parameters = extract_tree(best_model)

else:
    best_model = mlp_model
    parameters = extract_mlp(best_model)

# 7. save best model and parameters

with open(MODEL_DIR / "best_model_type.txt", "w") as f:
    f.write(best_model_type)

if best_model_type == "tree":
    # we need to pickle the tree model
    with open(MODEL_DIR / "best_model.pkl", "wb") as f:
        pickle.dump(best_model, f)

else:
    # for other models we just save the parameters as a json
    with open(MODEL_DIR / "best_model_params.json", "w") as f:
        json.dump(parameters, f, indent=4)

# save the evaluation metrics
metrics = {
    "test_accuracy": float(accuracy_score(y_test, best_model.predict(X_test))),
    "test_macro_f1": float(f1_score(y_test, best_model.predict(X_test), average="macro")),
    "confusion_matrix": confusion_matrix(y_test, best_model.predict(X_test)).tolist(),
    "all_model_test_f1": scores,
}

with open(MODEL_DIR / "metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("Training complete. Best model and parameters saved.\n")

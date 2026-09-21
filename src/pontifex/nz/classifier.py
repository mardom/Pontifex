"""Pontifex Classifier: Tomographic Bin Assignment & MoE Boundary Regularization."""

from typing import Optional, Tuple
import numpy as np

try:
    import xgboost as xgb
except ImportError:
    xgb = None


def train_tomographic_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    sample_weights: Optional[np.ndarray] = None,
    n_estimators: int = 180,
    max_depth: int = 6,
    learning_rate: float = 0.08,
    use_gpu: bool = True,
    random_state: int = 42,
):
    """Train gradient-boosted decision trees for tomographic bin classification.
    
    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix (magnitudes, errors, fluxes, colors).
    y_train : np.ndarray
        Target tomographic bin labels (0 to n_tomo_bins - 1).
    sample_weights : Optional[np.ndarray]
        SOM transfer density ratio weights.
    n_estimators : int
        Number of gradient boosted trees.
    max_depth : int
        Maximum tree depth.
    learning_rate : float
        Shrinkage factor.
    use_gpu : bool
        Whether to attempt GPU training via 'cuda' device.
    random_state : int
        Random seed.
    """
    if xgb is None:
        raise ImportError("xgboost is required for tomographic classification.")

    device = "cuda" if use_gpu else "cpu"
    clf = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        device=device,
        random_state=random_state,
        eval_metric="mlogloss",
    )
    clf.fit(X_train, y_train, sample_weight=sample_weights)
    return clf


def predict_tomographic_bins(
    clf,
    X_wfd: np.ndarray,
    n_tomo_bins: int,
    entropy_threshold: float = 1.25,
    regularization_weight: float = 0.12,
) -> Tuple[np.ndarray, np.ndarray]:
    """Predict tomographic bin assignments with Hybrid MoE boundary entropy regularization.
    
    Parameters
    ----------
    clf : xgb.XGBClassifier
        Trained classifier model.
    X_wfd : np.ndarray
        Wide survey target features.
    n_tomo_bins : int
        Number of tomographic bins.
    entropy_threshold : float
        Shannon entropy threshold above which boundary objects are regularized.
    regularization_weight : float
        Mixing weight for uniform prior on high-entropy objects.
        
    Returns
    -------
    bin_assignments : np.ndarray
        Argmax tomographic bin index per galaxy.
    probs : np.ndarray
        Regularized posterior probability matrix of shape (n_gal, n_tomo_bins).
    """
    probs = clf.predict_proba(X_wfd)

    # Hybrid MoE entropy regularization for uncertain boundary objects
    entropy = -np.sum(probs * np.log(np.maximum(probs, 1e-12)), axis=1)
    uncertain = entropy > entropy_threshold
    if np.any(uncertain):
        probs[uncertain] = (1.0 - regularization_weight) * probs[uncertain] + regularization_weight * (1.0 / n_tomo_bins)

    bin_assignments = np.argmax(probs, axis=1).astype(int)
    return bin_assignments, probs

import numpy as np
import pandas as pd
import torch
from db_ocsvm_04 import DBOCSVM
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, ConcatDataset
from sklearn.svm import OneClassSVM
from sklearn.model_selection import KFold
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
import optuna
import matplotlib.pyplot as plt
import random
from sklearn.cluster import DBSCAN
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)


def inner_objective_ocsvm(
    trial: optuna.Trial,
    dataset: pd.DataFrame,
    autoencoder: nn.Module,
    nu_range: tuple = (0.01, 0.5),
    gamma_range: tuple = (0.01, 1.0),
    kfold_splits: int = 7,
) -> float:
    """
    Inner objective function for optimizing One-Class SVM hyperparameters using encoded features.

    This function is used by Optuna for hyperparameter optimization of a One-Class SVM model
    that operates on features extracted from an autoencoder. It performs k-fold cross-validation
    to evaluate model performance with different hyperparameter settings.

    Parameters
    ----------
    trial : optuna.Trial
        Optuna trial object used for hyperparameter suggestion
    dataset : pd.DataFrame
        DataFrame containing features and target labels
    autoencoder : nn.Module
        Trained autoencoder model used for feature extraction
    nu_range : tuple, default=(0.01, 0.5)
        Range (min, max) for the nu parameter of OneClassSVM
    gamma_range : tuple, default=(0.01, 1.0)
        Range (min, max) for the gamma parameter of OneClassSVM
    kfold_splits : int, default=7
        Number of folds for cross-validation

    Returns
    -------
    float
        Mean accuracy score across all k-fold evaluation runs

    Notes
    -----
    - The function extracts features using the autoencoder's encode method
    - Only normal samples are used to train the One-Class SVM
    - Performance is evaluated using accuracy on the validation sets
    - Higher return values indicate better hyperparameter configurations
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    X = dataset.drop(
        ["attack_binary", "attack_categorical", "attack_class"], axis=1
    ).values
    y = dataset["attack_binary"].values

    # Extract features
    autoencoder.eval()

    # Extract in batches to prevent memory issues
    X_tensor = torch.FloatTensor(X).to(device)
    X_tensor_dataset = TensorDataset(X_tensor, torch.zeros(len(X_tensor)))
    normal_loader = DataLoader(X_tensor_dataset, batch_size=128)

    X_encoded = []
    with torch.no_grad():
        for data, _ in normal_loader:
            encoded = autoencoder.encode(data)
            X_encoded.append(encoded.cpu().numpy())
    X_encoded = np.vstack(X_encoded)

    nu = trial.suggest_float("nu", nu_range[0], nu_range[1])
    gamma = trial.suggest_float("gamma", gamma_range[0], gamma_range[1])

    kf = KFold(n_splits=kfold_splits, shuffle=True, random_state=42)

    scores = []
    for train_idx, val_idx in kf.split(X_encoded):
        X_train, X_val = X_encoded[train_idx], X_encoded[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        X_train_normal = X_train[y_train == 1]

        ocsvm = OneClassSVM(kernel="rbf", nu=nu, gamma=gamma)

        ocsvm.fit(X_train_normal)

        y_pred = ocsvm.predict(X_val)

        acc = accuracy_score(y_val, y_pred)

        scores.append(acc)

    return np.mean(scores)


def objective_dbscan(
    trial: optuna.Trial,
    X_encoded: np.ndarray,
    evaluation_metric: str = "silhouette",
    eps_range: tuple[float, float] = (0.1, 15.0),
    min_samples_range: tuple[int, int] = (20, 50),
    distance_metric: str = "euclidean",  # "manhattan", "cosine"
    n_jobs: int = -1,
) -> float:
    """
    Inner objective function for optimizing DBSCAN clustering hyperparameters.

    This function is used by Optuna for hyperparameter optimization of a DBSCAN model.
    It evaluates different clustering configurations using the specified evaluation metric.

    Parameters
    ----------
    trial : optuna.Trial
        Optuna trial object used for hyperparameter suggestion
    X_encoded : np.ndarray
        Array of encoded data points to cluster
    evaluation_metric : str, default="silhouette"
        Metric to evaluate cluster quality: "silhouette", "davies_bouldin", or "calinski_harabasz"
    eps_range : tuple[float, float], default=(0.1, 15.0)
        Range (min, max) for the eps parameter of DBSCAN
    min_samples_range : tuple[int, int], default=(20, 50)
        Range (min, max) for the min_samples parameter of DBSCAN
    metric : str, default="euclidean"
        Distance metric for DBSCAN
    score_threshold : float, default=0.60
        Minimum score threshold to consider a clustering configuration valid
    n_jobs : int, default=-1
        Number of parallel jobs to run for DBSCAN

    Returns
    -------
    float
        Clustering quality score (higher is better) unless evaluation_metric is "davies_bouldin"
        Returns -inf for invalid clustering configurations
    """

    def get_score(X, labels, metric_name, mask=None):
        if mask is not None:
            X = X[mask]
            labels = labels[mask]

        if metric_name == "silhouette":
            return silhouette_score(X, labels)
        elif metric_name == "davies_bouldin":
            return -davies_bouldin_score(
                X, labels
            )  # Negative because we want to maximize
        elif metric_name == "calinski_harabasz":
            return calinski_harabasz_score(X, labels)
        else:
            raise ValueError(f"Unknown metric: {metric_name}")

    eps = trial.suggest_float("eps", eps_range[0], eps_range[1])
    min_samples = trial.suggest_int(
        "min_samples", min_samples_range[0], min_samples_range[1]
    )

    dbscan = DBSCAN(
        eps=eps, min_samples=min_samples, metric=distance_metric, n_jobs=n_jobs
    )

    cluster_labels = dbscan.fit_predict(X_encoded)

    # Calculate the number of clusters (excluding noise points)
    unique_clusters = set(cluster_labels)
    n_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)
    print(n_clusters)

    # Set custom attribute for number of clusters
    trial.set_user_attr("n_clusters", n_clusters)

    # Set custom attribute for cluster data points
    cluster_data_points = {}
    for cluster_id in unique_clusters:
        # Store indices of points in each cluster
        cluster_indices = np.where(cluster_labels == cluster_id)[0].tolist()
        cluster_data_points[int(cluster_id)] = len(cluster_indices)
    trial.set_user_attr("cluster_data_points", cluster_data_points)
    print(cluster_data_points)

    if n_clusters < 2:
        print("not enough clusters")
        return -float("inf")  # Penalize solutions with too few clusters

    # For silhouette score, we need to exclude noise points (-1)
    if evaluation_metric == "silhouette":
        mask = cluster_labels != -1
        if sum(mask) < 2:
            print("not enough points in clusters")
            return -float("inf")
        score = get_score(X_encoded, cluster_labels, evaluation_metric, mask)
    else:
        score = get_score(X_encoded, cluster_labels, evaluation_metric)

    return score


def objective_dbocsvm_fit_ocsvm(
    trial: optuna.Trial,
    model: DBOCSVM,
    X_encoded_train: np.ndarray,
    X_encoded_validation: np.ndarray,
    y_validation: np.ndarray,
    X_encoded_test: np.ndarray,
    y_test: np.ndarray,
    cluster_count: int = 20,
    metric: str = "accuracy",
) -> float:

    parameter_list = {}
    for cluster in range(0, cluster_count):
        hyperparameter = {
            "kernel": "rbf",
            "gamma": trial.suggest_float(f"gamma_{cluster}", 1e-4, 1.0),
            "nu": trial.suggest_float(f"nu_{cluster}", 0.01, 0.5),
        }
        parameter_list[cluster] = hyperparameter

    model.fit_ocsvm(X_encoded_train, parameter_list=parameter_list)

    y_pred_test = model.predict(X_encoded_test)
    y_pred_val = model.predict(X_encoded_validation)

    acc_val = accuracy_score(y_validation, y_pred_val)
    f1_val = f1_score(y_validation, y_pred_val, pos_label=-1)
    precision_val = precision_score(y_validation, y_pred_val, pos_label=-1)
    recall_val = recall_score(y_validation, y_pred_val, pos_label=-1)
    print("Validation Results:")
    print(
        {
            "accuracy": f"{acc_val * 100:.2f}",
            "f1": f"{f1_val * 100:.2f}",
            "precision": f"{precision_val * 100:.2f}",
            "recall": f"{recall_val * 100:.2f}",
        }
    )

    print("\nTest Results:")
    acc_test = accuracy_score(y_test, y_pred_test)
    f1_test = f1_score(y_test, y_pred_test, pos_label=-1)
    precision_test = precision_score(y_test, y_pred_test, pos_label=-1)
    recall_test = recall_score(y_test, y_pred_test, pos_label=-1)
    print(
        {
            "accuracy": f"{acc_test * 100:.2f}",
            "f1": f"{f1_test * 100:.2f}",
            "precision": f"{precision_test * 100:.2f}",
            "recall": f"{recall_test * 100:.2f}",
        }
    )

    if metric == "accuracy":
        return acc_val
    elif metric == "f1":
        return f1_val
    elif metric == "precision":
        return precision_val
    elif metric == "recall":
        return recall_val
    else:
        raise ValueError(f"Unknown metric: {metric}")


def inner_objective_dbocsvm(
    trial: optuna.Trial,
    X_encoded: np.ndarray,
    y: np.ndarray,
    cluster_count: int = 20,
    kfold_splits: int = 7,
    eps: float = 1.5,
    min_samples: int = 50,
    tree_algorithm: str = "kd_tree",  # "ball_tree"
) -> float:

    parameter_list = {}
    for cluster in range(0, cluster_count):
        hyperparameter = {
            "kernel": "rbf",
            "gamma": trial.suggest_float(f"gamma_{cluster}", 1e-4, 1.0, log=True),
            "nu": trial.suggest_float(f"nu_{cluster}", 0.01, 0.5),
        }
        parameter_list[cluster] = hyperparameter

    kf = KFold(n_splits=kfold_splits, shuffle=True, random_state=42)

    scores = []

    for train_idx, val_idx in kf.split(X_encoded):
        X_train, X_val = X_encoded[train_idx], X_encoded[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        X_train_normal = X_train[y_train == 1]

        dbocsvm = DBOCSVM(
            kernel="rbf",
            gamma="auto",
            nu=0.2,
            eps=eps,
            min_samples=min_samples,
            algorithm=tree_algorithm,
        )

        dbocsvm.fit(X_train_normal, parameter_list=parameter_list)

        y_pred = dbocsvm.predict(X_val)

        acc = accuracy_score(y_val, y_pred)
        scores.append(acc)

    # TODO: write a different objective function but use the test set to evaluate the model
    # Return mean acc score across all folds
    return np.mean(scores)


def set_seed(seed, cudnn_deterministic=False):
    """Set all random seeds for reproducibility"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if cudnn_deterministic:
        # Make cudnn deterministic (slightly lower performance but reproducible)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_autoencoder(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    val_loader: DataLoader = None,
    epochs: int = 10,
    best_model_path: str = "./best_autoencoder_tuning.pth",
    verbose: bool = False,
    early_stopping_patience: int = 5,
    improvement_threshold: float = 0.001,  # Minimum improvement to be considered significant
    good_model_threshold: float = 0.05,  # Threshold for considering a model "good"
    plot_results: bool = True,
):
    """
    Train an autoencoder model with early stopping and model saving capabilities.

    This function handles the complete training workflow for an autoencoder, including:
    - Forward/backward passes and optimization
    - Loss tracking for training and validation
    - Early stopping when improvement plateaus
    - Saving the best model based on validation or training loss
    - Optional progress reporting and loss curve plotting
    - Model quality assessment based on final reconstruction loss

    Parameters
    ----------
    model : nn.Module
        The autoencoder model to train
    train_loader : DataLoader
        PyTorch DataLoader for training data
    optimizer : optim.Optimizer
        PyTorch optimizer for model parameter updates
    criterion : nn.Module
        Loss function (typically MSELoss for autoencoders)
    val_loader : DataLoader, optional
        PyTorch DataLoader for validation data, by default None
    epochs : int, optional
        Maximum number of training epochs, by default 10
    best_model_path : str, optional
        Path to save the best model checkpoint, by default "./best_autoencoder_tuning.pth"
    verbose : bool, optional
        Whether to print training progress, by default False
    early_stopping_patience : int, optional
        Number of epochs with no improvement after which to stop training, by default 5
    improvement_threshold : float, optional
        Minimum improvement in loss to be considered significant, by default 0.001
    good_model_threshold : float, optional
        Maximum loss value for a model to be considered "good", by default 0.05
    plot_results : bool, optional
        Whether to plot loss curves after training, by default True

    Returns
    -------
    tuple
        (history, is_good_model) where:
        - history: dict with loss values over epochs
        - is_good_model: bool indicating if model meets quality threshold

    Notes
    -----
    - When validation data is provided, early stopping is based on validation loss,
      otherwise training loss is used
    - The best model is loaded at the end of training
    - If best_val_loss < good_model_threshold, the model is considered good
    """

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)

    if verbose:
        print("")
        print(f"Using device: {device}")
        print("Training autoencoder...")

    history = {"loss": [], "val_loss": []}
    model.train()

    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(epochs):

        total_loss = 0.0
        for batch in train_loader:
            batch_x = batch[0].to(device)

            # Forward pass
            outputs = model(batch_x)
            loss = criterion(outputs, batch_x)

            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        history["loss"].append(avg_loss)

        if val_loader is not None:
            model.eval()
            val_loss = 0.0

            with torch.no_grad():
                for val_batch in val_loader:
                    val_inputs = val_batch[0].to(device)
                    val_outputs = model(val_inputs)
                    batch_val_loss = criterion(val_outputs, val_inputs).item()
                    val_loss += batch_val_loss

            avg_val_loss = val_loss / len(val_loader)
            history["val_loss"].append(avg_val_loss)

            if verbose:
                print(
                    f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.10f}, Val Loss: {avg_val_loss:.10f}"
                )
            # Save model if validation loss improved
            if avg_val_loss < best_val_loss:
                improvement = best_val_loss - avg_val_loss

                # Check if improvement is significant
                if improvement > improvement_threshold:
                    patience_counter = 0  # Reset patience counter
                else:
                    patience_counter += 1

                # Save the model
                best_val_loss = avg_val_loss
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": best_val_loss,
                    },
                    best_model_path,
                )

                if verbose:
                    print(f"✅ Model saved with val_loss: {best_val_loss:.10f}")

                model.train()
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(
                        f"Early stopping triggered after {epoch+1} epochs due to lack of improvement."
                    )
                break
        else:
            if verbose:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.10f}")

            # If no validation set, save based on training loss
            if avg_loss < best_val_loss:
                improvement = best_val_loss - avg_loss

                # Check if improvement is significant
                if improvement > improvement_threshold:
                    patience_counter = 0
                else:
                    patience_counter += 1

                best_val_loss = avg_loss
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "train_loss": best_val_loss,
                    },
                    best_model_path,
                )

                if verbose:
                    print(f"✅ Model saved with train_loss: {best_val_loss:.10f}")
            else:
                patience_counter += 1

            # Check early stopping conditions
            if patience_counter >= early_stopping_patience:
                if verbose:
                    print(
                        f"Early stopping triggered after {epoch+1} epochs due to lack of improvement."
                    )
                break

    if plot_results:
        plt.figure(figsize=(15, 10))
        plt.plot(history["loss"], label="Training Loss")
        if "val_loss" in history and history["val_loss"]:
            plt.plot(history["val_loss"], label="Validation Loss")
        plt.title(f"Loss History (Epoch {epoch+1})")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

        plt.close()

        # Load best model
        checkpoint = torch.load(best_model_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        best_loss = checkpoint.get("val_loss", checkpoint.get("train_loss"))

    if best_loss < good_model_threshold:
        print(f"Model is good with loss {best_loss}")
        is_good_model = True
    else:
        print(f"Model is bad with loss {best_loss}")
        is_good_model = False

    return history, is_good_model


def find_eps_range_with_elbow_method(X, k=20, multiplier=(0.5, 2.0), plot=True):
    """
    Find a suitable eps range for DBSCAN using the elbow method.

    Parameters:
    -----------
    X : array-like
        The encoded data points
    k : int, default=20
        Number of neighbors to consider (corresponds to min_samples)
    multiplier : tuple, default=(0.5, 2.0)
        Factors to multiply the elbow point by to create a range
    plot : bool, default=True
        Whether to show the k-distance plot

    Returns:
    --------
    tuple
        (min_eps, max_eps) suitable range for eps parameter
    """

    # Calculate distances to k nearest neighbors for each point
    nbrs = NearestNeighbors(n_neighbors=k, metric="manhattan").fit(X)
    distances, _ = nbrs.kneighbors(X)

    # Sort the distances to the kth neighbor in ascending order
    k_distances = np.sort(distances[:, -1])

    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(k_distances)
        plt.xlabel("Points (sorted)")
        plt.ylabel(f"Distance to {k}th nearest neighbor")
        plt.title("K-distance Plot for DBSCAN eps Selection")
        plt.grid(True, alpha=0.3)

        # Add horizontal lines at suggested eps range
        # This will be calculated below

        plt.show()

    # Find the elbow point (simple method - you might want a more sophisticated approach)
    # Look for maximum curvature in the sorted distance plot
    n_points = len(k_distances)
    all_coords = np.vstack((range(n_points), k_distances)).T

    # Compute point-to-line distances for all points
    line_vec = all_coords[-1] - all_coords[0]
    line_vec_norm = line_vec / np.sqrt(np.sum(line_vec**2))
    vec_from_first = all_coords - all_coords[0]
    scalar_prod = np.sum(vec_from_first * np.tile(line_vec_norm, (n_points, 1)), axis=1)
    vec_to_line = vec_from_first - np.outer(scalar_prod, line_vec_norm)
    dist_to_line = np.sqrt(np.sum(vec_to_line**2, axis=1))

    # Elbow point is the point with max distance to the line
    elbow_idx = np.argmax(dist_to_line)
    elbow_eps = k_distances[elbow_idx]

    # Create a range around the elbow point
    min_eps = elbow_eps * multiplier[0]
    max_eps = elbow_eps * multiplier[1]

    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(k_distances)
        plt.axhline(
            y=min_eps,
            color="r",
            linestyle="--",
            alpha=0.7,
            label=f"Min eps: {min_eps:.2f}",
        )
        plt.axhline(
            y=elbow_eps,
            color="g",
            linestyle="-",
            alpha=0.7,
            label=f"Elbow eps: {elbow_eps:.2f}",
        )
        plt.axhline(
            y=max_eps,
            color="r",
            linestyle="--",
            alpha=0.7,
            label=f"Max eps: {max_eps:.2f}",
        )
        plt.axvline(x=elbow_idx, color="g", linestyle=":", alpha=0.5)
        plt.xlabel("Points (sorted)")
        plt.ylabel(f"Distance to {k}th nearest neighbor")
        plt.title("K-distance Plot with Suggested eps Range")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    return min_eps, max_eps

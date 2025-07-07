import numpy as np
from db_ocsvm import DBOCSVM
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
from sklearn.cluster import DBSCAN
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)


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

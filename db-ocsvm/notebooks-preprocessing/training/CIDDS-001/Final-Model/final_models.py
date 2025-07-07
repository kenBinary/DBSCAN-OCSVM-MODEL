import torch
import torch.nn as nn
from typing import List, Optional
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.svm import OneClassSVM
from sklearn.neighbors import KDTree, BallTree
import pandas as pd


class BatchNormAutoencoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dims: List[int] = [64, 32],
        latent_dim: int = 16,
        activation_type: str = "ReLU",
        output_activation_type: Optional[str] = None,
    ) -> None:
        super(BatchNormAutoencoder, self).__init__()

        # Select activation function
        activation: nn.Module
        if activation_type == "ReLU":
            activation = nn.ReLU()
        elif activation_type == "LeakyReLU":
            activation = nn.LeakyReLU()
        elif activation_type == "ELU":
            activation = nn.ELU()
        else:
            raise ValueError("Unknown activation type provided")

        # Build encoder
        encoder_layers: List[nn.Module] = []
        current_dim: int = input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(current_dim, h_dim))
            encoder_layers.append(nn.BatchNorm1d(h_dim))
            encoder_layers.append(activation)
            current_dim = h_dim

        # Latent layer
        encoder_layers.append(nn.Linear(current_dim, latent_dim))
        self.encoder: nn.Sequential = nn.Sequential(*encoder_layers)

        # Select output activation function
        output_activation: Optional[nn.Module] = None
        if output_activation_type == "ReLU":
            output_activation = nn.ReLU()
        elif output_activation_type == "LeakyReLU":
            output_activation = nn.LeakyReLU()
        elif output_activation_type == "ELU":
            output_activation = nn.ELU()
        elif output_activation_type == "Sigmoid":
            output_activation = nn.Sigmoid()
        elif output_activation_type == "Tanh":
            output_activation = nn.Tanh()
        elif output_activation_type is None:
            output_activation = None
        else:
            raise ValueError("Unknown activation type provided")

        # Build decoder
        decoder_layers: List[nn.Module] = []
        current_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(current_dim, h_dim))
            decoder_layers.append(nn.BatchNorm1d(h_dim))
            decoder_layers.append(activation)
            current_dim = h_dim

        # Add final output layer (no batch norm on output layer)
        decoder_layers.append(nn.Linear(current_dim, input_dim))

        # Add output activation if specified
        if output_activation is not None:
            decoder_layers.append(output_activation)

        self.decoder: nn.Sequential = nn.Sequential(*decoder_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encoded: torch.Tensor = self.encoder(x)
        decoded: torch.Tensor = self.decoder(encoded)
        return decoded

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class DBOCSVM:
    """
    Parameters
    ----------
    kernel : str, default='rbf'
        Specifies the kernel type to be used in the One-Class SVM.
        Options: 'linear', 'poly', 'rbf', 'sigmoid'

    gamma : str or float, default='scale'
        Kernel coefficient for 'rbf', 'poly' and 'sigmoid' kernels.
        Options: 'scale', 'auto' or float value

    nu : float, default=0.5
        An upper bound on the fraction of training errors and a lower bound on the fraction
        of support vectors. Should be in the range (0, 1].

    eps : float, default=0.5
        The maximum distance between two samples for them to be considered as in the same
        neighborhood in DBSCAN.

    min_samples : int, default=10
        The number of samples in a neighborhood for a point to be considered as a core point
        in DBSCAN.

    tree_distance_metric : str, default='euclidean'
        The distance metric to use for the tree (KDTree or BallTree).

    dbscan_distance_metric : str, default='euclidean'
        The distance metric to use for DBSCAN clustering.

    tree_algorithm : str, default='kd_tree'
        The tree algorithm to use for finding nearest cluster centroids.
        Options: 'kd_tree', 'ball_tree'

    parameter_list : dict, optional
        A dictionary of dictionaries containing custom parameters for each OCSVM model.
        The outer dictionary keys are cluster IDs, and values are parameter dictionaries.
        Example: {0: {'kernel': 'rbf', 'gamma': 0.1, 'nu': 0.05},
                 1: {'kernel': 'linear', 'gamma': 'scale', 'nu': 0.1}}

    n_jobs : int, default=-1
        The number of parallel jobs to run. -1 means using all processors.

    Attributes
    ----------
    dbscan : DBSCAN
        The fitted DBSCAN clustering model.

    ocsvms : dict
        Dictionary of One-Class SVM models, one for each valid cluster detected.

    dbscan_centroids : dict
        Dictionary of cluster centroids for each valid cluster.

    cluster_points : dict
        Dictionary storing the data points belonging to each cluster.

    tree : KDTree or BallTree
        Tree structure for efficient nearest centroid search during prediction.

    cluster_sizes : dict
        Number of points in each detected cluster.

    cluster_labels : ndarray
        Cluster labels for each training point from DBSCAN.

    unique_clusters : ndarray
        Unique cluster labels identified by DBSCAN.

    valid_clusters : list
        List of valid cluster IDs (excluding noise points).

    Notes
    -----
    - Noise points from DBSCAN (cluster label -1) are not used for training OCSVMs
    - The model requires at least one valid cluster to be identified
    - Custom parameters can be provided for each cluster's OCSVM through parameter_list
    """

    def __init__(
        self,
        kernel="rbf",
        gamma="scale",
        nu=0.5,
        eps=0.5,
        min_samples=10,
        tree_distance_metric="euclidean",
        dbscan_distance_metric="euclidean",
        tree_algorithm="kd_tree",
        parameter_list=None,
        n_jobs=-1,
    ):
        self.kernel = kernel
        self.gamma = gamma
        self.nu = nu
        self.tree_algorithm = tree_algorithm
        self.tree_distance_metric = tree_distance_metric
        self.dbscan_distance_metric = dbscan_distance_metric
        self.dbscan = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            n_jobs=n_jobs,
            metric=dbscan_distance_metric,
        )
        self.parameter_list = parameter_list
        self.ocsvms = {}  # One OCSVM per cluster
        self.dbscan_centroids = {}  # To store cluster centroids
        self.cluster_points = {}  # Store points in each cluster
        self.tree = None
        # These attributes are mainly used for inspection purposes
        self.cluster_sizes = {}  # Number of points in each cluster
        self.n_jobs = n_jobs  # Store n_jobs
        self.cluster_labels = None
        self.unique_clusters = None

    def fit_cluster(
        self,
        X,
        verbose=False,
    ):
        """
        Perform only the clustering step of the model fitting process.

        This method allows for separate hyperparameter tuning of the clustering component.
        It fits the DBSCAN algorithm to the data and identifies clusters, but does not
        train the One-Class SVMs.

        Parameters
        ----------
        X : array-like or DataFrame of shape (n_samples, n_features)
            The training input samples.

        verbose : bool, default=False
            If True, prints detailed information about the fitting process.

        Returns
        -------
        self : object
            Returns self.

        Notes
        -----
        After calling this method, the following attributes are populated:
        - cluster_labels: Array of cluster assignments for each training point
        - unique_clusters: Array of unique cluster labels
        - cluster_sizes: Dictionary with the number of points in each cluster

        Examples
        --------
        >>> model = DBOCSVM(eps=0.5, min_samples=5)
        >>> model.fit_cluster(X_train, verbose=True)
        >>> print(f"Found {len(model.unique_clusters)} clusters")
        """

        X = X.values if isinstance(X, pd.DataFrame) else X

        if verbose:
            print("Fitting DBSCAN...")

        self.cluster_labels = self.dbscan.fit_predict(X)

        if verbose:
            print("DBSCAN Fitted...")

        self.unique_clusters = np.unique(self.cluster_labels)

        if verbose:
            print(f"Unique Clusters: {self.unique_clusters}")

        for cluster in self.unique_clusters:
            n_points = np.sum(self.cluster_labels == cluster)
            self.cluster_sizes[int(cluster)] = int(n_points)

        if verbose:
            print(f"Cluster Sizes: {self.cluster_sizes}")

        return self

    def fit_ocsvm(
        self,
        X,
        parameter_list=None,
        verbose=False,
    ):
        """
        Fit One-Class SVMs for each cluster identified by DBSCAN.

        This method trains a separate One-Class SVM for each cluster previously identified
        by the fit_cluster method. It requires that fit_cluster has been called first.

        Parameters
        ----------
        X : array-like or DataFrame of shape (n_samples, n_features)
            The training input samples.

        parameter_list : dict, optional
            Dictionary of dictionaries containing OCSVM parameters for each cluster.
            Each key corresponds to a cluster ID, and each value is a dictionary
            with OCSVM parameters (kernel, gamma, nu).
            Example: {
                0: {'kernel': 'rbf', 'gamma': 0.1, 'nu': 0.05},
                1: {'kernel': 'linear', 'gamma': 'auto', 'nu': 0.1}
            }

        verbose : bool, default=False
            If True, prints detailed information about the fitting process.

        Returns
        -------
        self : object
            Returns self.

        Raises
        ------
        ValueError
            If parameter_list is None or if the number of parameters is less than
            the number of identified clusters (excluding noise).

        Notes
        -----
        - Requires that fit_cluster has been called first to establish cluster assignments
        - Noise points (cluster label -1) are excluded from OCSVM training
        - After calling this method, the following attributes are populated:
          * ocsvms: Dictionary of trained One-Class SVM models
          * dbscan_centroids: Dictionary of cluster centroids
          * cluster_points: Dictionary of points belonging to each cluster
          * tree: KDTree or BallTree for efficient nearest centroid search

        Examples
        --------
        >>> model = DBOCSVM(eps=0.5, min_samples=5)
        >>> model.fit_cluster(X_train)
        >>> params = {
        ...     0: {'kernel': 'rbf', 'gamma': 0.1, 'nu': 0.05},
        ...     1: {'kernel': 'rbf', 'gamma': 0.2, 'nu': 0.1}
        ... }
        >>> model.fit_ocsvm(X_train, parameter_list=params, verbose=True)
        """

        X = X.values if isinstance(X, pd.DataFrame) else X

        if parameter_list is None:
            raise ValueError("parameter_list cannot be None")

        if len(parameter_list) < len(self.unique_clusters) - 1:
            raise ValueError(
                "Number of parameters should be equal or greater than the number of clusters"
            )

        def filter_parameters_by_cluster(parameters, valid_clusters):
            """Filter the parameters dictionary to only include keys from valid_clusters"""
            return {
                cluster: parameters[cluster]
                for cluster in valid_clusters
                if cluster in parameters
            }

        if len(parameter_list) >= len(self.unique_clusters) - 1:
            cluster_count = list(self.cluster_sizes.keys())
            if -1 in cluster_count:
                cluster_count.remove(-1)

            parameter_list = filter_parameters_by_cluster(parameter_list, cluster_count)

        for cluster in self.unique_clusters:

            if cluster == -1:
                continue

            if verbose:
                print(
                    f"Training for cluster {cluster} with {self.cluster_sizes[cluster]} points"
                )

            # Boolean masking to get points in the current cluster
            points = X[self.cluster_labels == cluster]
            self.cluster_points[cluster] = points

            if len(points) > 0:
                if parameter_list is None:
                    if verbose:
                        print("Using default parameters")

                    ocsvm = OneClassSVM(
                        kernel=self.kernel,
                        nu=self.nu,
                        gamma=self.gamma,
                    )
                else:
                    if verbose:
                        print("Using parameters from parameter_list")

                    ocsvm = OneClassSVM(
                        kernel=parameter_list[cluster]["kernel"],
                        nu=parameter_list[cluster]["nu"],
                        gamma=parameter_list[cluster]["gamma"],
                    )

                    if verbose:
                        print(
                            f"OCSVM for cluster {cluster} uses nu: {parameter_list[cluster]['nu']}, gamma: {parameter_list[cluster]['gamma']}, kernel: {parameter_list[cluster]['kernel']}"
                        )

                ocsvm.fit(points)

                self.ocsvms[cluster] = ocsvm

                """
                TODO: Explore other alternatives for centroid calculation
                "->" means the following line might be a downside of the current approach.

                - Median: More robust to outliers than the mean (`np.median(points, axis=0)`).
                    -> Less representative if data is asymmetric
                - Trimmed Mean: Removes extreme values before computing the mean (`scipy.stats.trim_mean`).
                    ->   Requires choosing the trimming percentage
                - Weighted Mean: Assigns importance to points based on reliability.
                    ->  Requires defining weights
                - Geometric Median: Minimizes sum of distances to all points. More robust to outliers than the mean.
                    -> computationally expensive (`scipy.spatial`)
                - Distance Metrics: Use median for Manhattan distance and mean for Euclidean distance.
                    -> Requires choosing the distance metric
                """
                self.dbscan_centroids[cluster] = np.mean(points, axis=0)

        # Build tree with cluster centroids
        centroids = [self.dbscan_centroids[c] for c in self.dbscan_centroids if c != -1]
        self.valid_clusters = list(self.dbscan_centroids.keys())
        if len(centroids) > 0:
            centroids = np.array(centroids)
            if self.tree_algorithm == "kd_tree":
                self.tree = KDTree(
                    centroids,
                    metric=self.tree_distance_metric,
                )
            elif self.tree_algorithm == "ball_tree":
                self.tree = BallTree(
                    centroids,
                    metric=self.tree_distance_metric,
                )

    def fit(
        self,
        X,
        verbose=False,
    ):
        """
        Fit the complete DB-OCSVM model.

        This method performs both clustering with DBSCAN and trains One-Class SVMs
        for each identified cluster in a single step.

        Parameters
        ----------
        X : array-like or DataFrame of shape (n_samples, n_features)
            The training input samples.

        verbose : bool, default=False
            If True, prints detailed information about the fitting process.

        Returns
        -------
        self : object
            Returns self.

        Raises
        ------
        ValueError
            If parameter_list is provided but contains fewer entries than the number
            of identified clusters (excluding noise).

        Notes
        -----
        This method combines the functionality of fit_cluster and fit_ocsvm into one step.
        It's more convenient for direct model training but less flexible for hyperparameter tuning.

        Examples
        --------
        >>> model = DBOCSVM(
        ...     eps=0.5,
        ...     min_samples=5,
        ...     parameter_list={
        ...         0: {'kernel': 'rbf', 'gamma': 0.1, 'nu': 0.05},
        ...         1: {'kernel': 'rbf', 'gamma': 0.2, 'nu': 0.1}
        ...     }
        ... )
        >>> model.fit(X_train, verbose=True)
        >>> predictions = model.predict(X_test)
        """

        X = X.values if isinstance(X, pd.DataFrame) else X

        if verbose:
            print("Fitting DBSCAN...")

        cluster_labels = self.dbscan.fit_predict(X)

        if verbose:
            print("DBSCAN Fitted...")

        unique_clusters = np.unique(cluster_labels)

        if verbose:
            print(f"Unique Clusters: {unique_clusters}")

        for cluster in unique_clusters:
            n_points = np.sum(cluster_labels == cluster)
            self.cluster_sizes[int(cluster)] = int(n_points)

        if verbose:
            print(f"Cluster Sizes: {self.cluster_sizes}")

        if self.parameter_list is None:
            print(
                "Warning: parameter_list is None. Using default parameters for all OCSVMs"
            )

        if self.parameter_list is not None and (len(self.parameter_list)) < (
            len(unique_clusters) - 1
        ):
            raise ValueError(
                "Number of parameters should be equal or greater than the number of clusters"
            )

        def filter_parameters_by_cluster(parameters, valid_clusters):
            """Filter the parameters dictionary to only include keys from valid_clusters"""
            return {
                cluster: parameters[cluster]
                for cluster in valid_clusters
                if cluster in parameters
            }

        # If parameter_list is provided and has enough entries for all valid clusters,
        # filter it to match only the existing cluster IDs
        if parameter_list is not None and (len(parameter_list)) >= (
            len(unique_clusters) - 1
        ):
            # Get all cluster IDs from our clustering results
            cluster_count = list(self.cluster_sizes.keys())
            # Remove noise cluster (labeled as -1) if present
            if (
                -1 in cluster_count
            ):  # Fixed variable name from 'cluster' to 'cluster_count'
                cluster_count.remove(-1)

            # Filter the parameter list to only include parameters for detected clusters
            parameter_list = filter_parameters_by_cluster(parameter_list, cluster_count)

            # Store the final parameter list for later reference
            self.parameter_list = parameter_list

        # Store cluster labels for potential later use
        self.cluster_labels = cluster_labels
        self.unique_clusters = unique_clusters

        # Train an OCSVM for each valid cluster
        for cluster in unique_clusters:
            # Skip the noise cluster (DBSCAN labels outliers as -1)
            if cluster == -1:  # Skip noise cluster for SVM training
                continue

            if verbose:
                print(
                    f"Training for cluster {cluster} with {self.cluster_sizes[cluster]} points"
                )

            # Extract data points belonging to the current cluster
            points = X[cluster_labels == cluster]
            self.cluster_points[cluster] = points

            if len(points) > 0:
                # Only proceed if the cluster has points

                # Initialize the OCSVM model with appropriate parameters
                if self.parameter_list is None:
                    if verbose:
                        print(
                            "Using default OCSVM parameters from class initialization"
                        )

                    ocsvm = OneClassSVM(
                        kernel=self.kernel,
                        nu=self.nu,
                        gamma=self.gamma,
                    )
                else:
                    if verbose:
                        print("Using parameters from parameter_list")

                    ocsvm = OneClassSVM(
                        kernel=parameter_list[cluster]["kernel"],
                        nu=parameter_list[cluster]["nu"],
                        gamma=parameter_list[cluster]["gamma"],
                    )

                    if verbose:
                        print(
                            f"OCSVM for cluster {cluster} uses nu: {parameter_list[cluster]['nu']}, gamma: {parameter_list[cluster]['gamma']}, kernel: {parameter_list[cluster]['kernel']}"
                        )

                ocsvm.fit(points)

                self.ocsvms[cluster] = ocsvm

                self.dbscan_centroids[cluster] = np.mean(points, axis=0)

        # Build tree with cluster centroids
        centroids = [self.dbscan_centroids[c] for c in self.dbscan_centroids if c != -1]
        self.valid_clusters = list(self.dbscan_centroids.keys())
        if len(centroids) > 0:
            centroids = np.array(centroids)
            if self.tree_algorithm == "kd_tree":
                self.tree = KDTree(
                    centroids,
                    metric=self.tree_distance_metric,
                )
            elif self.tree_algorithm == "ball_tree":
                self.tree = BallTree(
                    centroids,
                    metric=self.tree_distance_metric,
                )

    def predict(self, X):
        """
        Predict if samples are outliers or not.

        For each input sample, the method:
        1. Finds the nearest cluster centroid
        2. Applies the corresponding One-Class SVM to make a prediction

        Parameters
        ----------
        X : array-like or DataFrame of shape (n_samples, n_features)
            The input samples to predict.

        Returns
        -------
        y : ndarray of shape (n_samples,)
            The predicted classes:
            - 1 for inliers (normal samples)
            - -1 for outliers (anomalies)

        Notes
        -----
        - If the model has no trained tree (no valid clusters), all samples are
          classified as outliers (-1)
        - If a point's nearest cluster has no associated OCSVM, it is classified as
          an outlier (-1)

        Examples
        --------
        >>> model = DBOCSVM(eps=0.5, min_samples=5)
        >>> model.fit(X_train)
        >>> predictions = model.predict(X_test)
        >>> print(f"Found {sum(predictions > 0)} normal samples and {sum(predictions < 0)} anomalies")
        """
        predictions = np.ones(len(X))
        X = X.values if isinstance(X, pd.DataFrame) else X

        if self.tree is None:
            return -1 * np.ones(len(X))

        # Find nearest centroid
        _, ind = self.tree.query(X, k=1)
        nearest_clusters = [self.valid_clusters[i] for i in ind.flatten()]

        for i, cluster in enumerate(nearest_clusters):
            if cluster in self.ocsvms:
                predictions[i] = self.ocsvms[cluster].predict([X[i]])[0]
            else:
                predictions[i] = -1  # Anomaly if no OCSVM for cluster

        return predictions

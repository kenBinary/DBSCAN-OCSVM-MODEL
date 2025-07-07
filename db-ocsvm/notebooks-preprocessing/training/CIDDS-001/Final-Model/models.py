import torch
import torch.nn as nn
from typing import List, Optional
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.svm import OneClassSVM
from sklearn.neighbors import KDTree, BallTree
import pandas as pd
import pprint
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)


# Added dropout to prevent overfitting
# Add negative slope to LeakyReLU
# Add GELU activation function
class BatchNormAutoencoderV2(nn.Module):
    def __init__(
        self,
        input_dim: int = 128,
        hidden_dims: List[int] = [64, 32],
        latent_dim: int = 16,
        activation_type: str = "ReLU",
        negative_slope: float = 0.2,
        dropout_rate: float = 0.2,  # Added dropout
        output_activation_type: Optional[str] = "Sigmoid",  # Default for [0,1] data
    ) -> None:
        super(BatchNormAutoencoderV2, self).__init__()

        # Select activation function
        activation: nn.Module
        if activation_type == "ReLU":
            activation = nn.ReLU()
        elif activation_type == "LeakyReLU":
            activation = nn.LeakyReLU(negative_slope)  # Better negative gradient
        elif activation_type == "ELU":
            activation = nn.ELU()
        elif activation_type == "GELU":  # Adding GELU option
            activation = nn.GELU()
        else:
            raise ValueError("Unknown activation type provided")

        # Build encoder
        encoder_layers: List[nn.Module] = []
        current_dim: int = input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(current_dim, h_dim))
            encoder_layers.append(nn.BatchNorm1d(h_dim))
            encoder_layers.append(activation)
            encoder_layers.append(nn.Dropout(dropout_rate))  # Add dropout
            current_dim = h_dim

        # Latent layer
        encoder_layers.append(nn.Linear(current_dim, latent_dim))
        self.encoder: nn.Sequential = nn.Sequential(*encoder_layers)

        # Select output activation function
        output_activation: Optional[nn.Module] = None
        if output_activation_type == "ReLU":
            output_activation = nn.ReLU()
        elif output_activation_type == "LeakyReLU":
            output_activation = nn.LeakyReLU(0.2)
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
            decoder_layers.append(nn.Dropout(dropout_rate))  # Add dropout
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
    def __init__(
        self,
        kernel="rbf",  # or 'linear', 'poly', 'sigmoid'
        gamma="scale",  # or 'auto' or a float
        nu=0.5,  # float between 0 and 1
        cache_size=200,  # in MB, bigger values use more memory which can speed up training
        eps=0.5,
        min_samples=10,
        dbscan_metric="euclidean",  # cosine, manhattan
        algorithm="kd_tree",  # or 'ball_tree'
        tree_metric="euclidean",  # BallTree.valid_metrics to see all valid metrics
        leaf_size=30,
        n_jobs=-1,
    ):
        self.kernel = kernel
        self.gamma = gamma
        self.nu = nu
        self.cache_size = cache_size
        self.dbscan = DBSCAN(
            eps=eps, min_samples=min_samples, n_jobs=n_jobs, metric=dbscan_metric
        )
        self.algorithm = algorithm
        self.tree_metric = tree_metric
        self.leaf_size = leaf_size
        self.svms = {}  # One SVM per cluster
        self.dbscan_centroids = {}  # To store cluster centroids
        self.cluster_points = {}  # Store points in each cluster
        self.tree = None
        # These attributes are mainly used for inspection purposes
        self.cluster_sizes = {}  # Number of points in each cluster
        self.cluster_labels = None
        self.unique_clusters = None

    def fit(
        self,
        X,
        parameter_list=None,
        verbose=False,
    ):
        """
        Parameters:
        -----------
        X : array-like
            Training data
        parameter_list: dictionary of dictionaries
            Each key in the dictionary is the cluster number and
            the value is a dictionary containing the parameters for OCSVM
            each dictionary looks like this:
            {
                0 : {
                kernel: rbf, linear, poly, or sigmoid,
                gamma: 'scale', 'auto' or a float,
                nu: a float between 0 and 1 e.g 0.2,
                }
            }
        verbose: bool
            Whether to print progress messages or not
        """

        X = X.values if isinstance(X, pd.DataFrame) else X

        if verbose:
            pprint.pprint("Fitting DBSCAN...")

        # NOTE: we use the dbscan that was initialized in the constructor
        cluster_labels = self.dbscan.fit_predict(X)

        if verbose:
            pprint.pprint("DBSCAN Fitted...")

        self.unique_clusters = np.unique(cluster_labels)

        if verbose:
            pprint.pprint(f"Unique Clusters: {self.unique_clusters}")

        for cluster in self.unique_clusters:
            # Store the number of points per cluster
            n_points = np.sum(cluster_labels == cluster)
            self.cluster_sizes[int(cluster)] = int(n_points)

        if verbose:
            pprint.pprint(f"Cluster Sizes: {self.cluster_sizes}")

        if parameter_list is not None and (len(parameter_list)) < (
            len(self.unique_clusters) - 1
        ):
            raise ValueError(
                "Number of parameters should be equal or greater than the number of clusters"
            )

        def filter_dict(original_dict, keys_to_keep):
            return {k: original_dict[k] for k in keys_to_keep if k in original_dict}

        if parameter_list is not None and (len(parameter_list)) >= (
            len(self.unique_clusters) - 1
        ):
            cluster_count = list(self.cluster_sizes.keys())
            if -1 in cluster_count:
                cluster_count.remove(-1)
                cluster_count

            parameter_list = filter_dict(parameter_list, cluster_count)

        self.parameter_list = parameter_list

        for cluster in self.unique_clusters:

            if cluster == -1:  # Skip noise cluster for SVM training
                continue

            if verbose:
                pprint.pprint(
                    f"Training for cluster {cluster} with {self.cluster_sizes[cluster]} points"
                )

            # Boolean masking to get points in the current cluster
            points = X[cluster_labels == cluster]
            self.cluster_points[cluster] = points

            if len(points) > 0:
                # use parameters defined in constructor if not provided
                if parameter_list is None:
                    ocsvm = OneClassSVM(
                        kernel=self.kernel,
                        nu=self.nu,
                        gamma=self.gamma,
                        cache_size=self.cache_size,
                    )
                else:
                    ocsvm = OneClassSVM(
                        kernel=parameter_list[cluster]["kernel"],
                        nu=parameter_list[cluster]["nu"],
                        gamma=parameter_list[cluster]["gamma"],
                        cache_size=self.cache_size,
                    )

                    if verbose:
                        pprint.pprint(
                            f"OCSVM for cluster {cluster} uses nu: {parameter_list[cluster]['nu']}, gamma: {parameter_list[cluster]['gamma']}, kernel: {parameter_list[cluster]['kernel']}"
                        )

                ocsvm.fit(points)

                self.svms[cluster] = ocsvm

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
            if self.algorithm == "kd_tree":
                self.tree = KDTree(
                    centroids, leaf_size=self.leaf_size, metric=self.tree_metric
                )
            elif self.algorithm == "ball_tree":
                self.tree = BallTree(
                    centroids, leaf_size=self.leaf_size, metric=self.tree_metric
                )

    def predict(self, X):
        predictions = np.ones(len(X))
        X = X.values if isinstance(X, pd.DataFrame) else X

        if self.tree is None:
            pprint.pprint("Model not yet fitted")
            return -1 * np.ones(len(X))

        # Find nearest centroid
        dist, ind = self.tree.query(X, k=1)
        nearest_clusters = [self.valid_clusters[i] for i in ind.flatten()]

        for i, cluster in enumerate(nearest_clusters):
            if cluster in self.svms:
                predictions[i] = self.svms[cluster].predict([X[i]])[0]
            else:
                predictions[i] = -1  # Anomaly if no SVM for cluster

        return predictions


class DBOCSVM_V2:
    def __init__(
        self,
        kernel="rbf",
        gamma="scale",
        nu=0.5,
        eps=0.5,
        min_samples=10,
        tree_metric="euclidean",
        dbscan_metric="euclidean",
        algorithm="kd_tree",  # or 'ball_tree'
        n_jobs=-1,  # Add n_jobs parameter
    ):
        self.kernel = kernel
        self.gamma = gamma
        self.nu = nu
        self.algorithm = algorithm
        self.tree_metric = tree_metric
        self.dbscan_metric = dbscan_metric
        self.dbscan = DBSCAN(
            eps=eps, min_samples=min_samples, n_jobs=n_jobs, metric=dbscan_metric
        )  # Make it so that it can accept a metric parameter
        self.svms = {}  # One SVM per cluster
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
        Parameters:
        -----------
        X : array-like
            Training data
        """

        X = X.values if isinstance(X, pd.DataFrame) else X

        """
        NOTE: Current DBSCAN only uses euclidean distance, so the metric parameter is not used
        TODO: Add metric parameter to DBSCAN to handle different distance metrics
        'euclidean': Standard Euclidean distance. This is the default metric.
        'manhattan': Manhattan or L1 distance (sum of absolute differences).
        'chebyshev': Chebyshev or maximum distance.
        'minkowski': Minkowski distance, a generalization of Euclidean and Manhattan distance. The power parameter p of the Minkowski metric can be controlled by the p parameter of DBSCAN.
        'wminkowski': Weighted Minkowski distance.
        'seuclidean': Standardized Euclidean distance.
        'mahalanobis': Mahalanobis distance.
        """

        if verbose:
            print("Fitting DBSCAN...")
        # NOTE: we use the dbscan that was initialized in the constructor
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

    def fit_ocsvm(
        self,
        X,
        parameter_list=None,
        verbose=False,
    ):
        """
        Parameters:
        -----------
        X : array-like
            Training data
        parameter_list: dictionary of dictionaries
            Each key in the dictionary is the cluster number and
            the value is a dictionary containing the parameters for OCSVM
            each dictionary looks like this:
            {
                0 : {
                kernel: rbf, linear, poly, or sigmoid,
                gamma: 'scale', 'auto' or a float,
                nu: a float between 0 and 1 e.g 0.2,
                }
            }
        """
        X = X.values if isinstance(X, pd.DataFrame) else X

        if parameter_list is None:
            raise ValueError("parameter_list cannot be None")

        if len(parameter_list) < len(self.unique_clusters) - 1:
            raise ValueError(
                "Number of parameters should be equal or greater than the number of clusters"
            )

        def filter_dict(original_dict, keys_to_keep):
            return {k: original_dict[k] for k in keys_to_keep if k in original_dict}

        if len(parameter_list) >= len(self.unique_clusters) - 1:
            cluster_count = list(self.cluster_sizes.keys())
            if -1 in cluster_count:
                cluster_count.remove(-1)
            cluster_count

            parameter_list = filter_dict(parameter_list, cluster_count)

        for cluster in self.unique_clusters:

            if cluster == -1:  # Skip noise cluster for SVM training
                continue

            if verbose:
                print(
                    f"Training for cluster {cluster} with {self.cluster_sizes[cluster]} points"
                )

            # Boolean masking to get points in the current cluster
            points = X[self.cluster_labels == cluster]
            self.cluster_points[cluster] = points

            if len(points) > 0:
                # use parameters defined in constructor if not provided
                if parameter_list is None:
                    ocsvm = OneClassSVM(
                        kernel=self.kernel,
                        nu=self.nu,
                        gamma=self.gamma,
                    )
                else:
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

                self.svms[cluster] = ocsvm

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
            if self.algorithm == "kd_tree":
                self.tree = KDTree(centroids, metric=self.tree_metric)
            elif self.algorithm == "ball_tree":
                self.tree = BallTree(centroids, metric=self.tree_metric)

    def fit(
        self,
        X,
        dbscan_evaluation_metric="silhouette",  # only used for reruns
        dbscan_rerun=False,  # only used for reruns
        dbscan_rerun_trials=10,  # only used for reruns
        parameter_list=None,
        verbose=False,
    ):
        """
        Parameters:
        -----------
        X : array-like
            Training data
        dbscan_evaluation_metric : str
            Metric to optimize ('silhouette', 'davies_bouldin', or 'calinski_harabasz')
        dbscan_rerun : bool
            Whether to rerun DBSCAN after fitting the model with the best parameters
        dbscan_rerun_trials : int
            Number of reruns for DBSCAN after fitting the model with the best parameters
        parameter_list: dictionary of dictionaries
            Each key in the dictionary is the cluster number and
            the value is a dictionary containing the parameters for OCSVM
            each dictionary looks like this:
            {
                0 : {
                kernel: rbf, linear, poly, or sigmoid,
                gamma: 'scale', 'auto' or a float,
                nu: a float between 0 and 1 e.g 0.2,
                }
            }
        """

        X = X.values if isinstance(X, pd.DataFrame) else X

        """
        NOTE: Current DBSCAN only uses euclidean distance, so the metric parameter is not used
        TODO: Add metric parameter to DBSCAN to handle different distance metrics
        'euclidean': Standard Euclidean distance. This is the default metric.
        'manhattan': Manhattan or L1 distance (sum of absolute differences).
        'chebyshev': Chebyshev or maximum distance.
        'minkowski': Minkowski distance, a generalization of Euclidean and Manhattan distance. The power parameter p of the Minkowski metric can be controlled by the p parameter of DBSCAN.
        'wminkowski': Weighted Minkowski distance.
        'seuclidean': Standardized Euclidean distance.
        'mahalanobis': Mahalanobis distance.
        """
        if verbose:
            print("Fitting DBSCAN...")
        # NOTE: we use the dbscan that was initialized in the constructor
        cluster_labels = self.dbscan.fit_predict(X)
        if verbose:
            print("DBSCAN Fitted...")

        if dbscan_rerun:
            if verbose:
                print("Rerunning DBSCAN...")

            if dbscan_evaluation_metric == "silhouette":
                current_score = silhouette_score(X, cluster_labels)
            elif dbscan_evaluation_metric == "davies_bouldin":
                current_score = davies_bouldin_score(X, cluster_labels)
            else:  # calinski_harabasz
                current_score = calinski_harabasz_score(X, cluster_labels)

            for i in range(dbscan_rerun_trials):
                if verbose:
                    print(f"DBSCAN Rerun {i+1}...")

                new_cluster_labels = self.dbscan.fit_predict(X)

                if dbscan_evaluation_metric == "silhouette":
                    new_score = silhouette_score(X, new_cluster_labels)
                    if new_score > current_score:
                        cluster_labels = new_cluster_labels
                        current_score = new_score
                elif dbscan_evaluation_metric == "davies_bouldin":
                    new_score = davies_bouldin_score(X, new_cluster_labels)
                    if new_score < current_score:
                        cluster_labels = new_cluster_labels
                        current_score = new_score
                else:  # calinski_harabasz
                    new_score = calinski_harabasz_score(X, new_cluster_labels)
                    if new_score > current_score:
                        cluster_labels = new_cluster_labels
                        current_score = new_score

        unique_clusters = np.unique(cluster_labels)

        if verbose:
            print(f"Unique Clusters: {unique_clusters}")

        for cluster in unique_clusters:
            # Store the number of points in the cluster
            # mainly for inspection purposes
            n_points = np.sum(cluster_labels == cluster)
            self.cluster_sizes[int(cluster)] = int(n_points)

        if verbose:
            print(f"Cluster Sizes: {self.cluster_sizes}")

        if parameter_list is not None and (len(parameter_list)) < (
            len(unique_clusters) - 1
        ):
            raise ValueError(
                "Number of parameters should be equal or greater than the number of clusters"
            )

        def filter_dict(original_dict, keys_to_keep):
            return {k: original_dict[k] for k in keys_to_keep if k in original_dict}

        if parameter_list is not None and (len(parameter_list)) >= (
            len(unique_clusters) - 1
        ):
            cluster_count = list(self.cluster_sizes.keys())
            cluster_count.remove(-1)
            cluster_count

            parameter_list = filter_dict(parameter_list, cluster_count)

        self.parameter_list = parameter_list

        for cluster in unique_clusters:

            # Store the number of points in the cluster
            # n_points = np.sum(cluster_labels == cluster)
            # self.cluster_sizes[int(cluster)] = int(n_points)

            if cluster == -1:  # Skip noise cluster for SVM training
                continue

            if verbose:
                print(
                    f"Training for cluster {cluster} with {self.cluster_sizes[cluster]} points"
                )

            # Boolean masking to get points in the current cluster
            points = X[cluster_labels == cluster]
            self.cluster_points[cluster] = points

            if len(points) > 0:
                # use parameters defined in constructor if not provided
                if parameter_list is None:
                    ocsvm = OneClassSVM(
                        kernel=self.kernel,
                        nu=self.nu,
                        gamma=self.gamma,
                        degree=self.degree,
                        coef0=self.coef0,
                        tol=self.tol,
                        shrinking=self.shrinking,
                        cache_size=self.cache_size,
                        max_iter=self.max_iter,
                    )
                else:
                    ocsvm = OneClassSVM(
                        kernel=parameter_list[cluster]["kernel"],
                        nu=parameter_list[cluster]["nu"],
                        gamma=parameter_list[cluster]["gamma"],
                        degree=self.degree,
                        coef0=self.coef0,
                        tol=self.tol,
                        shrinking=self.shrinking,
                        cache_size=self.cache_size,
                        max_iter=self.max_iter,
                    )
                    if verbose:
                        print(
                            f"OCSVM for cluster {cluster} uses nu: {parameter_list[cluster]['nu']}, gamma: {parameter_list[cluster]['gamma']}, kernel: {parameter_list[cluster]['kernel']}"
                        )
                ocsvm.fit(points)

                self.svms[cluster] = ocsvm

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
            if self.algorithm == "kd_tree":
                self.tree = KDTree(
                    centroids, leaf_size=self.leaf_size, metric=self.tree_metric
                )
            elif self.algorithm == "ball_tree":
                self.tree = BallTree(
                    centroids, leaf_size=self.leaf_size, metric=self.tree_metric
                )

    def predict(self, X):
        predictions = np.ones(len(X))
        X = X.values if isinstance(X, pd.DataFrame) else X

        if self.tree is None:
            return -1 * np.ones(len(X))

        # Find nearest centroid
        dist, ind = self.tree.query(X, k=1)
        nearest_clusters = [self.valid_clusters[i] for i in ind.flatten()]

        for i, cluster in enumerate(nearest_clusters):
            if cluster in self.svms:
                predictions[i] = self.svms[cluster].predict([X[i]])[0]
            else:
                predictions[i] = -1  # Anomaly if no SVM for cluster

        return predictions

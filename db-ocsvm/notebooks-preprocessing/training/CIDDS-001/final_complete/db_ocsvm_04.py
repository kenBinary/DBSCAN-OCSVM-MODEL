import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.svm import OneClassSVM
from sklearn.neighbors import KDTree, BallTree
import pandas as pd


# class DBOCSVM:
#     """
#     Parameters
#     ----------
#     kernel : str, default='rbf'
#         Specifies the kernel type to be used in the One-Class SVM.
#         Options: 'linear', 'poly', 'rbf', 'sigmoid'

#     gamma : str or float, default='scale'
#         Kernel coefficient for 'rbf', 'poly' and 'sigmoid' kernels.
#         Options: 'scale', 'auto' or float value

#     nu : float, default=0.5
#         An upper bound on the fraction of training errors and a lower bound on the fraction
#         of support vectors. Should be in the range (0, 1].

#     eps : float, default=0.5
#         The maximum distance between two samples for them to be considered as in the same
#         neighborhood in DBSCAN.

#     min_samples : int, default=10
#         The number of samples in a neighborhood for a point to be considered as a core point
#         in DBSCAN.

#     tree_distance_metric : str, default='euclidean'
#         The distance metric to use for the tree (KDTree or BallTree).

#     dbscan_distance_metric : str, default='euclidean'
#         The distance metric to use for DBSCAN clustering.

#     tree_algorithm : str, default='kd_tree'
#         The tree algorithm to use for finding nearest cluster centroids.
#         Options: 'kd_tree', 'ball_tree'

#     parameter_list : dict, optional
#         A dictionary of dictionaries containing custom parameters for each OCSVM model.
#         The outer dictionary keys are cluster IDs, and values are parameter dictionaries.
#         Example: {0: {'kernel': 'rbf', 'gamma': 0.1, 'nu': 0.05},
#                  1: {'kernel': 'linear', 'gamma': 'scale', 'nu': 0.1}}

#     n_jobs : int, default=-1
#         The number of parallel jobs to run. -1 means using all processors.

#     Attributes
#     ----------
#     dbscan : DBSCAN
#         The fitted DBSCAN clustering model.

#     ocsvms : dict
#         Dictionary of One-Class SVM models, one for each valid cluster detected.

#     dbscan_centroids : dict
#         Dictionary of cluster centroids for each valid cluster.

#     cluster_points : dict
#         Dictionary storing the data points belonging to each cluster.

#     tree : KDTree or BallTree
#         Tree structure for efficient nearest centroid search during prediction.

#     cluster_sizes : dict
#         Number of points in each detected cluster.

#     cluster_labels : ndarray
#         Cluster labels for each training point from DBSCAN.

#     unique_clusters : ndarray
#         Unique cluster labels identified by DBSCAN.

#     valid_clusters : list
#         List of valid cluster IDs (excluding noise points).

#     Notes
#     -----
#     - Noise points from DBSCAN (cluster label -1) are not used for training OCSVMs
#     - The model requires at least one valid cluster to be identified
#     - Custom parameters can be provided for each cluster's OCSVM through parameter_list
#     """

#     def __init__(
#         self,
#         kernel="rbf",
#         gamma="scale",
#         nu=0.5,
#         eps=0.5,
#         min_samples=10,
#         tree_distance_metric="euclidean",
#         dbscan_distance_metric="euclidean",
#         tree_algorithm="kd_tree",
#         parameter_list=None,
#         n_jobs=-1,
#     ):
#         self.kernel = kernel
#         self.gamma = gamma
#         self.nu = nu
#         self.tree_algorithm = tree_algorithm
#         self.tree_distance_metric = tree_distance_metric
#         self.dbscan_distance_metric = dbscan_distance_metric
#         self.dbscan = DBSCAN(
#             eps=eps,
#             min_samples=min_samples,
#             n_jobs=n_jobs,
#             metric=dbscan_distance_metric,
#         )
#         self.parameter_list = parameter_list
#         self.ocsvms = {}  # One OCSVM per cluster
#         self.dbscan_centroids = {}  # To store cluster centroids
#         self.cluster_points = {}  # Store points in each cluster
#         self.tree = None
#         # These attributes are mainly used for inspection purposes
#         self.cluster_sizes = {}  # Number of points in each cluster
#         self.n_jobs = n_jobs  # Store n_jobs
#         self.cluster_labels = None
#         self.unique_clusters = None

#     def fit_cluster(
#         self,
#         X,
#         verbose=False,
#     ):
#         """
#         Perform only the clustering step of the model fitting process.

#         This method allows for separate hyperparameter tuning of the clustering component.
#         It fits the DBSCAN algorithm to the data and identifies clusters, but does not
#         train the One-Class SVMs.

#         Parameters
#         ----------
#         X : array-like or DataFrame of shape (n_samples, n_features)
#             The training input samples.

#         verbose : bool, default=False
#             If True, prints detailed information about the fitting process.

#         Returns
#         -------
#         self : object
#             Returns self.

#         Notes
#         -----
#         After calling this method, the following attributes are populated:
#         - cluster_labels: Array of cluster assignments for each training point
#         - unique_clusters: Array of unique cluster labels
#         - cluster_sizes: Dictionary with the number of points in each cluster

#         Examples
#         --------
#         >>> model = DBOCSVM(eps=0.5, min_samples=5)
#         >>> model.fit_cluster(X_train, verbose=True)
#         >>> print(f"Found {len(model.unique_clusters)} clusters")
#         """

#         X = X.values if isinstance(X, pd.DataFrame) else X

#         if verbose:
#             print("Fitting DBSCAN...")

#         self.cluster_labels = self.dbscan.fit_predict(X)

#         if verbose:
#             print("DBSCAN Fitted...")

#         self.unique_clusters = np.unique(self.cluster_labels)

#         if verbose:
#             print(f"Unique Clusters: {self.unique_clusters}")

#         for cluster in self.unique_clusters:
#             n_points = np.sum(self.cluster_labels == cluster)
#             self.cluster_sizes[int(cluster)] = int(n_points)

#         if verbose:
#             print(f"Cluster Sizes: {self.cluster_sizes}")

#         return self

#     def fit_ocsvm(
#         self,
#         X,
#         parameter_list=None,
#         verbose=False,
#     ):
#         """
#         Fit One-Class SVMs for each cluster identified by DBSCAN.

#         This method trains a separate One-Class SVM for each cluster previously identified
#         by the fit_cluster method. It requires that fit_cluster has been called first.

#         Parameters
#         ----------
#         X : array-like or DataFrame of shape (n_samples, n_features)
#             The training input samples.

#         parameter_list : dict, optional
#             Dictionary of dictionaries containing OCSVM parameters for each cluster.
#             Each key corresponds to a cluster ID, and each value is a dictionary
#             with OCSVM parameters (kernel, gamma, nu).
#             Example: {
#                 0: {'kernel': 'rbf', 'gamma': 0.1, 'nu': 0.05},
#                 1: {'kernel': 'linear', 'gamma': 'auto', 'nu': 0.1}
#             }

#         verbose : bool, default=False
#             If True, prints detailed information about the fitting process.

#         Returns
#         -------
#         self : object
#             Returns self.

#         Raises
#         ------
#         ValueError
#             If parameter_list is None or if the number of parameters is less than
#             the number of identified clusters (excluding noise).

#         Notes
#         -----
#         - Requires that fit_cluster has been called first to establish cluster assignments
#         - Noise points (cluster label -1) are excluded from OCSVM training
#         - After calling this method, the following attributes are populated:
#           * ocsvms: Dictionary of trained One-Class SVM models
#           * dbscan_centroids: Dictionary of cluster centroids
#           * cluster_points: Dictionary of points belonging to each cluster
#           * tree: KDTree or BallTree for efficient nearest centroid search

#         Examples
#         --------
#         >>> model = DBOCSVM(eps=0.5, min_samples=5)
#         >>> model.fit_cluster(X_train)
#         >>> params = {
#         ...     0: {'kernel': 'rbf', 'gamma': 0.1, 'nu': 0.05},
#         ...     1: {'kernel': 'rbf', 'gamma': 0.2, 'nu': 0.1}
#         ... }
#         >>> model.fit_ocsvm(X_train, parameter_list=params, verbose=True)
#         """

#         X = X.values if isinstance(X, pd.DataFrame) else X

#         if parameter_list is None:
#             raise ValueError("parameter_list cannot be None")

#         if len(parameter_list) < len(self.unique_clusters) - 1:
#             raise ValueError(
#                 "Number of parameters should be equal or greater than the number of clusters"
#             )

#         def filter_parameters_by_cluster(parameters, valid_clusters):
#             """Filter the parameters dictionary to only include keys from valid_clusters"""
#             return {
#                 cluster: parameters[cluster]
#                 for cluster in valid_clusters
#                 if cluster in parameters
#             }

#         if len(parameter_list) >= len(self.unique_clusters) - 1:
#             cluster_count = list(self.cluster_sizes.keys())
#             if -1 in cluster_count:
#                 cluster_count.remove(-1)

#             parameter_list = filter_parameters_by_cluster(parameter_list, cluster_count)

#         for cluster in self.unique_clusters:

#             if cluster == -1:
#                 continue

#             if verbose:
#                 print(
#                     f"Training for cluster {cluster} with {self.cluster_sizes[cluster]} points"
#                 )

#             # Boolean masking to get points in the current cluster
#             points = X[self.cluster_labels == cluster]
#             self.cluster_points[cluster] = points

#             if len(points) > 0:
#                 if parameter_list is None:
#                     if verbose:
#                         print("Using default parameters")

#                     ocsvm = OneClassSVM(
#                         kernel=self.kernel,
#                         nu=self.nu,
#                         gamma=self.gamma,
#                     )
#                 else:
#                     if verbose:
#                         print("Using parameters from parameter_list")

#                     ocsvm = OneClassSVM(
#                         kernel=parameter_list[cluster]["kernel"],
#                         nu=parameter_list[cluster]["nu"],
#                         gamma=parameter_list[cluster]["gamma"],
#                     )

#                     if verbose:
#                         print(
#                             f"OCSVM for cluster {cluster} uses nu: {parameter_list[cluster]['nu']}, gamma: {parameter_list[cluster]['gamma']}, kernel: {parameter_list[cluster]['kernel']}"
#                         )

#                 ocsvm.fit(points)

#                 self.ocsvms[cluster] = ocsvm

#                 """
#                 TODO: Explore other alternatives for centroid calculation
#                 "->" means the following line might be a downside of the current approach.

#                 - Median: More robust to outliers than the mean (`np.median(points, axis=0)`).
#                     -> Less representative if data is asymmetric
#                 - Trimmed Mean: Removes extreme values before computing the mean (`scipy.stats.trim_mean`).
#                     ->   Requires choosing the trimming percentage
#                 - Weighted Mean: Assigns importance to points based on reliability.
#                     ->  Requires defining weights
#                 - Geometric Median: Minimizes sum of distances to all points. More robust to outliers than the mean.
#                     -> computationally expensive (`scipy.spatial`)
#                 - Distance Metrics: Use median for Manhattan distance and mean for Euclidean distance.
#                     -> Requires choosing the distance metric
#                 """
#                 self.dbscan_centroids[cluster] = np.mean(points, axis=0)

#         # Build tree with cluster centroids
#         centroids = [self.dbscan_centroids[c] for c in self.dbscan_centroids if c != -1]
#         self.valid_clusters = list(self.dbscan_centroids.keys())
#         if len(centroids) > 0:
#             centroids = np.array(centroids)
#             if self.tree_algorithm == "kd_tree":
#                 self.tree = KDTree(
#                     centroids,
#                     metric=self.tree_distance_metric,
#                 )
#             elif self.tree_algorithm == "ball_tree":
#                 self.tree = BallTree(
#                     centroids,
#                     metric=self.tree_distance_metric,
#                 )

#     def fit(
#         self,
#         X,
#         verbose=False,
#     ):
#         """
#         Fit the complete DB-OCSVM model.

#         This method performs both clustering with DBSCAN and trains One-Class SVMs
#         for each identified cluster in a single step.

#         Parameters
#         ----------
#         X : array-like or DataFrame of shape (n_samples, n_features)
#             The training input samples.

#         verbose : bool, default=False
#             If True, prints detailed information about the fitting process.

#         Returns
#         -------
#         self : object
#             Returns self.

#         Raises
#         ------
#         ValueError
#             If parameter_list is provided but contains fewer entries than the number
#             of identified clusters (excluding noise).

#         Notes
#         -----
#         This method combines the functionality of fit_cluster and fit_ocsvm into one step.
#         It's more convenient for direct model training but less flexible for hyperparameter tuning.

#         Examples
#         --------
#         >>> model = DBOCSVM(
#         ...     eps=0.5,
#         ...     min_samples=5,
#         ...     parameter_list={
#         ...         0: {'kernel': 'rbf', 'gamma': 0.1, 'nu': 0.05},
#         ...         1: {'kernel': 'rbf', 'gamma': 0.2, 'nu': 0.1}
#         ...     }
#         ... )
#         >>> model.fit(X_train, verbose=True)
#         >>> predictions = model.predict(X_test)
#         """

#         X = X.values if isinstance(X, pd.DataFrame) else X

#         if verbose:
#             print("Fitting DBSCAN...")

#         cluster_labels = self.dbscan.fit_predict(X)

#         if verbose:
#             print("DBSCAN Fitted...")

#         unique_clusters = np.unique(cluster_labels)

#         if verbose:
#             print(f"Unique Clusters: {unique_clusters}")

#         for cluster in unique_clusters:
#             n_points = np.sum(cluster_labels == cluster)
#             self.cluster_sizes[int(cluster)] = int(n_points)

#         if verbose:
#             print(f"Cluster Sizes: {self.cluster_sizes}")

#         if self.parameter_list is None:
#             print(
#                 "Warning: parameter_list is None. Using default parameters for all OCSVMs"
#             )

#         if self.parameter_list is not None and (len(self.parameter_list)) < (
#             len(unique_clusters) - 1
#         ):
#             raise ValueError(
#                 "Number of parameters should be equal or greater than the number of clusters"
#             )

#         def filter_parameters_by_cluster(parameters, valid_clusters):
#             """Filter the parameters dictionary to only include keys from valid_clusters"""
#             return {
#                 cluster: parameters[cluster]
#                 for cluster in valid_clusters
#                 if cluster in parameters
#             }

#         # If parameter_list is provided and has enough entries for all valid clusters,
#         # filter it to match only the existing cluster IDs
#         if parameter_list is not None and (len(parameter_list)) >= (
#             len(unique_clusters) - 1
#         ):
#             # Get all cluster IDs from our clustering results
#             cluster_count = list(self.cluster_sizes.keys())
#             # Remove noise cluster (labeled as -1) if present
#             if (
#                 -1 in cluster_count
#             ):  # Fixed variable name from 'cluster' to 'cluster_count'
#                 cluster_count.remove(-1)

#             # Filter the parameter list to only include parameters for detected clusters
#             parameter_list = filter_parameters_by_cluster(parameter_list, cluster_count)

#             # Store the final parameter list for later reference
#             self.parameter_list = parameter_list

#         # Store cluster labels for potential later use
#         self.cluster_labels = cluster_labels
#         self.unique_clusters = unique_clusters

#         # Train an OCSVM for each valid cluster
#         for cluster in unique_clusters:
#             # Skip the noise cluster (DBSCAN labels outliers as -1)
#             if cluster == -1:  # Skip noise cluster for SVM training
#                 continue

#             if verbose:
#                 print(
#                     f"Training for cluster {cluster} with {self.cluster_sizes[cluster]} points"
#                 )

#             # Extract data points belonging to the current cluster
#             points = X[cluster_labels == cluster]
#             self.cluster_points[cluster] = points

#             if len(points) > 0:
#                 # Only proceed if the cluster has points

#                 # Initialize the OCSVM model with appropriate parameters
#                 if self.parameter_list is None:
#                     if verbose:
#                         print(
#                             "Using default OCSVM parameters from class initialization"
#                         )

#                     ocsvm = OneClassSVM(
#                         kernel=self.kernel,
#                         nu=self.nu,
#                         gamma=self.gamma,
#                     )
#                 else:
#                     if verbose:
#                         print("Using parameters from parameter_list")

#                     ocsvm = OneClassSVM(
#                         kernel=parameter_list[cluster]["kernel"],
#                         nu=parameter_list[cluster]["nu"],
#                         gamma=parameter_list[cluster]["gamma"],
#                     )

#                     if verbose:
#                         print(
#                             f"OCSVM for cluster {cluster} uses nu: {parameter_list[cluster]['nu']}, gamma: {parameter_list[cluster]['gamma']}, kernel: {parameter_list[cluster]['kernel']}"
#                         )

#                 ocsvm.fit(points)

#                 self.ocsvms[cluster] = ocsvm

#                 self.dbscan_centroids[cluster] = np.mean(points, axis=0)

#         # Build tree with cluster centroids
#         centroids = [self.dbscan_centroids[c] for c in self.dbscan_centroids if c != -1]
#         self.valid_clusters = list(self.dbscan_centroids.keys())
#         if len(centroids) > 0:
#             centroids = np.array(centroids)
#             if self.tree_algorithm == "kd_tree":
#                 self.tree = KDTree(
#                     centroids,
#                     metric=self.tree_distance_metric,
#                 )
#             elif self.tree_algorithm == "ball_tree":
#                 self.tree = BallTree(
#                     centroids,
#                     metric=self.tree_distance_metric,
#                 )

#     def predict(self, X):
#         """
#         Predict if samples are outliers or not.

#         For each input sample, the method:
#         1. Finds the nearest cluster centroid
#         2. Applies the corresponding One-Class SVM to make a prediction

#         Parameters
#         ----------
#         X : array-like or DataFrame of shape (n_samples, n_features)
#             The input samples to predict.

#         Returns
#         -------
#         y : ndarray of shape (n_samples,)
#             The predicted classes:
#             - 1 for inliers (normal samples)
#             - -1 for outliers (anomalies)

#         Notes
#         -----
#         - If the model has no trained tree (no valid clusters), all samples are
#           classified as outliers (-1)
#         - If a point's nearest cluster has no associated OCSVM, it is classified as
#           an outlier (-1)

#         Examples
#         --------
#         >>> model = DBOCSVM(eps=0.5, min_samples=5)
#         >>> model.fit(X_train)
#         >>> predictions = model.predict(X_test)
#         >>> print(f"Found {sum(predictions > 0)} normal samples and {sum(predictions < 0)} anomalies")
#         """
#         predictions = np.ones(len(X))
#         X = X.values if isinstance(X, pd.DataFrame) else X

#         if self.tree is None:
#             return -1 * np.ones(len(X))

#         # Find nearest centroid
#         _, ind = self.tree.query(X, k=1)
#         nearest_clusters = [self.valid_clusters[i] for i in ind.flatten()]

#         for i, cluster in enumerate(nearest_clusters):
#             if cluster in self.ocsvms:
#                 predictions[i] = self.ocsvms[cluster].predict([X[i]])[0]
#             else:
#                 predictions[i] = -1  # Anomaly if no OCSVM for cluster

#         return predictions


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

    def visualize_decision_boundaries(
        self,
        X,
        method="umap",
        cluster_id=None,
        figsize=(20, 15),
        perplexity=30,
        n_neighbors=15,
        min_dist=0.1,
        mesh_granularity=200,
        show_scores=True,
    ):
        """
        Visualize decision boundaries for one or all cluster-specific OCSVMs.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The data used to train the model, needed for visualization.

        method : str, default='umap'
            Dimensionality reduction method to use. Options: 'umap', 'tsne'

        cluster_id : int or None, default=None
            If provided, visualize only the specific cluster's decision boundary.
            If None, visualize all valid clusters.

        figsize : tuple, default=(20, 15)
            Figure size for the plots.

        perplexity : int, default=30
            Perplexity parameter for t-SNE (ignored if method='umap').

        n_neighbors : int, default=15
            Number of neighbors to consider for UMAP (ignored if method='tsne').

        min_dist : float, default=0.1
            Minimum distance for UMAP points (ignored if method='tsne').

        mesh_granularity : int, default=200
            Number of points in the mesh grid for decision boundary visualization.

        show_scores : bool, default=True
            If True, show an additional plot with decision scores as a heatmap.

        Returns
        -------
        dict : Dictionary mapping cluster IDs to their trained dimensionality reducers
            for potential reuse with new data.

        Notes
        -----
        This method will display:
        - For each cluster: the decision boundary with normal/anomaly regions
        - Optionally: a heatmap of decision scores
        - Points colored by their original OCSVM predictions
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        import numpy as np

        X = X.values if isinstance(X, pd.DataFrame) else X

        # If no clusters were found or no OCSVMs were trained
        if not hasattr(self, "valid_clusters") or len(self.valid_clusters) == 0:
            print("No valid clusters found. Cannot visualize decision boundaries.")
            return {}

        # Determine which clusters to visualize
        clusters_to_visualize = (
            [cluster_id] if cluster_id is not None else self.valid_clusters
        )

        # Filter clusters that don't have trained OCSVMs
        clusters_to_visualize = [c for c in clusters_to_visualize if c in self.ocsvms]

        if not clusters_to_visualize:
            print(
                "No valid clusters to visualize. Check if the specified cluster has a trained OCSVM."
            )
            return {}

        # Create reducers for each cluster
        reducers = {}

        for cluster in clusters_to_visualize:
            plt.figure(figsize=figsize)

            # Get the points for this cluster
            cluster_points = self.cluster_points[cluster]

            # Get predictions from the OCSVM for these points
            original_predictions = self.ocsvms[cluster].predict(cluster_points)

            # Apply dimensionality reduction
            if method.lower() == "tsne":
                X_reduced, reducer = self._apply_tsne(
                    cluster_points, perplexity=perplexity
                )
            elif method.lower() == "umap":
                X_reduced, reducer = self._apply_umap(
                    cluster_points, n_neighbors=n_neighbors, min_dist=min_dist
                )
            else:
                raise ValueError("Method must be either 'umap' or 'tsne'")

            reducers[cluster] = reducer

            # Create a mesh grid in the reduced space
            x_min, x_max = X_reduced[:, 0].min() - 1, X_reduced[:, 0].max() + 1
            y_min, y_max = X_reduced[:, 1].min() - 1, X_reduced[:, 1].max() + 1
            xx, yy = np.meshgrid(
                np.linspace(x_min, x_max, mesh_granularity),
                np.linspace(y_min, y_max, mesh_granularity),
            )

            # Create a separate OCSVM for visualization in the reduced space
            reduced_svm = OneClassSVM(
                kernel=self.ocsvms[cluster].kernel,
                nu=self.ocsvms[cluster].nu,
                gamma=self.ocsvms[cluster].gamma,
            )
            reduced_svm.fit(X_reduced[original_predictions == 1])

            # Get decision function values for the grid
            grid_points = np.c_[xx.ravel(), yy.ravel()]
            Z = reduced_svm.decision_function(grid_points).reshape(xx.shape)

            # Define custom colormap for visualization
            normal_color = "#8FBB99"  # Light green
            anomaly_color = "#F2BFCB"  # Light red
            cmap = ListedColormap([anomaly_color, normal_color])

            # Plot decision regions
            plt.contourf(xx, yy, Z > 0, alpha=0.5, cmap=cmap)

            # Plot the decision boundary
            plt.contour(xx, yy, Z, levels=[0], linewidths=2.5, colors="darkgreen")

            # Plot points colored by their original predictions
            plt.scatter(
                X_reduced[original_predictions == 1, 0],
                X_reduced[original_predictions == 1, 1],
                c="blue",
                s=50,
                edgecolors="k",
                label="Normal",
            )
            plt.scatter(
                X_reduced[original_predictions == -1, 0],
                X_reduced[original_predictions == -1, 1],
                c="red",
                s=50,
                edgecolors="k",
                label="Anomalies",
            )

            plt.title(
                f"Cluster {cluster} OCSVM Decision Boundary with {method.upper()}",
                fontsize=16,
            )
            plt.legend(fontsize=12)
            plt.tight_layout()
            plt.show()

            # Show decision scores as a heatmap if requested
            if show_scores:
                plt.figure(figsize=figsize)
                contour = plt.contourf(xx, yy, Z, levels=50, cmap=plt.cm.RdYlGn)
                plt.colorbar(contour, label="Decision Score")
                plt.scatter(
                    X_reduced[:, 0],
                    X_reduced[:, 1],
                    c=original_predictions,
                    cmap=plt.cm.coolwarm,
                    s=50,
                    edgecolors="k",
                    alpha=0.8,
                )
                plt.title(
                    f"Cluster {cluster} OCSVM Decision Scores with {method.upper()}",
                    fontsize=16,
                )
                plt.tight_layout()
                plt.show()

        return reducers

    def _apply_tsne(self, X, perplexity=30, random_state=42):
        """
        Apply t-SNE dimensionality reduction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        perplexity : int, default=30
            The perplexity parameter for t-SNE.

        random_state : int, default=42
            Random seed for reproducibility.

        Returns
        -------
        X_tsne : ndarray of shape (n_samples, 2)
            The reduced data.

        tsne : TSNE object
            The fitted t-SNE model.
        """
        from sklearn.manifold import TSNE

        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            random_state=random_state,
            learning_rate="auto",
            init="pca",
        )
        X_tsne = tsne.fit_transform(X)

        return X_tsne, tsne

    def _apply_umap(self, X, n_neighbors=15, min_dist=0.1, random_state=42):
        """
        Apply UMAP dimensionality reduction.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        n_neighbors : int, default=15
            Number of neighbors to consider.

        min_dist : float, default=0.1
            Minimum distance between points in the reduced space.

        random_state : int, default=42
            Random seed for reproducibility.

        Returns
        -------
        X_umap : ndarray of shape (n_samples, 2)
            The reduced data.

        reducer : UMAP object
            The fitted UMAP model.
        """
        import umap

        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=2,
            random_state=random_state,
        )
        X_umap = reducer.fit_transform(X)

        return X_umap, reducer

    def plot_cluster_embedding(
        self,
        X,
        method="umap",
        perplexity=30,
        n_neighbors=15,
        min_dist=0.1,
        figsize=(12, 10),
    ):
        """
        Create a global embedding of all data points colored by cluster assignment.

        This plot shows the overall structure of the clusters without decision boundaries.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        method : str, default='umap'
            Dimensionality reduction method to use. Options: 'umap', 'tsne'

        perplexity : int, default=30
            The perplexity parameter for t-SNE (ignored if method='umap').

        n_neighbors : int, default=15
            Number of neighbors to consider for UMAP (ignored if method='tsne').

        min_dist : float, default=0.1
            Minimum distance for UMAP points (ignored if method='tsne').

        figsize : tuple, default=(12, 10)
            Figure size for the plot.

        Returns
        -------
        reducer : The fitted dimensionality reduction model.
        """
        import matplotlib.pyplot as plt
        import numpy as np

        X = X.values if isinstance(X, pd.DataFrame) else X

        # Apply dimensionality reduction to all data
        if method.lower() == "tsne":
            X_reduced, reducer = self._apply_tsne(X, perplexity=perplexity)
        elif method.lower() == "umap":
            X_reduced, reducer = self._apply_umap(
                X, n_neighbors=n_neighbors, min_dist=min_dist
            )
        else:
            raise ValueError("Method must be either 'umap' or 'tsne'")

        # Get cluster labels for all points
        cluster_labels = self.dbscan.labels_

        # Plot
        plt.figure(figsize=figsize)

        # Plot noise points first
        noise_mask = cluster_labels == -1
        plt.scatter(
            X_reduced[noise_mask, 0],
            X_reduced[noise_mask, 1],
            c="lightgray",
            s=40,
            alpha=0.5,
            label="Noise",
        )

        # Plot points by cluster with different colors
        unique_clusters = [c for c in self.valid_clusters if c != -1]
        cmap = plt.cm.tab10
        for i, cluster in enumerate(unique_clusters):
            mask = cluster_labels == cluster
            plt.scatter(
                X_reduced[mask, 0],
                X_reduced[mask, 1],
                c=[cmap(i)],
                s=50,
                label=f"Cluster {cluster}",
            )

        plt.title(f"Cluster Embedding with {method.upper()}", fontsize=16)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.show()

        return reducer

    def compare_vis_methods(self, X, cluster_id=None, figsize=(20, 10)):
        """
        Compare t-SNE and UMAP visualizations side by side for a specific cluster.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        cluster_id : int or None, default=None
            If provided, visualize only that specific cluster.
            If None, visualize all data colored by cluster.

        figsize : tuple, default=(20, 10)
            Figure size for the plot.

        Returns
        -------
        None
        """
        import matplotlib.pyplot as plt

        X = X.values if isinstance(X, pd.DataFrame) else X

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        if cluster_id is not None and cluster_id in self.valid_clusters:
            # Get cluster-specific points
            cluster_points = self.cluster_points[cluster_id]

            # Get predictions
            predictions = self.ocsvms[cluster_id].predict(cluster_points)

            # Apply t-SNE
            X_tsne, _ = self._apply_tsne(cluster_points)

            # Apply UMAP
            X_umap, _ = self._apply_umap(cluster_points)

            # Plot t-SNE
            axes[0].scatter(
                X_tsne[predictions == 1, 0],
                X_tsne[predictions == 1, 1],
                c="blue",
                s=50,
                edgecolors="k",
                label="Normal",
            )
            axes[0].scatter(
                X_tsne[predictions == -1, 0],
                X_tsne[predictions == -1, 1],
                c="red",
                s=50,
                edgecolors="k",
                label="Anomalies",
            )
            axes[0].set_title(f"Cluster {cluster_id} with t-SNE", fontsize=14)
            axes[0].legend()

            # Plot UMAP
            axes[1].scatter(
                X_umap[predictions == 1, 0],
                X_umap[predictions == 1, 1],
                c="blue",
                s=50,
                edgecolors="k",
                label="Normal",
            )
            axes[1].scatter(
                X_umap[predictions == -1, 0],
                X_umap[predictions == -1, 1],
                c="red",
                s=50,
                edgecolors="k",
                label="Anomalies",
            )
            axes[1].set_title(f"Cluster {cluster_id} with UMAP", fontsize=14)
            axes[1].legend()
        else:
            # Apply t-SNE to all data
            X_tsne, _ = self._apply_tsne(X)

            # Apply UMAP to all data
            X_umap, _ = self._apply_umap(X)

            # Get cluster labels
            cluster_labels = self.dbscan.labels_

            # Plot t-SNE
            cmap = plt.cm.tab10
            for i, cluster in enumerate(self.valid_clusters):
                if cluster == -1:
                    axes[0].scatter(
                        X_tsne[cluster_labels == -1, 0],
                        X_tsne[cluster_labels == -1, 1],
                        c="lightgray",
                        s=40,
                        alpha=0.5,
                        label="Noise",
                    )
                else:
                    axes[0].scatter(
                        X_tsne[cluster_labels == cluster, 0],
                        X_tsne[cluster_labels == cluster, 1],
                        c=[cmap(i)],
                        s=50,
                        label=f"Cluster {cluster}",
                    )
            axes[0].set_title("All Clusters with t-SNE", fontsize=14)
            axes[0].legend()

            # Plot UMAP
            for i, cluster in enumerate(self.valid_clusters):
                if cluster == -1:
                    axes[1].scatter(
                        X_umap[cluster_labels == -1, 0],
                        X_umap[cluster_labels == -1, 1],
                        c="lightgray",
                        s=40,
                        alpha=0.5,
                        label="Noise",
                    )
                else:
                    axes[1].scatter(
                        X_umap[cluster_labels == cluster, 0],
                        X_umap[cluster_labels == cluster, 1],
                        c=[cmap(i)],
                        s=50,
                        label=f"Cluster {cluster}",
                    )
            axes[1].set_title("All Clusters with UMAP", fontsize=14)
            axes[1].legend()

        plt.tight_layout()
        plt.show()

    # def visualize_test_data(
    #     self,
    #     X_train,
    #     X_test,
    #     y_pred=None,
    #     method="umap",
    #     cluster_id=None,
    #     figsize=(15, 12),
    #     n_neighbors=15,
    #     perplexity=30,
    #     min_dist=0.1,
    #     mesh_granularity=200,
    # ):
    #     """
    #     Visualize test data points relative to the trained clusters and decision boundaries.

    #     Parameters
    #     ----------
    #     X_train : array-like of shape (n_samples, n_features)
    #         The training data used to train the model.

    #     X_test : array-like of shape (n_test_samples, n_features)
    #         The test data to visualize.

    #     y_pred : array-like of shape (n_test_samples,), optional
    #         Predicted labels for test data. If None, predictions will be made.

    #     method : str, default='umap'
    #         Dimensionality reduction method to use. Options: 'umap', 'tsne'

    #     cluster_id : int or None, default=None
    #         If provided, visualize only the specified cluster.
    #         If None, visualize all valid clusters.

    #     figsize : tuple, default=(15, 12)
    #         Figure size for the plots.

    #     n_neighbors : int, default=15
    #         Number of neighbors for UMAP (ignored if method='tsne').

    #     perplexity : int, default=30
    #         Perplexity parameter for t-SNE (ignored if method='umap').

    #     min_dist : float, default=0.1
    #         Minimum distance for UMAP points (ignored if method='tsne').

    #     mesh_granularity : int, default=200
    #         Number of points in mesh grid for decision boundary visualization.

    #     Returns
    #     -------
    #     dict : Dictionary mapping cluster IDs to their fitted dimension reducers.
    #     """
    #     import matplotlib.pyplot as plt
    #     from matplotlib.colors import ListedColormap
    #     import numpy as np

    #     X_train = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
    #     X_test = X_test.values if isinstance(X_test, pd.DataFrame) else X_test

    #     # Make predictions if not provided
    #     if y_pred is None:
    #         y_pred = self.predict(X_test)

    #     # Find which test points belong to which cluster
    #     test_cluster_assignments = {}

    #     # For each test point, find the nearest centroid
    #     if self.tree is not None:
    #         _, ind = self.tree.query(X_test, k=1)
    #         nearest_clusters = [self.valid_clusters[i] for i in ind.flatten()]

    #         # Group test points by their assigned cluster
    #         for i, cluster in enumerate(nearest_clusters):
    #             if cluster not in test_cluster_assignments:
    #                 test_cluster_assignments[cluster] = []
    #             test_cluster_assignments[cluster].append(i)

    #     # Determine which clusters to visualize
    #     clusters_to_visualize = (
    #         [cluster_id] if cluster_id is not None else self.valid_clusters
    #     )
    #     clusters_to_visualize = [
    #         c for c in clusters_to_visualize if c != -1 and c in self.ocsvms
    #     ]

    #     reducers = {}

    #     for cluster in clusters_to_visualize:
    #         plt.figure(figsize=figsize)

    #         # Get training points for this cluster
    #         train_cluster_points = self.cluster_points[cluster]

    #         # Get test points assigned to this cluster
    #         if cluster in test_cluster_assignments:
    #             test_indices = test_cluster_assignments[cluster]
    #             test_cluster_points = X_test[test_indices]
    #             test_predictions = y_pred[test_indices]
    #         else:
    #             test_cluster_points = np.array([]).reshape(0, X_test.shape[1])
    #             test_predictions = np.array([])

    #         # Apply dimensionality reduction to combined data
    #         combined_data = (
    #             np.vstack([train_cluster_points, test_cluster_points])
    #             if len(test_cluster_points) > 0
    #             else train_cluster_points
    #         )

    #         if method.lower() == "tsne":
    #             combined_reduced, reducer = self._apply_tsne(
    #                 combined_data, perplexity=perplexity
    #             )
    #         elif method.lower() == "umap":
    #             combined_reduced, reducer = self._apply_umap(
    #                 combined_data, n_neighbors=n_neighbors, min_dist=min_dist
    #             )
    #         else:
    #             raise ValueError("Method must be either 'umap' or 'tsne'")

    #         reducers[cluster] = reducer

    #         # Split the reduced data back into train and test
    #         train_reduced = combined_reduced[: len(train_cluster_points)]
    #         test_reduced = (
    #             combined_reduced[len(train_cluster_points) :]
    #             if len(test_cluster_points) > 0
    #             else np.array([])
    #         )

    #         # Get original OCSVM predictions for training data
    #         train_predictions = self.ocsvms[cluster].predict(train_cluster_points)

    #         # Create a mesh grid in the reduced space
    #         x_min = (
    #             min(
    #                 train_reduced[:, 0].min(),
    #                 test_reduced[:, 0].min() if len(test_reduced) > 0 else float("inf"),
    #             )
    #             - 1
    #         )
    #         x_max = (
    #             max(
    #                 train_reduced[:, 0].max(),
    #                 (
    #                     test_reduced[:, 0].max()
    #                     if len(test_reduced) > 0
    #                     else float("-inf")
    #                 ),
    #             )
    #             + 1
    #         )
    #         y_min = (
    #             min(
    #                 train_reduced[:, 1].min(),
    #                 test_reduced[:, 1].min() if len(test_reduced) > 0 else float("inf"),
    #             )
    #             - 1
    #         )
    #         y_max = (
    #             max(
    #                 train_reduced[:, 1].max(),
    #                 (
    #                     test_reduced[:, 1].max()
    #                     if len(test_reduced) > 0
    #                     else float("-inf")
    #                 ),
    #             )
    #             + 1
    #         )

    #         xx, yy = np.meshgrid(
    #             np.linspace(x_min, x_max, mesh_granularity),
    #             np.linspace(y_min, y_max, mesh_granularity),
    #         )

    #         # Create a separate OCSVM for visualization in the reduced space
    #         reduced_svm = OneClassSVM(
    #             kernel=self.ocsvms[cluster].kernel,
    #             nu=self.ocsvms[cluster].nu,
    #             gamma=self.ocsvms[cluster].gamma,
    #         )
    #         reduced_svm.fit(train_reduced[train_predictions == 1])

    #         # Get decision function values for the grid
    #         grid_points = np.c_[xx.ravel(), yy.ravel()]
    #         Z = reduced_svm.decision_function(grid_points).reshape(xx.shape)

    #         # Define custom colormap for visualization
    #         normal_color = "#8FBB99"  # Light green
    #         anomaly_color = "#F2BFCB"  # Light red
    #         cmap = ListedColormap([anomaly_color, normal_color])

    #         # Plot decision regions
    #         plt.contourf(xx, yy, Z > 0, alpha=0.3, cmap=cmap)

    #         # Plot contour lines for decision values
    #         contour = plt.contour(
    #             xx,
    #             yy,
    #             Z,
    #             levels=[-2, -1, 0, 1, 2],
    #             colors=["purple", "red", "black", "blue", "green"],
    #             linewidths=1.5,
    #             alpha=0.8,
    #         )
    #         plt.clabel(contour, inline=True, fontsize=10)

    #         # Plot the decision boundary
    #         plt.contour(xx, yy, Z, levels=[0], linewidths=2.5, colors="darkgreen")

    #         # Plot training points (smaller, more transparent)
    #         plt.scatter(
    #             train_reduced[train_predictions == 1, 0],
    #             train_reduced[train_predictions == 1, 1],
    #             c="blue",
    #             s=30,
    #             alpha=0.5,
    #             edgecolors="k",
    #             label="Train Normal",
    #         )
    #         plt.scatter(
    #             train_reduced[train_predictions == -1, 0],
    #             train_reduced[train_predictions == -1, 1],
    #             c="red",
    #             s=30,
    #             alpha=0.5,
    #             edgecolors="k",
    #             label="Train Anomaly",
    #         )

    #         # Plot test points (larger, more visible)
    #         if len(test_reduced) > 0:
    #             # Get decision scores for test points
    #             test_scores = reduced_svm.decision_function(test_reduced)

    #             # Plot test points with color based on prediction
    #             plt.scatter(
    #                 test_reduced[test_predictions == 1, 0],
    #                 test_reduced[test_predictions == 1, 1],
    #                 c="green",
    #                 s=100,
    #                 marker="^",
    #                 edgecolors="k",
    #                 linewidth=1.5,
    #                 label="Test Normal",
    #             )
    #             plt.scatter(
    #                 test_reduced[test_predictions == -1, 0],
    #                 test_reduced[test_predictions == -1, 1],
    #                 c="darkred",
    #                 s=100,
    #                 marker="^",
    #                 edgecolors="k",
    #                 linewidth=1.5,
    #                 label="Test Anomaly",
    #             )

    #             # Draw arrows from test points to cluster centroid
    #             centroid_reduced = np.mean(
    #                 train_reduced[train_predictions == 1], axis=0
    #             )
    #             for i, point in enumerate(test_reduced):
    #                 # Only draw arrows for anomalies or for points far from the decision boundary
    #                 if test_predictions[i] == -1 or abs(test_scores[i]) > 0.5:
    #                     plt.arrow(
    #                         point[0],
    #                         point[1],
    #                         (centroid_reduced[0] - point[0]) * 0.3,
    #                         (centroid_reduced[1] - point[1]) * 0.3,
    #                         head_width=0.1,
    #                         head_length=0.2,
    #                         fc="gray",
    #                         ec="gray",
    #                         alpha=0.4,
    #                     )

    #         plt.title(
    #             f"Cluster {cluster} - Test Data vs. Decision Boundary ({method.upper()})",
    #             fontsize=16,
    #         )
    #         plt.legend(fontsize=12, loc="best")
    #         plt.tight_layout()
    #         plt.show()

    #         # Show distance plot for test points
    #         if len(test_reduced) > 0:
    #             self._plot_distance_metrics(
    #                 cluster, test_indices, X_test, test_predictions, test_scores
    #             )

    #     return reducers

    def visualize_test_data(
        self,
        X_train,
        X_test,
        y_pred=None,
        method="umap",
        cluster_id=None,
        figsize=(15, 12),
        n_neighbors=15,
        perplexity=30,
        min_dist=0.1,
        mesh_granularity=200,
    ):
        """
        Visualize test data points relative to the trained clusters and decision boundaries.

        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            The training data used to train the model.

        X_test : array-like of shape (n_test_samples, n_features)
            The test data to visualize.

        y_pred : array-like of shape (n_test_samples,), optional
            Predicted labels for test data. If None, predictions will be made.

        method : str, default='umap'
            Dimensionality reduction method to use. Options: 'umap', 'tsne'

        cluster_id : int or None, default=None
            If provided, visualize only the specified cluster.
            If None, visualize all valid clusters.

        figsize : tuple, default=(15, 12)
            Figure size for the plots.

        n_neighbors : int, default=15
            Number of neighbors for UMAP (ignored if method='tsne').

        perplexity : int, default=30
            Perplexity parameter for t-SNE (ignored if method='umap').

        min_dist : float, default=0.1
            Minimum distance for UMAP points (ignored if method='tsne').

        mesh_granularity : int, default=200
            Number of points in mesh grid for decision boundary visualization.

        Returns
        -------
        dict : Dictionary mapping cluster IDs to their fitted dimension reducers.
        """
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
        import numpy as np

        X_train = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
        X_test = X_test.values if isinstance(X_test, pd.DataFrame) else X_test

        # Make predictions if not provided
        if y_pred is None:
            y_pred = self.predict(X_test)

        # Find which test points belong to which cluster
        test_cluster_assignments = {}

        # For each test point, find the nearest centroid
        if self.tree is not None:
            _, ind = self.tree.query(X_test, k=1)
            nearest_clusters = [self.valid_clusters[i] for i in ind.flatten()]

            # Group test points by their assigned cluster
            for i, cluster in enumerate(nearest_clusters):
                if cluster not in test_cluster_assignments:
                    test_cluster_assignments[cluster] = []
                test_cluster_assignments[cluster].append(i)

        # Determine which clusters to visualize
        clusters_to_visualize = (
            [cluster_id] if cluster_id is not None else self.valid_clusters
        )
        clusters_to_visualize = [
            c for c in clusters_to_visualize if c != -1 and c in self.ocsvms
        ]

        reducers = {}

        for cluster in clusters_to_visualize:
            plt.figure(figsize=figsize)

            # Get training points for this cluster
            train_cluster_points = self.cluster_points[cluster]

            # Get test points assigned to this cluster
            if cluster in test_cluster_assignments:
                test_indices = test_cluster_assignments[cluster]
                test_cluster_points = X_test[test_indices]
                test_predictions = y_pred[test_indices]
            else:
                test_cluster_points = np.array([]).reshape(0, X_test.shape[1])
                test_predictions = np.array([])

            # Apply dimensionality reduction to combined data
            combined_data = (
                np.vstack([train_cluster_points, test_cluster_points])
                if len(test_cluster_points) > 0
                else train_cluster_points
            )

            if method.lower() == "tsne":
                combined_reduced, reducer = self._apply_tsne(
                    combined_data, perplexity=perplexity
                )
            elif method.lower() == "umap":
                combined_reduced, reducer = self._apply_umap(
                    combined_data, n_neighbors=n_neighbors, min_dist=min_dist
                )
            else:
                raise ValueError("Method must be either 'umap' or 'tsne'")

            reducers[cluster] = reducer

            # Split the reduced data back into train and test
            train_reduced = combined_reduced[: len(train_cluster_points)]
            test_reduced = (
                combined_reduced[len(train_cluster_points) :]
                if len(test_cluster_points) > 0
                else np.array([])
            )

            # Get original OCSVM predictions for training data
            train_predictions = self.ocsvms[cluster].predict(train_cluster_points)

            # Create a mesh grid in the reduced space
            x_min = (
                min(
                    train_reduced[:, 0].min(),
                    test_reduced[:, 0].min() if len(test_reduced) > 0 else float("inf"),
                )
                - 1
            )
            x_max = (
                max(
                    train_reduced[:, 0].max(),
                    (
                        test_reduced[:, 0].max()
                        if len(test_reduced) > 0
                        else float("-inf")
                    ),
                )
                + 1
            )
            y_min = (
                min(
                    train_reduced[:, 1].min(),
                    test_reduced[:, 1].min() if len(test_reduced) > 0 else float("inf"),
                )
                - 1
            )
            y_max = (
                max(
                    train_reduced[:, 1].max(),
                    (
                        test_reduced[:, 1].max()
                        if len(test_reduced) > 0
                        else float("-inf")
                    ),
                )
                + 1
            )

            xx, yy = np.meshgrid(
                np.linspace(x_min, x_max, mesh_granularity),
                np.linspace(y_min, y_max, mesh_granularity),
            )

            # Create a separate OCSVM for visualization in the reduced space
            reduced_svm = OneClassSVM(
                kernel=self.ocsvms[cluster].kernel,
                nu=self.ocsvms[cluster].nu,
                gamma=self.ocsvms[cluster].gamma,
            )
            reduced_svm.fit(train_reduced[train_predictions == 1])

            # Get decision function values for the grid
            grid_points = np.c_[xx.ravel(), yy.ravel()]
            Z = reduced_svm.decision_function(grid_points).reshape(xx.shape)

            # Define custom colormap for visualization
            normal_color = "#8FBB99"  # Light green
            anomaly_color = "#F2BFCB"  # Light red
            cmap = ListedColormap([anomaly_color, normal_color])

            # Plot decision regions
            plt.contourf(xx, yy, Z > 0, alpha=0.3, cmap=cmap)

            # Plot contour lines for decision values
            contour = plt.contour(
                xx,
                yy,
                Z,
                levels=[-2, -1, 0, 1, 2],
                colors=["purple", "red", "black", "blue", "green"],
                linewidths=1.5,
                alpha=0.8,
            )
            plt.clabel(contour, inline=True, fontsize=10)

            # Plot the decision boundary
            plt.contour(xx, yy, Z, levels=[0], linewidths=2.5, colors="darkgreen")

            # Plot training points (smaller, more transparent)
            plt.scatter(
                train_reduced[train_predictions == 1, 0],
                train_reduced[train_predictions == 1, 1],
                c="blue",
                s=30,
                alpha=0.5,
                edgecolors="k",
                label="Train Normal",
            )
            plt.scatter(
                train_reduced[train_predictions == -1, 0],
                train_reduced[train_predictions == -1, 1],
                c="red",
                s=30,
                alpha=0.5,
                edgecolors="k",
                label="Train Anomaly",
            )

            # Plot test points (larger, more visible)
            if len(test_reduced) > 0:
                # Plot test points with color based on prediction
                plt.scatter(
                    test_reduced[test_predictions == 1, 0],
                    test_reduced[test_predictions == 1, 1],
                    c="green",
                    s=100,
                    marker="^",
                    edgecolors="k",
                    linewidth=1.5,
                    label="Test Normal",
                )
                plt.scatter(
                    test_reduced[test_predictions == -1, 0],
                    test_reduced[test_predictions == -1, 1],
                    c="darkred",
                    s=100,
                    marker="^",
                    edgecolors="k",
                    linewidth=1.5,
                    label="Test Anomaly",
                )

                # No arrows pointing to cluster centroids

            plt.title(
                f"Cluster {cluster} - Test Data vs. Decision Boundary ({method.upper()})",
                fontsize=16,
            )
            plt.legend(fontsize=12, loc="best")
            plt.tight_layout()
            plt.show()

            # No distance metrics plot

        return reducers

    def _plot_distance_metrics(
        self, cluster, test_indices, X_test, predictions, scores, figsize=(12, 8)
    ):
        """
        Plot distance metrics for test points relative to the cluster.

        Parameters
        ----------
        cluster : int
            Cluster ID

        test_indices : list
            Indices of test points assigned to this cluster

        X_test : array-like
            Test data points

        predictions : array-like
            Predictions for the test points

        scores : array-like
            Decision scores for test points

        figsize : tuple
            Figure size
        """
        import matplotlib.pyplot as plt
        import numpy as np

        if len(test_indices) == 0:
            return

        test_points = X_test[test_indices]

        # Get distance to cluster centroid for each test point
        centroid = self.dbscan_centroids[cluster]
        distances = np.linalg.norm(test_points - centroid, axis=1)

        # Sort by decision score
        sorted_indices = np.argsort(scores)
        sorted_scores = scores[sorted_indices]
        sorted_distances = distances[sorted_indices]
        sorted_predictions = predictions[sorted_indices]

        plt.figure(figsize=figsize)

        # Create color array based on predictions
        colors = ["green" if p == 1 else "red" for p in sorted_predictions]

        # Plot decision score vs distance to centroid
        plt.scatter(sorted_distances, sorted_scores, c=colors, s=100, edgecolors="k")

        # Add horizontal line at decision boundary (score = 0)
        plt.axhline(y=0, color="black", linestyle="--", alpha=0.7)

        # Add labels and styling
        plt.xlabel("Distance to Cluster Centroid", fontsize=14)
        plt.ylabel("Decision Score", fontsize=14)
        plt.title(
            f"Cluster {cluster} - Test Points Decision Scores vs. Distance", fontsize=16
        )

        # Add color legend
        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="green",
                markersize=10,
                label="Normal",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="red",
                markersize=10,
                label="Anomaly",
            ),
        ]
        plt.legend(handles=legend_elements, fontsize=12)

        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_global_test_assignment(
        self,
        X_train,
        X_test,
        y_pred=None,
        method="umap",
        figsize=(14, 12),
        n_neighbors=15,
        min_dist=0.1,
        perplexity=30,
    ):
        """
        Visualize how test points are assigned to different clusters globally.

        Parameters
        ----------
        X_train : array-like
            Training data

        X_test : array-like
            Test data

        y_pred : array-like, optional
            Predictions for test data

        method : str, default='umap'
            Dimensionality reduction method

        figsize : tuple, default=(14, 12)
            Figure size

        n_neighbors, min_dist, perplexity : UMAP/t-SNE parameters

        Returns
        -------
        reducer : The fitted dimensionality reduction model
        """
        import matplotlib.pyplot as plt
        import numpy as np

        X_train = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
        X_test = X_test.values if isinstance(X_test, pd.DataFrame) else X_test

        # Get predictions if not provided
        if y_pred is None:
            y_pred = self.predict(X_test)

        # Get cluster assignments for test data
        if self.tree is not None:
            _, ind = self.tree.query(X_test, k=1)
            test_cluster_assignments = [self.valid_clusters[i] for i in ind.flatten()]
        else:
            test_cluster_assignments = [-1] * len(
                X_test
            )  # Assign all to noise if no tree

        # Combine train and test data
        combined_data = np.vstack([X_train, X_test])

        # Apply dimensionality reduction
        if method.lower() == "tsne":
            combined_reduced, reducer = self._apply_tsne(
                combined_data, perplexity=perplexity
            )
        elif method.lower() == "umap":
            combined_reduced, reducer = self._apply_umap(
                combined_data, n_neighbors=n_neighbors, min_dist=min_dist
            )

        # Split back to train and test
        train_reduced = combined_reduced[: len(X_train)]
        test_reduced = combined_reduced[len(X_train) :]

        # Plot
        plt.figure(figsize=figsize)

        # Plot training data by cluster
        train_clusters = self.dbscan.labels_

        # Plot noise points first
        noise_mask = train_clusters == -1
        plt.scatter(
            train_reduced[noise_mask, 0],
            train_reduced[noise_mask, 1],
            c="lightgray",
            s=20,
            alpha=0.3,
            label="Train (Noise)",
        )

        # Plot other clusters
        cmap = plt.cm.tab10
        for i, cluster in enumerate(self.valid_clusters):
            if cluster == -1:
                continue
            mask = train_clusters == cluster
            plt.scatter(
                train_reduced[mask, 0],
                train_reduced[mask, 1],
                c=[cmap(i % 10)],
                s=30,
                alpha=0.5,
                label=f"Train (Cluster {cluster})",
            )

        # Plot test points by assigned cluster and prediction
        for i, cluster in enumerate(np.unique(test_cluster_assignments)):
            if cluster == -1:
                continue

            # Get points assigned to this cluster
            cluster_mask = np.array(test_cluster_assignments) == cluster

            # Split by prediction
            normal_mask = cluster_mask & (y_pred == 1)
            anomaly_mask = cluster_mask & (y_pred == -1)

            # Plot
            if np.any(normal_mask):
                plt.scatter(
                    test_reduced[normal_mask, 0],
                    test_reduced[normal_mask, 1],
                    marker="^",
                    s=120,
                    edgecolors="darkgreen",
                    facecolors="none",
                    linewidth=2,
                    alpha=0.9,
                    label=f"Test Normal (Cluster {cluster})" if i == 0 else "",
                )

            if np.any(anomaly_mask):
                plt.scatter(
                    test_reduced[anomaly_mask, 0],
                    test_reduced[anomaly_mask, 1],
                    marker="X",
                    s=120,
                    color="red",
                    edgecolors="darkred",
                    linewidth=2,
                    alpha=0.9,
                    label=f"Test Anomaly (Cluster {cluster})" if i == 0 else "",
                )

        # Show outliers (test points assigned to noise)
        noise_mask = np.array(test_cluster_assignments) == -1
        if np.any(noise_mask):
            plt.scatter(
                test_reduced[noise_mask, 0],
                test_reduced[noise_mask, 1],
                marker="*",
                s=150,
                color="purple",
                alpha=0.9,
                label="Test (Unassigned)",
            )

        plt.title(
            f"Global Test Data Cluster Assignment ({method.upper()})", fontsize=16
        )
        plt.legend(fontsize=10, loc="best")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

        return reducer

"""
Feature preprocessing for GNN-AdaptGraph
"""

import numpy as np
from sklearn.preprocessing import StandardScaler


class FeaturePreprocessor:
    """
    Standardizes partition features before GraphSAGE.
    """

    def __init__(self):
        self.scaler = StandardScaler()

    def fit_transform(self, features: np.ndarray):
        """
        Fit the scaler and transform features.

        Parameters
        ----------
        features : np.ndarray
            Shape = (num_partitions, num_features)

        Returns
        -------
        np.ndarray
            Standardized feature matrix.
        """
        return self.scaler.fit_transform(features)

    def transform(self, features: np.ndarray):
        """
        Transform using previously fitted scaler.
        """
        return self.scaler.transform(features)
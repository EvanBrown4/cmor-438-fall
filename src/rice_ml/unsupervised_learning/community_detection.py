import numpy as np
import pandas as pd
from typing import Optional, List, Union

from rice_ml.utilities._validation import (
    _validate_2d_array,
    _check_square_matrix,
)

ArrayLike = Union[list, tuple, np.ndarray, pd.Series, pd.DataFrame]


class CommunityDetector:
    """
    Greedy modularity-based community detection algorithm.

    Parameters
    ----------
    resolution : float, default=1.0
        Resolution parameter. Values > 1.0 produce smaller communities.
    max_iter : int, default=100
        Maximum number of passes over all nodes.
    min_modularity_gain : float, default=1e-5
        Minimum modularity improvement required to accept a move.
    random_state : int or None, default=None
        Random seed for node processing order.

    Attributes
    ----------
    labels_ : ndarray of shape (n_nodes,)
        Community assignment for each node.
    n_communities_ : int
        Number of detected communities.
    modularity_ : float
        Final modularity score.
    adjacency_ : ndarray of shape (n_nodes, n_nodes)
        Symmetrized adjacency matrix.
    degrees_ : ndarray of shape (n_nodes,)
        Degree of each node.
    m_ : float
        Total edge weight divided by 2.
    """

    def __init__(
        self,
        resolution: float = 1.0,
        max_iter: int = 100,
        min_modularity_gain: float = 1e-5,
        random_state: Optional[int] = None,
    ) -> None:
        self.resolution = resolution
        self.max_iter = max_iter
        self.min_modularity_gain = min_modularity_gain
        self.random_state = random_state

    def fit(self, adjacency: ArrayLike) -> "CommunityDetector":
        """
        Detect communities in the graph.

        Parameters
        ----------
        adjacency : array-like of shape (n_nodes, n_nodes)
            Adjacency matrix. Non-symmetric matrices are symmetrized.

        Returns
        -------
        self : CommunityDetector
            Fitted estimator.
        """
        A = _validate_2d_array(adjacency)
        _check_square_matrix(A, name="adjacency")

        if np.any(A < 0):
            raise ValueError("Adjacency matrix must be non-negative.")

        if not np.allclose(A, A.T):
            A = 0.5 * (A + A.T)

        n_nodes = A.shape[0]
        self.adjacency_ = A
        self.degrees_ = A.sum(axis=1)
        self.m_ = A.sum() / 2.0

        if self.m_ == 0:
            self.labels_ = np.arange(n_nodes, dtype=int)
            self.n_communities_ = n_nodes
            self.modularity_ = 0.0
            return self

        rng = np.random.default_rng(self.random_state)
        labels = np.arange(n_nodes, dtype=int)
        
        comm_weights = {}
        comm_degrees = {}
        
        for i in range(n_nodes):
            comm_weights[i] = A[i, i]
            comm_degrees[i] = self.degrees_[i]

        for iteration in range(self.max_iter):
            improved = False

            for i in rng.permutation(n_nodes):
                current_comm = labels[i]
                best_comm = current_comm
                best_gain = 0.0

                neighbor_indices = np.nonzero(A[i] > 0)[0]
                candidate_comms = set(labels[neighbor_indices])
                candidate_comms.add(current_comm)

                ki = self.degrees_[i]
                ki_in_current = A[i, labels == current_comm].sum()
                
                for c in candidate_comms:
                    if c == current_comm:
                        continue

                    ki_in_c = A[i, labels == c].sum()
                    
                    gain = self._modularity_gain(
                        ki, ki_in_c, comm_degrees.get(c, 0.0),
                        ki_in_current, comm_degrees.get(current_comm, 0.0)
                    )

                    if gain > best_gain:
                        best_gain = gain
                        best_comm = c

                if best_gain > self.min_modularity_gain:
                    old_comm = current_comm
                    
                    comm_degrees[old_comm] -= ki
                    edges_within_old = A[i, labels == old_comm].sum()
                    comm_weights[old_comm] -= 2 * edges_within_old - A[i, i]
                    
                    if best_comm not in comm_degrees:
                        comm_degrees[best_comm] = 0.0
                        comm_weights[best_comm] = 0.0
                    
                    comm_degrees[best_comm] += ki
                    edges_within_new = A[i, labels == best_comm].sum()
                    comm_weights[best_comm] += 2 * edges_within_new + A[i, i]
                    
                    labels[i] = best_comm
                    improved = True

            if not improved:
                break

        unique_comms = np.unique(labels)
        remap = {old: new for new, old in enumerate(unique_comms)}
        labels = np.array([remap[c] for c in labels], dtype=int)

        self.labels_ = labels
        self.n_communities_ = len(unique_comms)
        self.modularity_ = self._modularity(A, labels)

        return self

    def fit_predict(self, adjacency: ArrayLike) -> np.ndarray:
        """
        Fit and return community labels.

        Parameters
        ----------
        adjacency : array-like of shape (n_nodes, n_nodes)
            Adjacency matrix.

        Returns
        -------
        labels : ndarray of shape (n_nodes,)
            Community labels.
        """
        self.fit(adjacency)
        return self.labels_

    def get_communities(self) -> List[np.ndarray]:
        """
        Get node indices for each community.

        Returns
        -------
        communities : list of ndarray
            List where communities[k] contains node indices in community k.
        """
        if not hasattr(self, "labels_"):
            raise RuntimeError("The model has not been fit yet.")

        communities: List[np.ndarray] = []
        for k in range(self.n_communities_):
            communities.append(np.where(self.labels_ == k)[0])
        return communities

    def _modularity_gain(
        self, 
        ki: float, 
        ki_in_new: float, 
        sigma_tot_new: float,
        ki_in_old: float, 
        sigma_tot_old: float
    ) -> float:
        """Calculate modularity gain for moving a node."""
        m = self.m_
        gamma = self.resolution
        
        if m == 0:
            return 0.0
        
        gain_new = (ki_in_new / m) - gamma * (sigma_tot_new * ki) / (2.0 * m * m)
        loss_old = (ki_in_old / m) - gamma * (sigma_tot_old * ki) / (2.0 * m * m)
        
        return gain_new - loss_old

    def _modularity(self, A: np.ndarray, labels: np.ndarray) -> float:
        """Compute modularity of a partition."""
        m = self.m_
        if m == 0:
            return 0.0

        degrees = self.degrees_
        gamma = self.resolution
        Q = 0.0

        for c in np.unique(labels):
            idx = (labels == c)
            if idx.sum() == 0:
                continue

            w_in = A[np.ix_(idx, idx)].sum()
            k_tot = degrees[idx].sum()
            Q += (w_in / (2.0 * m)) - gamma * (k_tot / (2.0 * m)) ** 2

        return Q
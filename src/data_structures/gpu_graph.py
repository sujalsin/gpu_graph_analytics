import cupy as cp
import numpy as np
from typing import Tuple, Optional

class GPUGraph:
    """
    GPU-optimized graph data structure using CSR (Compressed Sparse Row) format.
    Designed for efficient parallel processing on CUDA-enabled GPUs.
    """
    
    def __init__(self, num_nodes: int):
        """
        Initialize an empty graph with specified number of nodes.
        
        Args:
            num_nodes: Number of nodes in the graph
        """
        self.num_nodes = num_nodes
        self.row_offsets = None  # CSR row offsets
        self.column_indices = None  # CSR column indices
        self.edge_weights = None  # Optional edge weights
        
    @classmethod
    def from_edge_list(cls, edges: np.ndarray, weights: Optional[np.ndarray] = None) -> 'GPUGraph':
        """
        Create a GPU graph from an edge list.
        
        Args:
            edges: Nx2 array of edges (source, destination)
            weights: Optional array of edge weights
            
        Returns:
            GPUGraph instance
        """
        num_nodes = edges.max() + 1
        graph = cls(num_nodes)
        
        # Sort edges by source node for CSR conversion
        sorted_idx = np.argsort(edges[:, 0])
        sorted_edges = edges[sorted_idx]
        
        if weights is not None:
            sorted_weights = weights[sorted_idx]
        
        # Calculate row offsets
        unique_sources, counts = np.unique(sorted_edges[:, 0], return_counts=True)
        row_offsets = np.zeros(num_nodes + 1, dtype=np.int32)
        np.cumsum(counts, out=row_offsets[unique_sources + 1])
        
        # Transfer to GPU
        graph.row_offsets = cp.asarray(row_offsets)
        graph.column_indices = cp.asarray(sorted_edges[:, 1])
        
        if weights is not None:
            graph.edge_weights = cp.asarray(sorted_weights)
            
        return graph
    
    def get_neighbors(self, node: int) -> cp.ndarray:
        """
        Get neighbors of a node.
        
        Args:
            node: Node ID
            
        Returns:
            Array of neighbor node IDs
        """
        start = int(self.row_offsets[node])
        end = int(self.row_offsets[node + 1])
        return self.column_indices[start:end]
    
    def get_degree(self, node: int) -> int:
        """
        Get the out-degree of a node.
        
        Args:
            node: Node ID
            
        Returns:
            Number of outgoing edges
        """
        return int(self.row_offsets[node + 1] - self.row_offsets[node])
    
    def to_device(self) -> None:
        """Transfer graph data to GPU if not already there."""
        if not isinstance(self.row_offsets, cp.ndarray):
            self.row_offsets = cp.asarray(self.row_offsets)
        if not isinstance(self.column_indices, cp.ndarray):
            self.column_indices = cp.asarray(self.column_indices)
        if self.edge_weights is not None and not isinstance(self.edge_weights, cp.ndarray):
            self.edge_weights = cp.asarray(self.edge_weights)
    
    def to_host(self) -> None:
        """Transfer graph data back to CPU."""
        self.row_offsets = cp.asnumpy(self.row_offsets)
        self.column_indices = cp.asnumpy(self.column_indices)
        if self.edge_weights is not None:
            self.edge_weights = cp.asnumpy(self.edge_weights)
    
    @property
    def num_edges(self) -> int:
        """Get the total number of edges in the graph."""
        return len(self.column_indices)
    
    def get_edge_weights(self, node: int) -> Optional[cp.ndarray]:
        """
        Get weights of edges from a node.
        
        Args:
            node: Node ID
            
        Returns:
            Array of edge weights if they exist, None otherwise
        """
        if self.edge_weights is None:
            return None
        start = int(self.row_offsets[node])
        end = int(self.row_offsets[node + 1])
        return self.edge_weights[start:end]

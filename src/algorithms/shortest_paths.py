import cupy as cp
import numpy as np
from typing import Optional, Dict, List
from ..data_structures.gpu_graph import GPUGraph

class ShortestPaths:
    """
    GPU-accelerated implementation of shortest path algorithms.
    Supports both single-source and all-pairs shortest paths.
    """
    
    def __init__(self):
        """Initialize CUDA kernels for shortest path computations."""
        # CUDA kernel for Bellman-Ford algorithm
        self.bellman_ford_kernel = cp.RawKernel(r'''
        extern "C" __global__
        void bellman_ford_kernel(const int* row_offsets,
                               const int* column_indices,
                               const float* weights,
                               float* distances,
                               bool* changed,
                               const int num_nodes) {
            int tid = blockDim.x * blockIdx.x + threadIdx.x;
            if (tid >= num_nodes) return;
            
            float old_dist = distances[tid];
            float min_dist = old_dist;
            
            // Check all incoming edges
            int start = row_offsets[tid];
            int end = row_offsets[tid + 1];
            
            for (int i = start; i < end; i++) {
                int src = column_indices[i];
                float weight = weights != nullptr ? weights[i] : 1.0f;
                float new_dist = distances[src] + weight;
                
                if (new_dist < min_dist) {
                    min_dist = new_dist;
                    *changed = true;
                }
            }
            
            distances[tid] = min_dist;
        }
        ''', 'bellman_ford_kernel')
        
        # CUDA kernel for Dijkstra's algorithm
        self.dijkstra_kernel = cp.RawKernel(r'''
        extern "C" __global__
        void dijkstra_kernel(const int* row_offsets,
                           const int* column_indices,
                           const float* weights,
                           float* distances,
                           bool* visited,
                           int* predecessors,
                           const int current_node,
                           const int num_nodes) {
            int tid = blockDim.x * blockIdx.x + threadIdx.x;
            if (tid >= num_nodes || visited[tid]) return;
            
            int start = row_offsets[current_node];
            int end = row_offsets[current_node + 1];
            
            for (int i = start; i < end; i++) {
                if (column_indices[i] != tid) continue;
                
                float weight = weights != nullptr ? weights[i] : 1.0f;
                float new_dist = distances[current_node] + weight;
                
                if (new_dist < distances[tid]) {
                    distances[tid] = new_dist;
                    predecessors[tid] = current_node;
                }
            }
        }
        ''', 'dijkstra_kernel')
    
    def bellman_ford(self,
                    graph: GPUGraph,
                    source: int,
                    max_iterations: Optional[int] = None) -> cp.ndarray:
        """
        Compute single-source shortest paths using Bellman-Ford algorithm.
        
        Args:
            graph: Input graph
            source: Source node
            max_iterations: Maximum number of iterations
            
        Returns:
            Array of shortest distances from source
        """
        if max_iterations is None:
            max_iterations = graph.num_nodes - 1
            
        num_nodes = graph.num_nodes
        
        # Initialize distances
        distances = cp.full(num_nodes, cp.inf, dtype=cp.float32)
        distances[source] = 0
        
        # Configure CUDA kernel
        threads_per_block = 256
        blocks = (num_nodes + threads_per_block - 1) // threads_per_block
        
        # Main loop
        for _ in range(max_iterations):
            changed = cp.array([False])
            
            self.bellman_ford_kernel(
                (blocks,), (threads_per_block,),
                (graph.row_offsets, graph.column_indices,
                 graph.edge_weights, distances, changed,
                 num_nodes)
            )
            
            if not changed.get():
                break
                
        return distances
    
    def dijkstra(self,
                 graph: GPUGraph,
                 source: int) -> Tuple[cp.ndarray, cp.ndarray]:
        """
        Compute single-source shortest paths using Dijkstra's algorithm.
        
        Args:
            graph: Input graph
            source: Source node
            
        Returns:
            Tuple of (distances, predecessors)
        """
        num_nodes = graph.num_nodes
        
        # Initialize arrays
        distances = cp.full(num_nodes, cp.inf, dtype=cp.float32)
        distances[source] = 0
        
        visited = cp.zeros(num_nodes, dtype=cp.bool_)
        predecessors = cp.full(num_nodes, -1, dtype=cp.int32)
        
        # Configure CUDA kernel
        threads_per_block = 256
        blocks = (num_nodes + threads_per_block - 1) // threads_per_block
        
        # Main loop
        for _ in range(num_nodes):
            # Find minimum unvisited node
            unvisited_distances = cp.where(visited, cp.inf, distances)
            current = int(cp.argmin(unvisited_distances))
            
            if distances[current] == cp.inf:
                break
                
            visited[current] = True
            
            # Update distances through current node
            self.dijkstra_kernel(
                (blocks,), (threads_per_block,),
                (graph.row_offsets, graph.column_indices,
                 graph.edge_weights, distances, visited,
                 predecessors, current, num_nodes)
            )
        
        return distances, predecessors
    
    def get_path(self,
                 predecessors: cp.ndarray,
                 target: int) -> List[int]:
        """
        Reconstruct path from source to target using predecessor array.
        
        Args:
            predecessors: Array of predecessors
            target: Target node
            
        Returns:
            List of nodes in the path
        """
        path = []
        current = target
        
        while current != -1:
            path.append(int(current))
            current = int(predecessors[current])
            
        return list(reversed(path))

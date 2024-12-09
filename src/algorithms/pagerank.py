import cupy as cp
from typing import Optional
from ..data_structures.gpu_graph import GPUGraph

# CUDA kernel for PageRank computation
pagerank_kernel = cp.RawKernel(r'''
extern "C" __global__
void pagerank_kernel(const int* row_offsets,
                    const int* column_indices,
                    const float* weights,
                    const float* curr_ranks,
                    float* next_ranks,
                    const float damping_factor,
                    const int num_nodes) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    
    if (tid < num_nodes) {
        float sum = 0.0f;
        int start = row_offsets[tid];
        int end = row_offsets[tid + 1];
        
        // Sum contributions from incoming edges
        for (int i = start; i < end; i++) {
            int src = column_indices[i];
            float contribution = curr_ranks[src];
            if (weights != nullptr) {
                contribution *= weights[i];
            }
            sum += contribution;
        }
        
        // Apply damping factor
        next_ranks[tid] = (1.0f - damping_factor) / num_nodes + 
                         damping_factor * sum;
    }
}
''', 'pagerank_kernel')

class PageRank:
    """
    GPU-accelerated PageRank implementation using CUDA.
    """
    
    def __init__(self, 
                 num_iterations: int = 100,
                 damping_factor: float = 0.85,
                 tolerance: float = 1e-6):
        """
        Initialize PageRank algorithm.
        
        Args:
            num_iterations: Maximum number of iterations
            damping_factor: Damping factor (typically 0.85)
            tolerance: Convergence tolerance
        """
        self.num_iterations = num_iterations
        self.damping_factor = damping_factor
        self.tolerance = tolerance
        
    def compute(self, graph: GPUGraph) -> cp.ndarray:
        """
        Compute PageRank scores for all nodes.
        
        Args:
            graph: GPUGraph instance
            
        Returns:
            Array of PageRank scores
        """
        # Ensure graph data is on GPU
        graph.to_device()
        
        num_nodes = graph.num_nodes
        
        # Initialize PageRank scores
        curr_ranks = cp.full(num_nodes, 1.0 / num_nodes, dtype=cp.float32)
        next_ranks = cp.zeros_like(curr_ranks)
        
        # Normalize edge weights if they exist
        if graph.edge_weights is not None:
            weights = cp.asarray(graph.edge_weights, dtype=cp.float32)
            # Normalize by out-degree
            for i in range(num_nodes):
                start = int(graph.row_offsets[i])
                end = int(graph.row_offsets[i + 1])
                if end > start:
                    weights[start:end] /= (end - start)
        else:
            weights = None
        
        # Configure CUDA kernel
        threads_per_block = 256
        blocks = (num_nodes + threads_per_block - 1) // threads_per_block
        
        # Main iteration loop
        for _ in range(self.num_iterations):
            # Launch kernel
            pagerank_kernel(
                (blocks,), (threads_per_block,),
                (graph.row_offsets, graph.column_indices, weights,
                 curr_ranks, next_ranks, self.damping_factor, num_nodes)
            )
            
            # Check convergence
            diff = cp.abs(next_ranks - curr_ranks).max()
            if diff < self.tolerance:
                break
                
            # Swap buffers
            curr_ranks, next_ranks = next_ranks, curr_ranks
        
        return curr_ranks
    
    def compute_cpu(self, graph: GPUGraph) -> cp.ndarray:
        """
        CPU implementation for comparison (slower but useful for validation).
        
        Args:
            graph: GPUGraph instance
            
        Returns:
            Array of PageRank scores
        """
        graph.to_host()
        num_nodes = graph.num_nodes
        
        # Initialize scores
        curr_ranks = np.full(num_nodes, 1.0 / num_nodes)
        next_ranks = np.zeros_like(curr_ranks)
        
        for _ in range(self.num_iterations):
            # Compute new ranks
            for i in range(num_nodes):
                sum_score = 0.0
                for j in range(graph.row_offsets[i], graph.row_offsets[i + 1]):
                    src = graph.column_indices[j]
                    if graph.edge_weights is not None:
                        sum_score += curr_ranks[src] * graph.edge_weights[j]
                    else:
                        sum_score += curr_ranks[src] / graph.get_degree(src)
                
                next_ranks[i] = (1.0 - self.damping_factor) / num_nodes + \
                               self.damping_factor * sum_score
            
            # Check convergence
            if np.abs(next_ranks - curr_ranks).max() < self.tolerance:
                break
                
            curr_ranks, next_ranks = next_ranks, curr_ranks
        
        return cp.asarray(curr_ranks)

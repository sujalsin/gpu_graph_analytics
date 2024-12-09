import cupy as cp
import numpy as np
from typing import Tuple, List, Optional
from ..data_structures.gpu_graph import GPUGraph

class Node2Vec:
    """
    GPU-accelerated implementation of Node2Vec for generating node embeddings.
    Uses random walks and skip-gram with negative sampling.
    """
    
    def __init__(self,
                 dimensions: int = 128,
                 walk_length: int = 80,
                 num_walks: int = 10,
                 p: float = 1.0,
                 q: float = 1.0,
                 window_size: int = 10,
                 num_negative: int = 5,
                 learning_rate: float = 0.025):
        """
        Initialize Node2Vec parameters.
        
        Args:
            dimensions: Embedding dimensions
            walk_length: Length of each random walk
            num_walks: Number of walks per node
            p: Return parameter
            q: In-out parameter
            window_size: Context window size
            num_negative: Number of negative samples
            learning_rate: Learning rate for optimization
        """
        self.dimensions = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q
        self.window_size = window_size
        self.num_negative = num_negative
        self.learning_rate = learning_rate
        
        # CUDA kernel for random walk generation
        self.random_walk_kernel = cp.RawKernel(r'''
        extern "C" __global__
        void random_walk_kernel(const int* row_offsets,
                              const int* column_indices,
                              const float* alias_table,
                              const int* alias_indices,
                              int* walks,
                              const int num_nodes,
                              const int walk_length,
                              const unsigned long long seed) {
            int tid = blockDim.x * blockIdx.x + threadIdx.x;
            if (tid >= num_nodes) return;
            
            // Initialize random state
            curandState state;
            curand_init(seed + tid, 0, 0, &state);
            
            // Start walk from each node
            int current = tid;
            walks[tid * walk_length] = current;
            
            for (int i = 1; i < walk_length; i++) {
                int start = row_offsets[current];
                int degree = row_offsets[current + 1] - start;
                
                if (degree == 0) {
                    walks[tid * walk_length + i] = current;
                    continue;
                }
                
                // Sample next node using alias method
                float r = curand_uniform(&state);
                int idx = (int)(r * degree);
                if (r > alias_table[start + idx])
                    idx = alias_indices[start + idx];
                
                current = column_indices[start + idx];
                walks[tid * walk_length + i] = current;
            }
        }
        ''', 'random_walk_kernel')
        
    def _create_alias_table(self, graph: GPUGraph) -> Tuple[cp.ndarray, cp.ndarray]:
        """Create alias table for efficient node sampling."""
        num_nodes = graph.num_nodes
        probs = []
        alias = []
        
        # Calculate transition probabilities
        for node in range(num_nodes):
            neighbors = graph.get_neighbors(node)
            if len(neighbors) == 0:
                continue
                
            unnormalized_probs = cp.ones(len(neighbors))
            if graph.edge_weights is not None:
                unnormalized_probs *= graph.get_edge_weights(node)
                
            normalized_probs = unnormalized_probs / unnormalized_probs.sum()
            
            # Create alias table for this node
            prob_arr = cp.zeros(len(neighbors))
            alias_arr = cp.zeros(len(neighbors), dtype=cp.int32)
            
            # Alias method setup
            small = []
            large = []
            
            for idx, prob in enumerate(normalized_probs):
                if prob < 1.0:
                    small.append(idx)
                else:
                    large.append(idx)
                    
            while small and large:
                s = small.pop()
                l = large.pop()
                
                prob_arr[s] = normalized_probs[s] * len(neighbors)
                alias_arr[s] = l
                
                normalized_probs[l] = (normalized_probs[l] + normalized_probs[s] - 1.0)
                if normalized_probs[l] < 1.0:
                    small.append(l)
                else:
                    large.append(l)
                    
            probs.extend(prob_arr.get())
            alias.extend(alias_arr.get())
            
        return cp.array(probs), cp.array(alias, dtype=cp.int32)
    
    def _generate_walks(self, graph: GPUGraph) -> cp.ndarray:
        """Generate random walks for each node."""
        num_nodes = graph.num_nodes
        
        # Create alias table for efficient sampling
        alias_probs, alias_indices = self._create_alias_table(graph)
        
        # Allocate memory for walks
        walks = cp.zeros((num_nodes * self.num_walks, self.walk_length), dtype=cp.int32)
        
        # Configure CUDA kernel
        threads_per_block = 256
        blocks = (num_nodes * self.num_walks + threads_per_block - 1) // threads_per_block
        
        # Generate walks
        for i in range(self.num_walks):
            seed = np.random.randint(0, 2**32)
            offset = i * num_nodes
            
            self.random_walk_kernel(
                (blocks,), (threads_per_block,),
                (graph.row_offsets, graph.column_indices,
                 alias_probs, alias_indices,
                 walks[offset:offset + num_nodes],
                 num_nodes, self.walk_length, seed)
            )
            
        return walks
    
    def train(self, graph: GPUGraph) -> cp.ndarray:
        """
        Train Node2Vec model and generate embeddings.
        
        Args:
            graph: Input graph
            
        Returns:
            Node embeddings matrix
        """
        # Generate random walks
        walks = self._generate_walks(graph)
        
        # Initialize embeddings
        embeddings = cp.random.uniform(
            low=-0.5/self.dimensions,
            high=0.5/self.dimensions,
            size=(graph.num_nodes, self.dimensions)
        ).astype(cp.float32)
        
        # Train skip-gram model using negative sampling
        for epoch in range(self.num_walks):
            for walk in walks:
                for i in range(len(walk)):
                    for j in range(max(0, i - self.window_size),
                                 min(len(walk), i + self.window_size + 1)):
                        if i == j:
                            continue
                            
                        # Positive sample
                        target = walk[i]
                        context = walk[j]
                        
                        # Update embeddings
                        target_emb = embeddings[target]
                        context_emb = embeddings[context]
                        
                        score = cp.dot(target_emb, context_emb)
                        sigmoid = 1 / (1 + cp.exp(-score))
                        
                        grad = self.learning_rate * (1 - sigmoid)
                        
                        embeddings[target] += grad * context_emb
                        embeddings[context] += grad * target_emb
                        
                        # Negative sampling
                        for _ in range(self.num_negative):
                            neg = cp.random.randint(0, graph.num_nodes)
                            if neg == target or neg == context:
                                continue
                                
                            neg_emb = embeddings[neg]
                            score = cp.dot(target_emb, neg_emb)
                            sigmoid = 1 / (1 + cp.exp(-score))
                            
                            grad = -self.learning_rate * sigmoid
                            
                            embeddings[target] += grad * neg_emb
                            embeddings[neg] += grad * target_emb
        
        return embeddings

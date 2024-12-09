import numpy as np
import time
from src.data_structures.gpu_graph import GPUGraph
from src.algorithms.pagerank import PageRank

def create_random_graph(num_nodes: int, edge_probability: float = 0.1) -> np.ndarray:
    """Create a random graph for testing."""
    # Generate random edges
    edges = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j and np.random.random() < edge_probability:
                edges.append([i, j])
    return np.array(edges)

def main():
    # Create a random graph
    num_nodes = 1000
    edges = create_random_graph(num_nodes)
    
    # Create GPU graph
    graph = GPUGraph.from_edge_list(edges)
    
    # Initialize PageRank
    pagerank = PageRank(num_iterations=100, damping_factor=0.85, tolerance=1e-6)
    
    # Compute PageRank scores using GPU
    start_time = time.time()
    gpu_scores = pagerank.compute(graph)
    gpu_time = time.time() - start_time
    
    # Compute PageRank scores using CPU for comparison
    start_time = time.time()
    cpu_scores = pagerank.compute_cpu(graph)
    cpu_time = time.time() - start_time
    
    # Print results
    print(f"GPU computation time: {gpu_time:.4f} seconds")
    print(f"CPU computation time: {cpu_time:.4f} seconds")
    print(f"Speedup: {cpu_time/gpu_time:.2f}x")
    
    # Print top 10 nodes by PageRank score
    top_nodes = np.argsort(gpu_scores.get())[-10:][::-1]
    print("\nTop 10 nodes by PageRank score:")
    for i, node in enumerate(top_nodes, 1):
        print(f"{i}. Node {node}: {gpu_scores[node]:.6f}")

if __name__ == "__main__":
    main()

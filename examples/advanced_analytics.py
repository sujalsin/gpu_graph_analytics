import numpy as np
import cupy as cp
import time
from src.data_structures.gpu_graph import GPUGraph
from src.algorithms.node2vec import Node2Vec
from src.algorithms.shortest_paths import ShortestPaths
from src.profiling.performance_profiler import PerformanceProfiler

def create_random_weighted_graph(num_nodes: int, edge_probability: float = 0.1) -> tuple:
    """Create a random weighted graph for testing."""
    edges = []
    weights = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            if i != j and np.random.random() < edge_probability:
                edges.append([i, j])
                weights.append(np.random.uniform(0.1, 1.0))
    return np.array(edges), np.array(weights)

def main():
    # Initialize profiler
    profiler = PerformanceProfiler()
    profiler.start_session()
    
    # Create a random weighted graph
    print("Creating random graph...")
    num_nodes = 1000
    edges, weights = create_random_weighted_graph(num_nodes)
    
    # Create GPU graph
    with profiler.profile_kernel("graph_creation"):
        graph = GPUGraph.from_edge_list(edges, weights)
    
    # Generate node embeddings
    print("\nGenerating node embeddings...")
    with profiler.profile_kernel("node2vec"):
        node2vec = Node2Vec(dimensions=128, walk_length=80, num_walks=10)
        embeddings = node2vec.train(graph)
    
    # Print sample embeddings
    print("\nSample node embeddings (first 3 nodes, first 5 dimensions):")
    print(embeddings[:3, :5].get())
    
    # Compute shortest paths
    print("\nComputing shortest paths...")
    shortest_paths = ShortestPaths()
    
    # Test Bellman-Ford
    source_node = 0
    with profiler.profile_kernel("bellman_ford"):
        distances_bf = shortest_paths.bellman_ford(graph, source_node)
    
    print(f"\nBellman-Ford distances from node {source_node} to first 5 nodes:")
    print(distances_bf[:5].get())
    
    # Test Dijkstra
    with profiler.profile_kernel("dijkstra"):
        distances_dj, predecessors = shortest_paths.dijkstra(graph, source_node)
    
    print(f"\nDijkstra distances from node {source_node} to first 5 nodes:")
    print(distances_dj[:5].get())
    
    # Get example path
    target_node = 5
    path = shortest_paths.get_path(predecessors, target_node)
    print(f"\nShortest path from node {source_node} to node {target_node}:")
    print(" -> ".join(map(str, path)))
    
    # Print profiling results
    print("\nPerformance Profile:")
    profiler.print_summary()
    
    # Export Chrome trace for detailed analysis
    profiler.export_chrome_trace("profile_trace.json")
    print("\nProfile trace exported to 'profile_trace.json'")
    
    # Compute some basic graph metrics
    print("\nGraph Metrics:")
    print(f"Number of nodes: {graph.num_nodes}")
    print(f"Number of edges: {graph.num_edges}")
    
    # Memory usage
    memory_usage = cp.get_default_memory_pool().used_bytes()
    print(f"Current GPU memory usage: {memory_usage/1024/1024:.2f} MB")

if __name__ == "__main__":
    main()

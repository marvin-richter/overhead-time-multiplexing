import logging
import networkx as nx
from collections import defaultdict, deque
import random

logger = logging.getLogger(__name__)


def cluster_qubits_bfs(graph, group_sizes, refinement_steps=0):
    """
    Cluster qubits using BFS to create compact, connected clusters with balanced sizes.

    Args:
        graph: NetworkX graph representing qubit connectivity
        max_cluster_size: Maximum number of qubits per cluster (switch capacity)

    Returns:
        dict: mapping from qubit (node) to cluster_id
    """

    def bfs_cluster(graph, seed, assigned, target_size):
        """
        Grow a single cluster using BFS from the seed qubit to reach target size.

        Args:
            graph: NetworkX graph
            seed: Starting qubit for this cluster
            assigned: Set of already assigned qubits
            target_size: Exact target size for this cluster

        Returns:
            list: qubits in this cluster
        """
        if seed in assigned:
            return []

        cluster = [seed]
        queue = deque([seed])
        visited = {seed}

        while queue and len(cluster) < target_size:
            current = queue.popleft()

            # get all unassigned neighbors
            unassigned_neighbors = [
                n for n in graph.neighbors(current) if n not in assigned and n not in visited
            ]

            # rank them by connection to cluster (more: better), orphan risk (higher priority), and degree (less: better)
            unassigned_neighbors.sort(
                key=lambda n: (
                    -len(
                        [nn for nn in graph.neighbors(n) if nn in cluster]
                    ),  # More connections to cluster is better
                    len(
                        [
                            nn
                            for nn in graph.neighbors(n)
                            if nn not in assigned and nn not in visited
                        ]
                    ),  # Fewer unassigned neighbors = higher orphan risk = higher priority
                    graph.degree(n),  # Less degree is better
                )
            )

            for neighbor in unassigned_neighbors:
                if len(cluster) < target_size:
                    cluster.append(neighbor)
                    queue.append(neighbor)
                    visited.add(neighbor)
                else:
                    break

        # If we couldn't fill the cluster through BFS (disconnected components),
        # fill remaining spots with unassigned qubits. start with orphans (no unassigned neighbors)
        if len(cluster) < target_size:
            all_nodes = list(graph.nodes())

            orphans = [
                node
                for node in all_nodes
                if node not in assigned
                and node not in visited
                and len([n for n in graph.neighbors(node) if n not in assigned]) == 0
            ]

            # sort by distance to cluster (closer = better)
            orphans.sort(
                key=lambda n: min(
                    nx.shortest_path_length(graph, source=n, target=qubit) for qubit in cluster
                )
            )

            for node in orphans:
                if node not in visited and len(cluster) < target_size:
                    cluster.append(node)
                    visited.add(node)

            # If still not enough, fill with any unassigned qubits
            if len(cluster) < target_size:
                random.shuffle(all_nodes)

                for node in all_nodes:
                    if node not in assigned and node not in visited and len(cluster) < target_size:
                        cluster.append(node)
                        visited.add(node)

        return cluster

    if graph.number_of_nodes() == 0:
        return {}

    # Track which qubits have been assigned to clusters
    assigned = set()
    qubit_to_cluster = {}

    # Get all nodes and shuffle for better distribution of seed points
    nodes = list(graph.nodes())

    for cluster_id, target_size in enumerate(group_sizes):
        # Find an unassigned seed qubit
        seed = None

        # Find the best unassigned seed qubit
        unassigned_nodes = [n for n in nodes if n not in assigned]
        if not unassigned_nodes:
            break  # No more unassigned qubits

        # Choose seed with highest degree among unassigned neighbors
        seed = max(
            unassigned_nodes,
            key=lambda n: len([nb for nb in graph.neighbors(n) if nb not in assigned]),
        )

        # Start new cluster with BFS from this seed
        cluster = bfs_cluster(graph, seed, assigned, target_size)

        # Assign cluster ID to all qubits in this cluster
        for qubit in cluster:
            qubit_to_cluster[qubit] = cluster_id
            assigned.add(qubit)

    # Track refinement statistics
    refinement_stats = []

    # Initial statistics (step 0)
    initial_compactness = avg_cluster_compactness(graph, qubit_to_cluster)
    initial_connected = count_connected_clusters(graph, qubit_to_cluster)
    refinement_stats.append(
        {
            "step": 0,
            "connected_clusters": initial_connected,
            "avg_compactness": initial_compactness,
        }
    )

    improvement_found = False
    for step in range(refinement_steps):
        # refine clusters - one iteration at a time
        old_qubit_to_cluster = qubit_to_cluster.copy()
        qubit_to_cluster = refine_clusters(graph, qubit_to_cluster)

        # Check if any improvement was made
        if qubit_to_cluster == old_qubit_to_cluster:
            break

        improvement_found = True

        # Calculate statistics for this step
        step_compactness = avg_cluster_compactness(graph, qubit_to_cluster)
        step_connected = count_connected_clusters(graph, qubit_to_cluster)
        refinement_stats.append(
            {
                "step": step + 1,
                "connected_clusters": step_connected,
                "avg_compactness": step_compactness,
            }
        )

    # Log statistics table if any improvement was found
    if improvement_found:
        lines = [
            "\nRefinement Statistics:",
            "=" * 50,
            f"{'Step':<6} {'Connected Clusters':<18} {'Avg Compactness':<15}",
            "-" * 50,
        ]
        for stat in refinement_stats:
            lines.append(
                f"{stat['step']:<6} {stat['connected_clusters']:<18} {stat['avg_compactness']:<15.4f}"
            )
        lines.append("=" * 50)
        logger.debug("\n".join(lines))

    return qubit_to_cluster


def avg_cluster_compactness(graph, qubit_to_cluster):
    """
    Calculate average compactness of clusters based on edge density.

    Args:
        graph: NetworkX graph
        qubit_to_cluster: mapping from qubit to cluster_id

    Returns:
        float: average compactness (edge density) of clusters
    """

    clusters = defaultdict(list)
    for qubit, cluster_id in qubit_to_cluster.items():
        clusters[cluster_id].append(qubit)

    total_compactness = 0.0
    num_clusters = 0

    for cluster_id, qubits in clusters.items():
        if len(qubits) > 1:
            compactness = subgraph_compactness_density(graph, qubits)
            total_compactness += compactness
            num_clusters += 1

    return total_compactness / num_clusters if num_clusters > 0 else 0.0


def subgraph_compactness_density(G, subgraph_nodes):
    """Compactness as edge density: edges / max_possible_edges"""
    subgraph = G.subgraph(subgraph_nodes)
    n = len(subgraph_nodes)
    if n <= 1:
        return 0.0

    actual_edges = subgraph.number_of_edges()
    max_edges = n * (n - 1) // 2  # for undirected graphs
    return actual_edges / max_edges


def refine_clusters(graph, qubit_to_cluster):
    """Try to improve clusters by trying to swap qubits between clusters and compare
    compactness (edge density) of the clusters. Performs only one iteration."""

    def rebuild_clusters():
        clusters = defaultdict(list)
        for qubit, cluster_id in qubit_to_cluster.items():
            clusters[cluster_id].append(qubit)
        return clusters

    clusters = rebuild_clusters()

    # Calculate initial compactness
    compactness = {
        cluster_id: subgraph_compactness_density(graph, qubits)
        for cluster_id, qubits in clusters.items()
    }

    # Store initial connectedness
    connected = {
        cluster_id: nx.is_connected(graph.subgraph(qubits))
        for cluster_id, qubits in clusters.items()
    }

    # Use list() to avoid dictionary changed during iteration
    cluster_items = list(clusters.items())

    for cluster_id, qubits in cluster_items:
        # get all qubits neighboring this cluster
        neighboring_qubits = set()
        for qubit in qubits:
            for neighbor in graph.neighbors(qubit):
                if neighbor not in qubits:
                    neighboring_qubits.add(neighbor)

        for other_cluster_id, other_qubits in cluster_items:
            if cluster_id == other_cluster_id:
                continue

            # Try swapping one qubit from each cluster
            for q2 in other_qubits:
                if q2 not in neighboring_qubits:
                    continue
                for q1 in qubits:
                    # Swap q1 and q2
                    new_qubit_to_cluster = qubit_to_cluster.copy()
                    new_qubit_to_cluster[q1] = other_cluster_id
                    new_qubit_to_cluster[q2] = cluster_id

                    # Recalculate compactness
                    new_clusters = defaultdict(list)
                    for qubit, cid in new_qubit_to_cluster.items():
                        new_clusters[cid].append(qubit)

                    new_compactness = {
                        cid: subgraph_compactness_density(graph, qs)
                        for cid, qs in new_clusters.items()
                    }

                    # check if subgraphs are now connected
                    new_connected = {
                        cid: nx.is_connected(graph.subgraph(qs))
                        for cid, qs in new_clusters.items()
                    }

                    # Store old values before updating
                    old_connected_cluster = connected[cluster_id]
                    old_connected_other = connected[other_cluster_id]
                    old_compactness_cluster = compactness[cluster_id]
                    old_compactness_other = compactness[other_cluster_id]

                    count_connected_before = old_connected_cluster + old_connected_other
                    count_connected_after = (
                        new_connected[cluster_id] + new_connected[other_cluster_id]
                    )

                    # Calculate total compactness before and after
                    total_compactness_before = old_compactness_cluster + old_compactness_other
                    total_compactness_after = (
                        new_compactness[cluster_id] + new_compactness[other_cluster_id]
                    )

                    # Weight connectivity and compactness together
                    connectivity_weight = 0.7  # Adjust based on importance
                    compactness_weight = 0.3

                    old_score = (
                        connectivity_weight * count_connected_before
                        + compactness_weight * total_compactness_before
                    )
                    new_score = (
                        connectivity_weight * count_connected_after
                        + compactness_weight * total_compactness_after
                    )

                    if new_score > old_score * 1.01:  # 1% improvement threshold
                        return new_qubit_to_cluster

    # No improvement found in this iteration
    return qubit_to_cluster


def count_connected_clusters(graph, qubit_to_cluster):
    """
    Count the number of connected clusters.

    Args:
        graph: NetworkX graph
        qubit_to_cluster: mapping from qubit to cluster_id

    Returns:
        int: number of connected clusters
    """
    clusters = defaultdict(list)
    for qubit, cluster_id in qubit_to_cluster.items():
        clusters[cluster_id].append(qubit)

    connected_count = 0
    for cluster_id, qubits in clusters.items():
        if len(qubits) > 1:
            subgraph = graph.subgraph(qubits)
            if nx.is_connected(subgraph):
                connected_count += 1
        else:
            # Single-node clusters are considered "connected"
            connected_count += 1

    return connected_count

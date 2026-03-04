from collections import defaultdict
import logging
import networkx as nx
import random
from concurrent.futures import ThreadPoolExecutor, as_completed


# Set up logging
logger = logging.getLogger(__name__)


def calculate_coloring_metrics(G, coloring):
    """Calculate minimum and average distance metrics for a coloring"""
    if coloring is None:
        return 0, 0.0

    min_distance = float("inf")
    min_distances = {}
    total_distance = 0
    pair_count = 0

    # Group nodes by color
    color_groups = defaultdict(list)
    for node, color in coloring.items():
        color_groups[color].append(node)

    # Calculate distances within each color group
    for color, nodes in color_groups.items():
        if len(nodes) < 2:
            continue

        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                try:
                    dist = nx.shortest_path_length(G, source=nodes[i], target=nodes[j])
                    min_distance = min(min_distance, dist)
                    total_distance += dist
                    pair_count += 1
                except nx.NetworkXNoPath:
                    # If nodes are disconnected, consider distance as infinity
                    continue
        min_distances[color] = min_distance

    if pair_count == 0:
        return float("inf"), 0.0

    # avg_distance = total_distance / pair_count if pair_count > 0 else 0.0
    avg_min_distance = sum(min_distances.values()) / len(min_distances) if min_distances else 0.0

    return min_distance, avg_min_distance


def calculate_coloring_metrics_cached(G, coloring, distance_cache=None):
    """Calculate metrics with distance caching"""
    if distance_cache is None:
        distance_cache = {}

    if coloring is None:
        return 0, 0.0, distance_cache

    min_distance = float("inf")
    min_distances = {}
    total_distance = 0
    pair_count = 0

    color_groups = defaultdict(list)
    for node, color in coloring.items():
        color_groups[color].append(node)

    for color, nodes in color_groups.items():
        if len(nodes) < 2:
            continue

        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                node_pair = tuple(sorted([nodes[i], nodes[j]]))

                if node_pair in distance_cache:
                    dist = distance_cache[node_pair]
                else:
                    try:
                        dist = nx.shortest_path_length(G, source=nodes[i], target=nodes[j])
                        distance_cache[node_pair] = dist
                    except nx.NetworkXNoPath:
                        distance_cache[node_pair] = float("inf")
                        continue

                min_distance = min(min_distance, dist)
                total_distance += dist
                pair_count += 1
        min_distances[color] = min_distance

    if pair_count == 0:
        return float("inf"), 0.0, distance_cache

    # avg_distance = total_distance / pair_count
    avg_min_distance = sum(min_distances.values()) / len(min_distances) if min_distances else 0.0
    return min_distance, avg_min_distance, distance_cache


def get_k_distance_graph(G, k):
    """Create a graph where edges connect nodes that are within distance k-1"""
    k_graph = nx.Graph()
    k_graph.add_nodes_from(G.nodes())

    for node in G.nodes():
        distances = nx.single_source_shortest_path_length(G, node, cutoff=k - 1)
        for other_node, dist in distances.items():
            if node != other_node and dist < k:
                k_graph.add_edge(node, other_node)

    return k_graph


def exact_k_distance_coloring(G, color_sizes, k, use_parallel=True, max_workers=None):
    """
    Attempts to find an exact k-distance coloring using DSATUR and parallel multi-start approaches.

    Returns:
        tuple: (coloring, achieved_min_distance, average_distance, success)
        - coloring: node -> color mapping (None if no exact solution found)
        - achieved_min_distance: minimum distance between same-colored nodes
        - average_distance: average distance between same-colored nodes
        - success: True if exact k-distance coloring was achieved
    """

    # Convert color_sizes to dict if needed
    if isinstance(color_sizes, list):
        color_sizes = {i: size for i, size in enumerate(color_sizes)}

    # Check feasibility
    total_nodes = len(G.nodes())
    total_capacity = sum(color_sizes.values())

    if total_nodes > total_capacity:
        logger.debug(
            f"Impossible: {total_nodes} nodes but only {total_capacity} total color capacity"
        )
        return None, 0, 0.0, False

    logger.debug(f"Attempting exact {k}-distance coloring...")

    # Build k-distance conflict graph
    k_graph = get_k_distance_graph(G, k)

    # Try DSATUR approach first
    try:
        result = greedy_dsatur_coloring(G, color_sizes, k_graph)
        if result is not None:
            min_dist, avg_dist = calculate_coloring_metrics(G, result)
            logger.debug(f"Exact {k}-distance coloring achieved!")
            return result, min_dist, avg_dist, True
    except Exception:
        pass

    # Try multi-start parallel approach
    if use_parallel:
        try:
            result = multi_start_approach_parallel(
                G, color_sizes, k_graph, num_starts=50, max_workers=max_workers
            )
            if result is not None:
                min_dist, avg_dist = calculate_coloring_metrics(G, result)
                logger.debug(f"Exact {k}-distance coloring achieved via parallel approach!")
                return result, min_dist, avg_dist, True
        except Exception:
            pass

    # No exact solution found
    logger.debug(f"Exact {k}-distance coloring not possible.")
    return None, 0, 0.0, False


def heuristic_distance_coloring(
    G, color_sizes, num_attempts=10, use_parallel=True, max_workers=None
):
    """
    Attempts to find the best possible coloring using heuristic methods when exact k-distance
    coloring is not possible.

    Returns:
        tuple: (coloring, achieved_min_distance, average_distance, success)
        - coloring: node -> color mapping
        - achieved_min_distance: minimum distance between same-colored nodes
        - average_distance: average distance between same-colored nodes
        - success: Always False (since this is the fallback method)
    """
    # Convert color_sizes to dict if needed
    if isinstance(color_sizes, list):
        color_sizes = {i: size for i, size in enumerate(color_sizes)}

    # Check feasibility
    total_nodes = len(G.nodes())
    total_capacity = sum(color_sizes.values())

    if total_nodes > total_capacity:
        logger.debug(
            f"Impossible: {total_nodes} nodes but only {total_capacity} total color capacity"
        )
        return None, 0, 0.0, False

    logger.debug("Optimizing for best distance metrics using heuristic methods...")

    if use_parallel:
        return heuristic_distance_coloring_parallel(G, color_sizes, num_attempts, max_workers)
    else:
        return heuristic_distance_coloring_sequential(G, color_sizes, num_attempts)


def heuristic_distance_coloring_sequential(G, color_sizes, num_attempts):
    """Sequential version of heuristic distance coloring"""
    best_coloring = None
    best_min_dist = 0
    best_avg_dist = 0.0

    for attempt in range(num_attempts):
        logger.debug(f"Heuristic attempt {attempt + 1}/{num_attempts}")

        coloring, min_dist, avg_dist = multi_start_with_metrics(
            G, color_sizes, num_starts=10, use_random_seeds=True
        )

        if coloring is not None:
            # Keep if better (prioritize minimum distance, then average distance)
            if min_dist > best_min_dist or (
                min_dist == best_min_dist and avg_dist > best_avg_dist
            ):
                best_coloring = coloring
                best_min_dist = min_dist
                best_avg_dist = avg_dist
                logger.debug(
                    f"  Found better solution: min_dist={min_dist}, avg_dist={avg_dist:.2f}"
                )

    if best_coloring is not None:
        logger.debug("Best achievable coloring found:")
        logger.debug(f"  Minimum distance: {best_min_dist}")
        logger.debug(f"  Average distance: {best_avg_dist:.2f}")
        return best_coloring, best_min_dist, best_avg_dist, False
    else:
        logger.debug("No valid coloring found even with relaxed constraints")
        return None, 0, 0.0, False


def heuristic_distance_coloring_parallel(G, color_sizes, num_attempts, max_workers=None):
    """Parallel version of heuristic distance coloring"""
    if max_workers is None:
        max_workers = min(num_attempts, 8)  # Default to 8 workers max

    # Generate different seeds for each attempt
    seeds = [random.randint(0, 1000000) for _ in range(num_attempts)]

    def single_heuristic_attempt(seed):
        """Single heuristic attempt worker"""
        coloring, min_dist, avg_dist = multi_start_with_metrics(
            G, color_sizes, num_starts=10, use_random_seeds=True
        )
        return coloring, min_dist, avg_dist, seed

    best_coloring = None
    best_min_dist = 0
    best_avg_dist = 0.0
    completed_attempts = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_seed = {executor.submit(single_heuristic_attempt, seed): seed for seed in seeds}

        # Process completed tasks as they finish
        for future in as_completed(future_to_seed):
            try:
                coloring, min_dist, avg_dist, seed = future.result()
                completed_attempts += 1

                logger.debug(f"Heuristic attempt {completed_attempts}/{num_attempts} completed")

                if coloring is not None:
                    # Keep if better (prioritize minimum distance, then average distance)
                    if min_dist > best_min_dist or (
                        min_dist == best_min_dist and avg_dist > best_avg_dist
                    ):
                        best_coloring = coloring
                        best_min_dist = min_dist
                        best_avg_dist = avg_dist
                        logger.debug(
                            f"  Found better solution: min_dist={min_dist}, avg_dist={avg_dist:.2f}"
                        )

            except Exception as exc:
                seed = future_to_seed[future]
                logger.debug(f"Heuristic attempt with seed {seed} generated an exception: {exc}")

    if best_coloring is not None:
        logger.debug("Best achievable coloring found:")
        logger.debug(f"  Minimum distance: {best_min_dist}")
        logger.debug(f"  Average distance: {best_avg_dist:.2f}")
        return best_coloring, best_min_dist, best_avg_dist, False
    else:
        logger.debug("No valid coloring found even with relaxed constraints")
        return None, 0, 0.0, False


def distance_k_coloring_with_fallback(G, color_sizes, k, use_parallel=True, max_workers=None):
    """
    Enhanced distance-k coloring that falls back to optimizing distance metrics
    when exact k-distance coloring is not possible.

    Returns:
        tuple: (coloring, achieved_min_distance, average_distance, success)
        - coloring: node -> color mapping
        - achieved_min_distance: minimum distance between same-colored nodes
        - average_distance: average distance between same-colored nodes
        - success: True if exact k-distance coloring was achieved
    """
    # First try exact k-distance coloring
    coloring, min_dist, avg_dist, success = exact_k_distance_coloring(
        G, color_sizes, k, use_parallel, max_workers
    )

    if success:
        return coloring, min_dist, avg_dist, success

    # If exact failed, try heuristic methods 10 times
    logger.debug("Falling back to heuristic distance optimization...")
    return heuristic_distance_coloring(
        G, color_sizes, num_attempts=10, use_parallel=use_parallel, max_workers=max_workers
    )


def verify_coloring_with_metrics(G, coloring, color_sizes):
    """Enhanced verification that also reports distance metrics"""
    if coloring is None:
        return False, 0, 0.0

    # Check color capacity constraints
    color_counts = defaultdict(int)
    for color in coloring.values():
        color_counts[color] += 1

    for color, count in color_counts.items():
        if count > color_sizes[color]:
            logger.debug(f"Color {color} has {count} nodes but capacity is {color_sizes[color]}")
            return False, 0, 0.0

    # Calculate distance metrics
    def calculate_coloring_metrics(G, coloring):
        min_distance = float("inf")
        total_distance = 0
        pair_count = 0

        color_groups = defaultdict(list)
        for node, color in coloring.items():
            color_groups[color].append(node)

        for color, nodes in color_groups.items():
            if len(nodes) < 2:
                continue

            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    try:
                        dist = nx.shortest_path_length(G, source=nodes[i], target=nodes[j])
                        min_distance = min(min_distance, dist)
                        total_distance += dist
                        pair_count += 1
                    except nx.NetworkXNoPath:
                        continue

        if pair_count == 0:
            return float("inf"), 0.0

        avg_distance = total_distance / pair_count if pair_count > 0 else 0.0
        return min_distance, avg_distance

    min_dist, avg_dist = calculate_coloring_metrics(G, coloring)

    logger.debug("Coloring validation:")
    logger.debug("  Capacity constraints: PASSED")
    logger.debug(f"  Minimum distance between same-colored nodes: {min_dist}")
    logger.debug(f"  Average distance between same-colored nodes: {avg_dist:.2f}")

    return True, min_dist, avg_dist


def greedy_dsatur_coloring(G, color_sizes, k_graph):
    """
    DSATUR-based greedy coloring with capacity constraints.
    More sophisticated than simple greedy.
    """
    nodes = list(G.nodes())
    node_colors = {node: None for node in nodes}
    used_colors = {color: 0 for color in color_sizes}

    # Calculate initial saturation degrees
    def get_saturation_degree(node):
        """Number of different colors in the k-neighborhood"""
        if node_colors[node] is not None:
            return float("inf")

        neighbor_colors = set()
        for neighbor in k_graph.neighbors(node):
            if node_colors[neighbor] is not None:
                neighbor_colors.add(node_colors[neighbor])
        return len(neighbor_colors)

    def get_available_colors(node):
        """Get colors that don't conflict with k-neighbors"""
        forbidden_colors = set()
        for neighbor in k_graph.neighbors(node):
            if node_colors[neighbor] is not None:
                forbidden_colors.add(node_colors[neighbor])

        available = []
        for color in color_sizes:
            if color not in forbidden_colors and used_colors[color] < color_sizes[color]:
                available.append(color)
        return available

    # Color nodes in order of decreasing saturation degree
    colored_nodes = 0
    while colored_nodes < len(nodes):
        # Find uncolored node with highest saturation degree
        best_node = None
        best_saturation = -1
        best_degree = -1

        for node in nodes:
            if node_colors[node] is None:
                saturation = get_saturation_degree(node)
                degree = G.degree(node)

                if saturation > best_saturation or (
                    saturation == best_saturation and degree > best_degree
                ):
                    best_node = node
                    best_saturation = saturation
                    best_degree = degree

        if best_node is None:
            break

        # Try to color the best node
        available_colors = get_available_colors(best_node)
        if not available_colors:
            return None

        # Choose color with most remaining capacity
        best_color = max(available_colors, key=lambda c: color_sizes[c] - used_colors[c])

        node_colors[best_node] = best_color
        used_colors[best_color] += 1
        colored_nodes += 1

    return node_colors


def single_attempt_worker(G, color_sizes, k_graph, seed=None):
    """
    Single attempt worker for parallel execution.
    Returns a solution if found, None otherwise.
    """
    if seed is not None:
        random.seed(seed)

    # Randomize node ordering for different starting points
    nodes = list(G.nodes())
    random.shuffle(nodes)

    # Modified greedy with randomized color selection
    node_colors = {node: None for node in nodes}
    used_colors = {color: 0 for color in color_sizes}

    success = True
    for node in nodes:
        available_colors = []
        for color in color_sizes:
            if used_colors[color] >= color_sizes[color]:
                continue

            # Check k-distance constraint
            valid = True
            for neighbor in k_graph.neighbors(node):
                if node_colors[neighbor] == color:
                    valid = False
                    break

            if valid:
                available_colors.append(color)

        if available_colors:
            # Add randomness in color selection
            if len(available_colors) > 1 and random.random() < 0.3:
                chosen_color = random.choice(available_colors)
            else:
                chosen_color = max(available_colors, key=lambda c: color_sizes[c] - used_colors[c])

            node_colors[node] = chosen_color
            used_colors[chosen_color] += 1
        else:
            success = False
            break

    if success:
        # Apply local search improvement
        improved_solution = large_neighborhood_search(
            G, color_sizes, k_graph, node_colors, max_iterations=50
        )
        return improved_solution

    return None


def large_neighborhood_search(G, color_sizes, k_graph, initial_solution=None, max_iterations=100):
    """
    Local search improvement using large neighborhood search.
    """
    if initial_solution is None:
        current_solution = greedy_dsatur_coloring(G, color_sizes, k_graph)
    else:
        current_solution = initial_solution.copy()

    if current_solution is None:
        return None

    best_solution = current_solution.copy()

    for iteration in range(max_iterations):
        # Select a subset of nodes to recolor
        subset_size = min(10, len(G.nodes()) // 10)  # Adaptive subset size
        nodes_to_recolor = random.sample(list(G.nodes()), subset_size)

        # Remove colors from selected nodes
        temp_solution = current_solution.copy()
        used_colors = {color: 0 for color in color_sizes}

        for node in G.nodes():
            if node not in nodes_to_recolor:
                used_colors[temp_solution[node]] += 1
            else:
                temp_solution[node] = None

        # Try to recolor the selected nodes
        success = True
        for node in nodes_to_recolor:
            available_colors = []
            for color in color_sizes:
                if used_colors[color] >= color_sizes[color]:
                    continue

                # Check k-distance constraint
                valid = True
                for neighbor in k_graph.neighbors(node):
                    if temp_solution[neighbor] == color:
                        valid = False
                        break

                if valid:
                    available_colors.append(color)

            if available_colors:
                # Choose color with most remaining capacity
                best_color = max(available_colors, key=lambda c: color_sizes[c] - used_colors[c])
                temp_solution[node] = best_color
                used_colors[best_color] += 1
            else:
                success = False
                break

        if success:
            current_solution = temp_solution
            best_solution = current_solution.copy()

    return best_solution


def multi_start_approach_parallel(G, color_sizes, k_graph, num_starts=5, max_workers=None):
    """
    Try multiple random starting configurations in parallel.
    Stops as soon as one solution is found.
    """
    if max_workers is None:
        max_workers = min(num_starts, 4)  # Default to 4 workers max

    # Use different seeds for reproducible but varied results
    seeds = [random.randint(0, 1000000) for _ in range(num_starts)]

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_seed = {
            executor.submit(single_attempt_worker, G, color_sizes, k_graph, seed): seed
            for seed in seeds
        }

        # Process completed tasks as they finish
        for future in as_completed(future_to_seed):
            try:
                result = future.result()
                if result is not None:
                    # Cancel remaining tasks
                    for f in future_to_seed:
                        if f != future:
                            f.cancel()
                    return result
            except Exception as exc:
                seed = future_to_seed[future]
                logger.debug(f"Attempt with seed {seed} generated an exception: {exc}")

    # No solution found
    return None


def multi_start_with_metrics(G, color_sizes, num_starts=20, use_random_seeds=True):
    """
    Multi-start approach optimizing for distance metrics with better randomization
    """
    best_coloring = None
    best_min_dist = 0
    best_avg_dist = 0.0

    # Generate different random seeds for each start
    if use_random_seeds:
        seeds = [random.randint(0, 1000000) for _ in range(num_starts)]
    else:
        seeds = list(range(num_starts))

    for start in range(num_starts):
        # Set a unique seed for this iteration
        if use_random_seeds:
            random.seed(seeds[start])

        # if there are only two colors, skip greedy and use checkerboard initilization
        if len(color_sizes) == 2:
            # Use checkerboard initialization for two colors
            initial_coloring = bfs_two_coloring(G, color_sizes, seed=seeds[start])

            improved_coloring = initial_coloring

        else:
            # Generate initial coloring with randomization
            initial_coloring = adaptive_greedy_coloring(G, color_sizes, randomize=True)

            if initial_coloring is None:
                continue

            # Apply some random perturbations to the initial solution
            if start > 0:  # Skip perturbation for the first attempt
                initial_coloring = random_perturbation(G, color_sizes, initial_coloring)

            # Improve it with randomized local search
            improved_coloring = iterative_improvement(
                G,
                color_sizes,
                initial_coloring,
                max_iterations=50 + random.randint(0, 50),
                randomize=True,
            )
            if improved_coloring is None:
                continue

        # Evaluate metrics
        min_dist, avg_dist = calculate_coloring_metrics(G, improved_coloring)

        # Keep if better (with small random tolerance to escape local optima)
        tolerance = 0.01 if start > num_starts // 2 else 0.0  # More tolerance in later iterations

        # Convert coloring to groups
        groups = defaultdict(list)
        for node, group_id in improved_coloring.items():
            groups[group_id].append(node)

        # Verify that groups have size specified in group_sizes
        for group_id, group in groups.items():
            if len(group) != color_sizes[group_id]:
                raise ValueError(
                    f"Group {group_id} has size {len(group)}, expected {color_sizes[group_id]}."
                )
        if min_dist > best_min_dist or (
            min_dist == best_min_dist and avg_dist > best_avg_dist + tolerance
        ):
            best_coloring = improved_coloring
            best_min_dist = min_dist
            best_avg_dist = avg_dist
            logger.debug(
                f"  Found better solution: min_dist={min_dist}, avg_dist={avg_dist:.2f} (start {start + 1}/{num_starts})"
            )

    return best_coloring, best_min_dist, best_avg_dist


def bfs_two_coloring(G, color_sizes, seed=None):
    """
    Generates a two-coloring using BFS, starting from a random node.

    More info:
    There is a simple algorithm for determining whether a graph is 2-colorable and assigning colors to its vertices: do a breadth-first search, assigning "red" to the first layer, "blue" to the second layer, "red" to the third layer, etc. Then go over all the edges and check whether the two endpoints of this edge have different colors. This algorithm is O(|V|+|E|) and the last step ensures its correctness.

    """

    if seed is not None:
        random.seed(seed)

    starting_node = random.choice(list(G.nodes()))

    coloring = {node: None for node in G.nodes()}
    queue = [starting_node]
    coloring[starting_node] = 0  # Start with color 0
    color_sizes = {0: color_sizes[0], 1: color_sizes[1]}  # Ensure we have two colors

    used_colors = {0: 1, 1: 0}  # Start with one node colored

    while queue:
        current_node = queue.pop(0)
        current_color = coloring[current_node]

        # Get neighbors and color them with the opposite color
        for neighbor in G.neighbors(current_node):
            if coloring[neighbor] is None:
                # Check if we can color with the opposite color
                opposite_color = 1 - current_color
                if used_colors[opposite_color] < color_sizes[opposite_color]:
                    coloring[neighbor] = opposite_color
                    used_colors[opposite_color] += 1
                    queue.append(neighbor)
                else:
                    # If we can't color with the opposite, try the current color
                    if used_colors[current_color] < color_sizes[current_color]:
                        coloring[neighbor] = current_color
                        used_colors[current_color] += 1
                        queue.append(neighbor)
            elif coloring[neighbor] == current_color:
                # If neighbor already has the same color, we can't 2-color this graph. log a warning and continue coloring to find sub-optimal colorings.

                pass

    return coloring


def random_perturbation(G, color_sizes, coloring, perturbation_rate=0.1):
    """
    Apply random perturbations to escape local optima
    """
    if coloring is None:
        return None

    perturbed_coloring = coloring.copy()
    nodes = list(G.nodes())
    num_to_perturb = max(1, int(len(nodes) * perturbation_rate))

    # Select random nodes to perturb
    nodes_to_perturb = random.sample(nodes, num_to_perturb)

    for node in nodes_to_perturb:
        current_color = perturbed_coloring[node]

        # Count current color usage
        used_colors = defaultdict(int)
        for n, c in perturbed_coloring.items():
            used_colors[c] += 1

        # Find available colors (excluding current)
        available_colors = []
        for color in color_sizes:
            if color != current_color:
                if color not in used_colors or used_colors[color] < color_sizes[color]:
                    available_colors.append(color)

        # Randomly select a new color
        if available_colors:
            new_color = random.choice(available_colors)
            perturbed_coloring[node] = new_color

    return perturbed_coloring


def iterative_improvement(G, color_sizes, initial_coloring, max_iterations=100, randomize=True):
    """
    Iteratively improve coloring by trying to increase minimum distance
    """
    current_coloring = initial_coloring.copy()
    current_min_dist, current_avg_dist, distance_cache = calculate_coloring_metrics_cached(
        G, current_coloring
    )

    best_coloring = current_coloring.copy()
    best_min_dist = current_min_dist
    best_avg_dist = current_avg_dist

    for iteration in range(max_iterations):
        # Find pairs of same-colored nodes with minimum distance
        color_groups = defaultdict(list)
        for node, color in current_coloring.items():
            color_groups[color].append(node)

        # Find candidate pairs with low distance
        candidate_pairs = []

        for color, nodes in color_groups.items():
            if len(nodes) < 2:
                continue
            for i in range(len(nodes)):
                for j in range(i + 1, len(nodes)):
                    try:
                        dist = nx.shortest_path_length(G, source=nodes[i], target=nodes[j])
                        candidate_pairs.append((nodes[i], nodes[j], color, dist))
                    except nx.NetworkXNoPath:
                        continue

        if not candidate_pairs:
            break

        # Sort by distance and add some randomness
        candidate_pairs.sort(key=lambda x: x[3])  # Sort by distance

        # Choose from the worst pairs but with some randomness
        if randomize:
            # Take the worst 20% of pairs and choose randomly from them
            worst_pairs = candidate_pairs[: max(1, len(candidate_pairs) // 5)]
            if len(worst_pairs) > 1:
                min_pair = random.choice(worst_pairs)
            else:
                min_pair = worst_pairs[0]
        else:
            min_pair = candidate_pairs[0]

        node1, node2, original_color, min_pair_dist = min_pair

        # Try to recolor one of the nodes (randomly choose which one)
        nodes_to_try = [node1, node2]
        if randomize:
            random.shuffle(nodes_to_try)

        improved = False
        for node_to_recolor in nodes_to_try:
            if improved:
                break

            # Get available colors and randomize order
            available_colors = [c for c in color_sizes if c != original_color]
            if randomize:
                random.shuffle(available_colors)

            for new_color in available_colors:
                # Check if we can move node to new_color
                used_colors = defaultdict(int)
                for n, c in current_coloring.items():
                    used_colors[c] += 1

                if used_colors[new_color] >= color_sizes[new_color]:
                    continue

                # Try the move
                test_coloring = current_coloring.copy()
                test_coloring[node_to_recolor] = new_color

                test_min_dist, test_avg_dist, distance_cache = calculate_coloring_metrics_cached(
                    G, test_coloring, distance_cache
                )

                # Accept if it improves minimum distance or average distance
                if test_min_dist > best_min_dist or (
                    test_min_dist == best_min_dist and test_avg_dist > best_avg_dist
                ):
                    current_coloring = test_coloring
                    best_coloring = test_coloring.copy()
                    best_min_dist = test_min_dist
                    best_avg_dist = test_avg_dist
                    improved = True
                    break

        if not improved:
            # If no improvement, occasionally accept a sideways move for exploration
            if randomize and random.random() < 0.1:
                node_to_recolor = random.choice([node1, node2])
                available_colors = [c for c in color_sizes if c != original_color]
                if available_colors:
                    new_color = random.choice(available_colors)

                    used_colors = defaultdict(int)
                    for n, c in current_coloring.items():
                        used_colors[c] += 1

                    if used_colors[new_color] < color_sizes[new_color]:
                        current_coloring[node_to_recolor] = new_color

    return best_coloring


def adaptive_greedy_coloring(G, color_sizes, randomize=True):
    """
    Adaptive greedy coloring that tries to maximize minimum distance
    """
    nodes = list(G.nodes())
    node_colors = {node: None for node in nodes}
    used_colors = {color: 0 for color in color_sizes}

    if randomize:
        # Add randomization to node ordering
        nodes_by_degree = sorted(nodes, key=lambda n: G.degree(n), reverse=True)

        # Group nodes by degree ranges for partial randomization
        degree_groups = []
        current_group = []
        current_degree = None

        for node in nodes_by_degree:
            degree = G.degree(node)
            if current_degree is None or abs(degree - current_degree) <= 1:
                current_group.append(node)
                current_degree = degree
            else:
                if current_group:
                    random.shuffle(current_group)
                    degree_groups.extend(current_group)
                current_group = [node]
                current_degree = degree

        if current_group:
            random.shuffle(current_group)
            degree_groups.extend(current_group)

        nodes = degree_groups
    else:
        # Sort nodes by degree (higher degree first for better results)
        nodes.sort(key=lambda n: G.degree(n), reverse=True)

    for node in nodes:
        best_color = None
        best_min_dist = 0
        best_avg_dist = 0.0

        # Try each available color
        available_colors = [
            color for color in color_sizes if used_colors[color] < color_sizes[color]
        ]

        if randomize:
            # Shuffle available colors to add randomness
            random.shuffle(available_colors)

        for color in available_colors:
            # Calculate what minimum distance this color would achieve
            same_color_nodes = [n for n in G.nodes() if node_colors[n] == color]

            if not same_color_nodes:
                # First node of this color
                min_dist_for_color = float("inf")
                avg_dist_for_color = 0.0
            else:
                # Calculate distances to existing nodes of this color
                distances = []
                for other_node in same_color_nodes:
                    try:
                        dist = nx.shortest_path_length(G, source=node, target=other_node)
                        distances.append(dist)
                    except nx.NetworkXNoPath:
                        distances.append(float("inf"))

                min_dist_for_color = min(distances) if distances else float("inf")
                avg_dist_for_color = (
                    sum(d for d in distances if d != float("inf")) / len(distances)
                    if distances
                    else 0.0
                )

            # Prefer colors that give better distance metrics
            if min_dist_for_color > best_min_dist or (
                min_dist_for_color == best_min_dist and avg_dist_for_color > best_avg_dist
            ):
                best_color = color
                best_min_dist = min_dist_for_color
                best_avg_dist = avg_dist_for_color

        # Add randomness to color selection when multiple colors are equally good
        if randomize and best_color is not None:
            # Find all colors with the same best metrics
            equally_good_colors = []
            for color in available_colors:
                same_color_nodes = [n for n in G.nodes() if node_colors[n] == color]

                if not same_color_nodes:
                    min_dist_for_color = float("inf")
                    avg_dist_for_color = 0.0
                else:
                    distances = []
                    for other_node in same_color_nodes:
                        try:
                            dist = nx.shortest_path_length(G, source=node, target=other_node)
                            distances.append(dist)
                        except nx.NetworkXNoPath:
                            distances.append(float("inf"))

                    min_dist_for_color = min(distances) if distances else float("inf")
                    avg_dist_for_color = (
                        sum(d for d in distances if d != float("inf")) / len(distances)
                        if distances
                        else 0.0
                    )

                # Check if this color is equally good
                if (
                    min_dist_for_color == best_min_dist
                    and abs(avg_dist_for_color - best_avg_dist) < 0.001
                ):
                    equally_good_colors.append(color)

            # If multiple equally good colors, choose randomly
            if len(equally_good_colors) > 1 and random.random() < 0.4:
                best_color = random.choice(equally_good_colors)

        if best_color is not None:
            node_colors[node] = best_color
            used_colors[best_color] += 1
        else:
            # No valid color found - this shouldn't happen if capacities are sufficient
            return None

    return node_colors

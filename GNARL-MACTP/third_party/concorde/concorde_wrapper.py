import os
import subprocess
import numpy as np
import hashlib
import warnings


def tour_to_next_node(tour: np.ndarray) -> np.ndarray:
    """
    Converts a tour (list of city indices) to a next node mapping.

    Args:
        tour: np.ndarray of shape (n,) with city indices in the order visited.

    Returns:
        next_node: np.ndarray of shape (n,) where next_node[i] is the index of the next city after city i.
    """
    n = len(tour)
    next_node = np.zeros(n, dtype=int)
    for i in range(n):
        next_node[tour[i]] = tour[(i + 1) % n]
    return next_node


def parse_concorde_solution(sol_file: str) -> list[int]:
    """
    Parses a Concorde solution file to extract the tour.

    Args:
        sol_file: Path to the Concorde solution file.

    Returns:
        list[int]: A list of city indices representing the tour.
    """
    with open(sol_file, "r") as f:
        lines = f.readlines()
        n = int(lines[0].strip())
        tour = []
        for line in lines[1:]:
            tour.extend(int(x) for x in line.strip().split())
        assert len(tour) == n, f"Expected {n} nodes in tour, got {len(tour)}"
    return tour


def write_tsp_problem_file(
    coordinates: np.ndarray,
    tsp_file: str,
) -> None:
    """
    Writes a TSP problem in TSPLIB format from coordinates.

    Args:
        coordinates: np.ndarray of shape (n, 2) with city coordinates.
        tsp_file: Path to the output TSP file.
        scale: Multiplier to convert float coordinates to integers (default: 1.0).
    """
    n = coordinates.shape[0]
    assert coordinates.shape == (
        n,
        2,
    ), "coordinates must be a 2D array with shape (n, 2)"
    # Check if coordinates are close to integers
    if not np.allclose(coordinates, np.round(coordinates), rtol=1e-5, atol=1e-5):
        warnings.warn(
            "Coordinates are not near integer values. Concorde expects integer coordinates. "
            "Rounding to nearest integers."
        )

    # Cast to integers for Concorde
    coordinates = np.round(coordinates).astype(np.int32)

    with open(tsp_file, "w") as f:
        f.write("NAME: TSP\n")
        f.write("TYPE: TSP\n")
        f.write("DIMENSION: {}\n".format(n))
        f.write("EDGE_WEIGHT_TYPE: EUC_2D\n")
        f.write("NODE_COORD_SECTION\n")
        for i in range(n):
            f.write("{} {} {}\n".format(i + 1, coordinates[i, 0], coordinates[i, 1]))
        f.write("EOF\n")


def solve_tsp_with_concorde(
    coordinates: np.ndarray,
    concorde_path="third_party/concorde/concorde/TSP/concorde",
    cache_dir="third_party/concorde/cache",
) -> np.ndarray:
    """
    Solves a TSP instance using the Concorde solver with an explicit edge weight matrix.

    Args:
        weights: np.ndarray of shape (n, n) with symmetric edge weights (distance matrix).
        concorde_path: Path to the Concorde executable.
        cache_dir: Directory to store problem and solution files.

    Returns:
        np.ndarray: A tour represented as a mapping from each city index to the next city index in the tour.
    """
    concorde_path = os.path.abspath(concorde_path)
    cache_dir = os.path.abspath(cache_dir)

    coords_hash = hashlib.md5(coordinates.tobytes()).hexdigest()
    os.makedirs(cache_dir, exist_ok=True)
    problem_dir = os.path.join(cache_dir, coords_hash)
    os.makedirs(problem_dir, exist_ok=True)

    tsp_file = os.path.join(problem_dir, "input.tsp")
    sol_file = os.path.join(problem_dir, "output.sol")

    # Check if solution already exists
    if os.path.exists(sol_file):
        tour = parse_concorde_solution(sol_file)
        return tour_to_next_node(np.array(tour))

    write_tsp_problem_file(coordinates, tsp_file)

    # Call Concorde
    cmd = [concorde_path, "-o", sol_file, tsp_file]
    subprocess.run(
        cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=problem_dir
    )

    # Parse solution
    tour = parse_concorde_solution(sol_file)
    return tour_to_next_node(np.array(tour))

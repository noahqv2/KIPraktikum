# Import necessary libraries
import numpy as np  # For numerical operations and array handling
import matplotlib.pyplot as plt  # For visualization of results
import random  # For generating random numbers and selections
import time  # For performance measurements
import tsplib95  # For loading TSP problem instances
import urllib.request  # For downloading files from URLs
from pathlib import Path  # For cross-platform file path handling


class TSP_EA:
    """
    Evolutionary Algorithm implementation for solving the Traveling Salesperson Problem (TSP).

    This class implements a (μ + λ) evolutionary algorithm with permutation representation,
    order crossover, and inversion mutation to find near-optimal solutions for TSP instances.
    """

    def __init__(self, problem, mu, lambda_, mutation_prob, max_generations):
        """
        Initialize the Evolutionary Algorithm for TSP.

        Parameters:
        -----------
        problem : tsplib95.models.Problem
            The TSP problem instance loaded from TSPLIB
        mu : int
            Number of parents selected for next generation (population size)
        lambda_ : int
            Number of offspring generated in each generation
        mutation_prob : float
            Probability of applying mutation to an individual (0.0 to 1.0)
        max_generations : int
            Maximum number of generations to run the algorithm
        """
        self.problem = problem  # The TSP problem instance
        self.mu = mu  # Number of parents
        self.lambda_ = lambda_  # Number of offspring
        self.mutation_prob = mutation_prob  # Mutation probability
        self.max_generations = max_generations  # Maximum generations
        self.dimension = self.problem.dimension  # Number of cities in the problem
        self.distances = self._compute_distance_matrix()  # Pre-compute distances
        self.best_fitness_history = []  # Track best fitness over generations
        self.avg_fitness_history = []  # Track average fitness over generations

    def _compute_distance_matrix(self):
        """
        Calculate the distance matrix for the TSP problem.

        This method computes distances between all pairs of cities based on
        the problem's edge weight type. It handles different TSPLIB formats
        including Euclidean 2D coordinates and explicit distance matrices.

        Returns:
        --------
        numpy.ndarray
            A 2D array where matrix[i,j] is the distance from city i to city j
        """
        n = self.dimension  # Number of cities
        matrix = np.zeros((n, n))  # Initialize distance matrix with zeros

        # For problems with node coordinates (e.g., Euclidean 2D)
        if self.problem.edge_weight_type == 'EUC_2D':
            # Extract coordinates for all cities (node indices in TSPLIB start from 1)
            coords = np.array([self.problem.node_coords[i + 1] for i in range(n)])

            # Calculate Euclidean distance between each pair of cities
            for i in range(n):
                for j in range(n):
                    if i != j:  # Distance to self is 0 (already initialized)
                        matrix[i, j] = np.sqrt(np.sum((coords[i] - coords[j]) ** 2))

        # For problems with explicit distance matrix
        elif hasattr(self.problem, 'edge_weights'):
            # Copy weights from the problem's edge_weights attribute
            for i, j in self.problem.get_edges():
                # Convert 1-indexed to 0-indexed
                matrix[i - 1, j - 1] = self.problem.get_weight(i, j)
                matrix[j - 1, i - 1] = self.problem.get_weight(i, j)  # Assuming symmetric TSP

        # For problems with full matrix format
        elif self.problem.edge_weight_format == 'FULL_MATRIX':
            edges = list(self.problem.get_edges())
            for i, j in edges:
                # Convert 1-indexed to 0-indexed
                matrix[i - 1, j - 1] = self.problem.get_weight(i, j)

        return matrix

    def evaluate_fitness(self, individual):
        """
        Calculate the total distance (fitness) of a tour.

        This method computes the total length of the tour represented by the given individual.
        For TSP, fitness is the total distance traveled, which we aim to minimize.

        Parameters:
        -----------
        individual : list
            A permutation of cities representing a tour

        Returns:
        --------
        float
            The total distance of the tour
        """
        total_distance = 0

        # Sum the distances between consecutive cities in the tour
        for i in range(len(individual)):
            from_city = individual[i]
            # Use modulo to connect the last city back to the first
            to_city = individual[(i + 1) % len(individual)]
            # Add the distance between these cities
            total_distance += self.distances[from_city, to_city]

        return total_distance

    def initialize_population(self, pop_size):
        """
        Create an initial population of random tours.

        Generates a population of random permutations, where each permutation
        represents a possible solution to the TSP problem (a complete tour).

        Parameters:
        -----------
        pop_size : int
            Number of individuals in the population

        Returns:
        --------
        list
            A list of individuals, where each individual is a permutation of cities
        """
        population = []  # Empty population to start
        cities = list(range(self.dimension))  # List of all city indices

        # Generate 'pop_size' random individuals
        for _ in range(pop_size):
            # Create a random permutation of cities (each city appears exactly once)
            tour = random.sample(cities, len(cities))
            population.append(tour)

        return population

    def order_crossover(self, parent1, parent2):
        """
        Order Crossover (OX) operator specialized for TSP.

        This crossover preserves the relative order of cities from both parents
        while ensuring no duplicates. It works as follows:
        1. Select a random subsequence from parent1
        2. Copy this subsequence to the child at the same positions
        3. Fill remaining positions with cities from parent2 in their original order,
           skipping cities already in the child

        Parameters:
        -----------
        parent1 : list
            First parent individual (a permutation of cities)
        parent2 : list
            Second parent individual (a permutation of cities)

        Returns:
        --------
        list
            Child individual created by combining genetic material from both parents
        """
        size = len(parent1)

        # Choose two random crossover points
        cx_points = sorted(random.sample(range(size), 2))
        start, end = cx_points

        # Create a child with empty slots (represented by -1)
        child = [-1] * size

        # Step 1: Copy the segment from parent1 to the child
        for i in range(start, end + 1):
            child[i] = parent1[i]

        # Step 2: Fill the remaining positions with cities from parent2
        j = 0  # Index for parent2
        for i in range(size):
            # Skip positions that are already filled
            if i >= start and i <= end:
                continue

            # Find the next city in parent2 that is not yet in child
            while j < size and parent2[j] in child:
                j += 1

            # If we found a city to add
            if j < size:
                child[i] = parent2[j]
                j += 1

        return child

    def inversion_mutation(self, individual):
        """
        Inversion Mutation operator for TSP.

        This mutation reverses the order of a randomly selected subsequence
        of cities in the tour. This operation:
        1. Preserves the permutation constraint (each city appears exactly once)
        2. Can fix crossed edges by inverting a segment of the tour
        3. Is similar to a 2-opt local search move in TSP heuristics

        The mutation is applied with probability self.mutation_prob.

        Parameters:
        -----------
        individual : list
            An individual (tour) to potentially mutate

        Returns:
        --------
        list
            A new individual, either mutated or a copy of the original
        """
        size = len(individual)

        # Apply mutation with probability mutation_prob
        if random.random() < self.mutation_prob:
            # Choose two random points in the tour
            points = sorted(random.sample(range(size), 2))
            start, end = points

            # Create a copy of the individual
            mutated = individual.copy()

            # Reverse the subsequence between start and end (inclusive)
            # This is the actual inversion mutation operation
            mutated[start:end + 1] = reversed(individual[start:end + 1])

            return mutated

        # If mutation is not applied, return a copy of the original
        return individual.copy()

    def select_parents(self, population, fitnesses, num_parents):
        """
        Tournament selection for parent selection.

        This method selects individuals to become parents for creating offspring.
        It uses tournament selection, which:
        1. Randomly selects a small subset of individuals (tournament)
        2. Chooses the best individual from each tournament
        3. Repeats until enough parents are selected

        Parameters:
        -----------
        population : list
            List of individuals in the current population
        fitnesses : list
            List of fitness values corresponding to each individual in the population
        num_parents : int
            Number of parents to select

        Returns:
        --------
        list
            Selected individuals to become parents
        """
        selected = []  # Empty list to store selected parents
        tournament_size = 3  # Number of individuals in each tournament

        # Select num_parents individuals
        for _ in range(num_parents):
            # Random selection of tournament_size individuals for the tournament
            candidates = random.sample(range(len(population)), tournament_size)

            # Select the best individual from the tournament
            # (lowest fitness since we're minimizing distance)
            best_idx = min(candidates, key=lambda i: fitnesses[i])
            selected.append(population[best_idx])

        return selected

    def survivor_selection(self, population, fitnesses):
        """
        Implements (μ + λ) survivor selection.

        This method:
        1. Combines parents and offspring
        2. Sorts them by fitness
        3. Selects the best μ individuals to survive to the next generation

        This is an elitist selection strategy that ensures the best solutions
        are never lost, which helps maintain good solutions once found.

        Parameters:
        -----------
        population : list
            Combined list of parent and offspring individuals
        fitnesses : list
            Fitness values for each individual in the population

        Returns:
        --------
        tuple
            (survivors, survivor_fitnesses) - Selected individuals and their fitness values
        """
        # Sort the population by fitness (ascending for minimization)
        # argsort returns indices that would sort the array
        sorted_indices = np.argsort(fitnesses)

        # Select the μ best individuals (lowest fitness values)
        survivors = [population[i] for i in sorted_indices[:self.mu]]
        survivor_fitnesses = [fitnesses[i] for i in sorted_indices[:self.mu]]

        return survivors, survivor_fitnesses

    def run(self):
        """
        Run the complete evolutionary algorithm.

        This method:
        1. Initializes the population
        2. Evaluates all individuals
        3. Runs the main evolutionary loop for max_generations:
           a. Selects parents
           b. Creates offspring through crossover and mutation
           c. Evaluates offspring
           d. Performs survivor selection
           e. Tracks statistics
        4. Returns the best solution found

        Returns:
        --------
        tuple
            (best_solution, best_fitness) - The best tour found and its length
        """
        # Initialize population with μ + λ individuals
        population = self.initialize_population(self.mu + self.lambda_)

        # Evaluate initial population
        fitnesses = [self.evaluate_fitness(ind) for ind in population]

        # Main evolutionary loop
        for generation in range(self.max_generations):
            # ---- Parent Selection ----
            # Select parents for reproduction using tournament selection
            parents = self.select_parents(population, fitnesses, self.lambda_)

            # ---- Variation (Recombination and Mutation) ----
            # Generate offspring through crossover and mutation
            offspring = []

            # Process parents two at a time to create pairs of offspring
            for i in range(0, self.lambda_, 2):
                if i + 1 < self.lambda_:
                    # Select two parents
                    parent1 = parents[i]
                    parent2 = parents[i + 1]

                    # Apply crossover
                    child1 = self.order_crossover(parent1, parent2)
                    child2 = self.order_crossover(parent2, parent1)

                    # Apply mutation
                    child1 = self.inversion_mutation(child1)
                    child2 = self.inversion_mutation(child2)

                    offspring.extend([child1, child2])
                else:
                    # Odd number of offspring, just create one more
                    parent1 = parents[i]
                    parent2 = random.choice(parents)  # Select another random parent

                    child = self.order_crossover(parent1, parent2)
                    child = self.inversion_mutation(child)

                    offspring.append(child)

            # ---- Evaluation ----
            # Calculate fitness for all offspring
            offspring_fitnesses = [self.evaluate_fitness(ind) for ind in offspring]

            # ---- Survivor Selection ----
            # Combine parents and offspring for (μ + λ) selection
            combined_population = population + offspring
            combined_fitnesses = fitnesses + offspring_fitnesses

            # Apply (μ + λ) survivor selection
            population, fitnesses = self.survivor_selection(combined_population, combined_fitnesses)

            # ---- Statistics and Reporting ----
            # Record statistics for this generation
            best_fitness = min(fitnesses)
            avg_fitness = sum(fitnesses) / len(fitnesses)
            self.best_fitness_history.append(best_fitness)
            self.avg_fitness_history.append(avg_fitness)

            # Print progress every 10 generations
            if generation % 10 == 0:
                print(f"Generation {generation}: Best fitness = {best_fitness}, Avg fitness = {avg_fitness}")

        # ---- Final Result ----
        # Find the best solution from the final population
        best_idx = np.argmin(fitnesses)
        best_solution = population[best_idx]
        best_fitness = fitnesses[best_idx]

        return best_solution, best_fitness

    def plot_progress(self):
        """
        Plot the evolution of fitness over generations.

        Creates a line plot showing how the best and average fitness values
        change over generations, providing a visual representation of the
        algorithm's convergence.
        """
        plt.figure(figsize=(10, 6))

        # Plot best and average fitness histories
        plt.plot(self.best_fitness_history, label='Best Fitness', color='blue', linewidth=2)
        plt.plot(self.avg_fitness_history, label='Average Fitness', color='red', linewidth=1.5, alpha=0.7)

        # Add labels and title
        plt.xlabel('Generation', fontsize=12)
        plt.ylabel('Fitness (Total Distance)', fontsize=12)
        plt.title('Evolution Progress Over Generations', fontsize=14)

        # Add legend, grid, and improve visual appearance
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()

        # Show the plot
        plt.show()

    def plot_tour(self, tour):
        """
        Plot the best tour found by the algorithm.

        Visualizes the TSP tour on a 2D plot, showing the order of cities visited.
        This provides an intuitive view of the solution quality.

        Parameters:
        -----------
        tour : list
            The tour (ordered list of cities) to visualize
        """
        # Check if the problem has node coordinates before attempting to plot
        if not hasattr(self.problem, 'node_coords'):
            print("Cannot plot tour: No node coordinates available in this TSP instance")
            return

        # Get coordinates for each city
        coords = np.array([self.problem.node_coords[i + 1] for i in range(self.dimension)])

        # Get coordinates for cities in the tour order
        tour_coords = coords[tour]

        # Add the starting city at the end to complete the tour
        tour_coords = np.vstack((tour_coords, tour_coords[0]))

        # Create the plot
        plt.figure(figsize=(10, 8))

        # Plot the tour as a path
        plt.plot(tour_coords[:, 0], tour_coords[:, 1], 'b-', marker='o', markersize=5,
                 alpha=0.7, label='Tour Path')

        # Highlight all cities
        plt.scatter(coords[:, 0], coords[:, 1], c='red', s=50, zorder=5, label='Cities')

        # Label cities with their indices for reference
        for i, (x, y) in enumerate(coords):
            plt.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points')

        # Add title and labels
        plt.title(f'TSP Tour - Total Distance: {self.evaluate_fitness(tour):.2f}', fontsize=14)
        plt.xlabel('X Coordinate', fontsize=12)
        plt.ylabel('Y Coordinate', fontsize=12)

        # Add legend and grid
        plt.legend(loc='best')
        plt.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()

        # Show the plot
        plt.show()


# Helper functions for downloading and testing TSP instances

def download_tsp_instance(instance_name):
    """
    Download a TSP instance from the TSPLIB website if not already downloaded.

    This function:
    1. Creates a directory to store TSP instances
    2. Checks if the requested instance already exists
    3. Downloads and extracts the instance file if needed
    4. Has a fallback mechanism if the primary download fails

    Parameters:
    -----------
    instance_name : str
        Name of the TSP instance (e.g., 'berlin52', 'a280')

    Returns:
    --------
    pathlib.Path
        Path to the downloaded TSP instance file
    """
    # Create a folder for the TSP instances
    data_dir = Path("tsp_instances")
    data_dir.mkdir(exist_ok=True)

    file_path = data_dir / f"{instance_name}.tsp"

    # Check if the file already exists
    if file_path.exists():
        print(f"Instance {instance_name} already downloaded.")
        return file_path

    # Base URL for TSPLIB instances
    base_url = "http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/tsp/"
    url = f"{base_url}{instance_name}.tsp.gz"

    try:
        print(f"Downloading {instance_name} from {url}...")

        # Create a temporary file for the compressed data
        temp_file = data_dir / f"{instance_name}.tsp.gz"

        # Download the compressed file
        urllib.request.urlretrieve(url, temp_file)

        # Use gzip to decompress
        import gzip
        with gzip.open(temp_file, 'rb') as f_in:
            with open(file_path, 'wb') as f_out:
                f_out.write(f_in.read())

        # Remove the temporary compressed file
        temp_file.unlink()

        print(f"Successfully downloaded and extracted {instance_name}")
        return file_path

    except Exception as e:
        print(f"Failed to download {instance_name}: {str(e)}")

        # Alternative approach: Try direct download without compression
        try:
            direct_url = f"{base_url.replace('/tsp/', '/')}{instance_name}.tsp"
            print(f"Trying direct download from {direct_url}...")
            urllib.request.urlretrieve(direct_url, file_path)
            print(f"Successfully downloaded {instance_name}")
            return file_path
        except Exception as e2:
            print(f"Alternative download also failed: {str(e2)}")
            raise


def test_parameters():
    """
    Test different parameter settings for the TSP evolutionary algorithm.

    This function:
    1. Downloads the chosen TSP instances (berlin52 and a280)
    2. Tests multiple parameter configurations for the EA
    3. Measures performance (solution quality and runtime)
    4. Visualizes results (convergence plots and tour maps)
    5. Compares the effectiveness of different parameter settings

    Parameter configurations tested include different:
    - Population sizes (μ and λ)
    - Mutation probabilities
    - Number of generations (fixed for comparison)

    Returns:
    --------
    None
        Results are printed and visualized through plots
    """
    # Download the TSP instances
    berlin52_path = download_tsp_instance("berlin52")
    a280_path = download_tsp_instance("a280")

    # Load the TSP problems
    problem1 = tsplib95.load(berlin52_path)
    problem2 = tsplib95.load(a280_path)

    # Define parameter settings to test
    # Format: (mu, lambda_, mutation_prob, max_generations)
    parameter_settings = [
        (30, 60, 0.1, 100),  # Low mutation probability
        (30, 60, 0.3, 100),  # High mutation probability
        (50, 50, 0.2, 100),  # Equal μ and λ
        (20, 100, 0.2, 100)  # Small μ, large λ
    ]

    results = []  # To store results for comparison

    # Test each parameter configuration
    for params in parameter_settings:
        mu, lambda_, mutation_prob, max_generations = params

        print(f"\n{'=' * 80}")
        print(f"Testing parameters: μ={mu}, λ={lambda_}, mutation_prob={mutation_prob}")
        print(f"{'=' * 80}")

        # Create and run EA with the current parameter configuration
        ea = TSP_EA(problem1, mu, lambda_, mutation_prob, max_generations)

        # Measure runtime
        start_time = time.time()
        best_solution, best_fitness = ea.run()
        end_time = time.time()
        runtime = end_time - start_time

        # Store results for this configuration
        results.append({
            'params': params,
            'best_fitness': best_fitness,
            'time': runtime
        })

        # Plot convergence progress
        print("\nPlotting convergence progress...")
        ea.plot_progress()

        # Plot best tour found
        print("\nPlotting best tour found...")
        ea.plot_tour(best_solution)

    # Compare results across different parameter settings
    print("\n" + "=" * 100)
    print("Parameter comparison:")
    print("=" * 100)
    print(f"{'Parameters':<30} | {'Best Distance':<15} | {'Runtime (s)':<12}")
    print("-" * 100)

    for result in results:
        params = result['params']
        param_str = f"μ={params[0]}, λ={params[1]}, mut_prob={params[2]}"
        print(f"{param_str:<30} | {result['best_fitness']:<15.2f} | {result['time']:<12.2f}")

    # Identify the best parameter setting
    best_result = min(results, key=lambda x: x['best_fitness'])
    best_params = best_result['params']

    print("\nBest parameter setting:")
    print(f"μ={best_params[0]}, λ={best_params[1]}, mutation_prob={best_params[2]}")
    print(f"Best fitness: {best_result['best_fitness']:.2f}")
    print(f"Runtime: {best_result['time']:.2f} seconds")


if __name__ == "__main__":
    """
    Main entry point for the TSP Evolutionary Algorithm program.

    This block executes when the script is run directly (not imported).
    It runs the parameter testing function to evaluate different 
    parameter configurations.
    """
    test_parameters()
import numpy as np
import matplotlib.pyplot as plt
import random
import time
import tsplib95
import requests
import os
import urllib.request
from pathlib import Path


class TSP_EA:
    def __init__(self, problem, mu, lambda_, mutation_prob, max_generations):
        """
        Initialize the Evolutionary Algorithm for TSP

        Parameters:
        - problem: tsplib95 problem instance
        - mu: Number of parents selected for next generation
        - lambda_: Number of offspring generated in each generation
        - mutation_prob: Probability of mutation
        - max_generations: Maximum number of generations
        """
        self.problem = problem
        self.mu = mu
        self.lambda_ = lambda_
        self.mutation_prob = mutation_prob
        self.max_generations = max_generations
        self.dimension = self.problem.dimension
        self.distances = self._compute_distance_matrix()
        self.best_fitness_history = []
        self.avg_fitness_history = []

    def _compute_distance_matrix(self):
        """Calculate the distance matrix for the problem"""
        n = self.dimension
        matrix = np.zeros((n, n))

        # For problems with node coordinates
        if self.problem.edge_weight_type == 'EUC_2D':
            coords = np.array([self.problem.node_coords[i + 1] for i in range(n)])
            for i in range(n):
                for j in range(n):
                    if i != j:
                        matrix[i, j] = np.sqrt(np.sum((coords[i] - coords[j]) ** 2))
        # For problems with explicit distance matrix
        elif hasattr(self.problem, 'edge_weights'):
            for i, j in self.problem.get_edges():
                matrix[i - 1, j - 1] = self.problem.get_weight(i, j)
                matrix[j - 1, i - 1] = self.problem.get_weight(i, j)  # Assuming symmetric TSP
        # For problems with edge weight sections
        elif self.problem.edge_weight_format == 'FULL_MATRIX':
            edges = list(self.problem.get_edges())
            for i, j in edges:
                matrix[i - 1, j - 1] = self.problem.get_weight(i, j)

        return matrix

    def evaluate_fitness(self, individual):
        """Calculate the total distance (fitness) of a tour"""
        total_distance = 0
        for i in range(len(individual)):
            from_city = individual[i]
            to_city = individual[(i + 1) % len(individual)]
            total_distance += self.distances[from_city, to_city]
        return total_distance

    def initialize_population(self, pop_size):
        """Create an initial population of random tours"""
        population = []
        cities = list(range(self.dimension))

        for _ in range(pop_size):
            # Create a random permutation of cities
            tour = random.sample(cities, len(cities))
            population.append(tour)

        return population

    def order_crossover(self, parent1, parent2):
        """Order Crossover (OX) operator for TSP
        Preserves the relative order of cities from both parents
        """
        size = len(parent1)
        # Choose two random crossover points
        cx_points = sorted(random.sample(range(size), 2))
        start, end = cx_points

        # Create a child with empty slots
        child = [-1] * size

        # Copy segment from parent1
        for i in range(start, end + 1):
            child[i] = parent1[i]

        # Fill the remaining slots with cities from parent2, preserving their order
        j = 0
        for i in range(size):
            # Skip positions that are already filled
            if i >= start and i <= end:
                continue

            # Find the next city in parent2 that is not yet in child
            while parent2[j] in child:
                j += 1
                if j >= size:
                    break

            if j < size:
                child[i] = parent2[j]
                j += 1

        return child

    def inversion_mutation(self, individual):
        """Inversion Mutation for TSP
        Reverses the order of a randomly selected subsequence
        """
        size = len(individual)
        if random.random() < self.mutation_prob:
            # Choose two random points
            points = sorted(random.sample(range(size), 2))
            start, end = points

            # Create a copy of the individual
            mutated = individual.copy()

            # Reverse the subsequence
            mutated[start:end + 1] = reversed(individual[start:end + 1])

            return mutated

        return individual.copy()

    def select_parents(self, population, fitnesses, num_parents):
        """Tournament selection for parents"""
        selected = []
        tournament_size = 3

        for _ in range(num_parents):
            # Select random individuals for tournament
            candidates = random.sample(range(len(population)), tournament_size)

            # Select the best individual from tournament
            best_idx = min(candidates, key=lambda i: fitnesses[i])
            selected.append(population[best_idx])

        return selected

    def survivor_selection(self, population, fitnesses):
        """(μ + λ) selection: Select the μ best individuals from the combined population"""
        # Sort the population by fitness (ascending for minimization)
        sorted_indices = np.argsort(fitnesses)

        # Select the μ best individuals
        survivors = [population[i] for i in sorted_indices[:self.mu]]
        survivor_fitnesses = [fitnesses[i] for i in sorted_indices[:self.mu]]

        return survivors, survivor_fitnesses

    def run(self):
        """Run the evolutionary algorithm"""
        # Initialize population with μ + λ individuals
        population = self.initialize_population(self.mu + self.lambda_)

        # Evaluate initial population
        fitnesses = [self.evaluate_fitness(ind) for ind in population]

        # Main evolutionary loop
        for generation in range(self.max_generations):
            # Select parents for reproduction
            parents = self.select_parents(population, fitnesses, self.lambda_)

            # Generate offspring through crossover and mutation
            offspring = []
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
                    # Odd number of offspring, just create one
                    parent1 = parents[i]
                    parent2 = random.choice(parents)

                    child = self.order_crossover(parent1, parent2)
                    child = self.inversion_mutation(child)

                    offspring.append(child)

            # Evaluate offspring
            offspring_fitnesses = [self.evaluate_fitness(ind) for ind in offspring]

            # Combine parents and offspring
            combined_population = population + offspring
            combined_fitnesses = fitnesses + offspring_fitnesses

            # Apply survivor selection (μ + λ)
            population, fitnesses = self.survivor_selection(combined_population, combined_fitnesses)

            # Record statistics
            best_fitness = min(fitnesses)
            avg_fitness = sum(fitnesses) / len(fitnesses)
            self.best_fitness_history.append(best_fitness)
            self.avg_fitness_history.append(avg_fitness)

            # Print progress
            if generation % 10 == 0:
                print(f"Generation {generation}: Best fitness = {best_fitness}, Avg fitness = {avg_fitness}")

        # Find the best solution
        best_idx = np.argmin(fitnesses)
        best_solution = population[best_idx]
        best_fitness = fitnesses[best_idx]

        return best_solution, best_fitness

    def plot_progress(self):
        """Plot the evolution of fitness over generations"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.best_fitness_history, label='Best Fitness')
        plt.plot(self.avg_fitness_history, label='Average Fitness')
        plt.xlabel('Generation')
        plt.ylabel('Fitness (Total Distance)')
        plt.title('Evolution Progress')
        plt.legend()
        plt.grid(True)
        plt.show()

    def plot_tour(self, tour):
        """Plot the best tour found"""
        if not hasattr(self.problem, 'node_coords'):
            print("Cannot plot tour: No node coordinates available")
            return

        # Get coordinates for each city in the tour
        coords = np.array([self.problem.node_coords[i + 1] for i in range(self.dimension)])
        tour_coords = coords[tour]

        # Add the starting city at the end to complete the tour
        tour_coords = np.vstack((tour_coords, tour_coords[0]))

        plt.figure(figsize=(10, 8))
        plt.plot(tour_coords[:, 0], tour_coords[:, 1], 'b-', marker='o', markersize=5)
        plt.scatter(coords[:, 0], coords[:, 1], c='red', s=50)

        # Label cities
        for i, (x, y) in enumerate(coords):
            plt.annotate(str(i), (x, y), xytext=(5, 5), textcoords='offset points')

        plt.title('TSP Tour')
        plt.xlabel('X Coordinate')
        plt.ylabel('Y Coordinate')
        plt.grid(True)
        plt.show()


# Example usage and parameter testing
def download_tsp_instance(instance_name):
    """Download a TSP instance from the TSPLIB website if not already downloaded"""
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

        # Use gzip to decompress (Python can handle this)
        import gzip
        with gzip.open(temp_file, 'rb') as f_in:
            with open(file_path, 'wb') as f_out:
                f_out.write(f_in.read())

        # Remove the temporary file
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
    # Download the TSP instances
    berlin52_path = download_tsp_instance("berlin52")
    a280_path = download_tsp_instance("a280")

    # Load the TSP problems
    problem1 = tsplib95.load(berlin52_path)
    problem2 = tsplib95.load(a280_path)

    # Test different parameter settings
    parameter_settings = [
        # (mu, lambda_, mutation_prob, max_generations)
        (30, 60, 0.1, 100),
        (30, 60, 0.3, 100),
        (50, 50, 0.2, 100),
        (20, 100, 0.2, 100)
    ]

    results = []

    for params in parameter_settings:
        mu, lambda_, mutation_prob, max_generations = params

        print(f"\nTesting parameters: μ={mu}, λ={lambda_}, mutation_prob={mutation_prob}")

        # Create and run EA
        ea = TSP_EA(problem1, mu, lambda_, mutation_prob, max_generations)
        start_time = time.time()
        best_solution, best_fitness = ea.run()
        end_time = time.time()

        results.append({
            'params': params,
            'best_fitness': best_fitness,
            'time': end_time - start_time
        })

        # Plot progress
        ea.plot_progress()

        # Plot best tour
        ea.plot_tour(best_solution)

    # Compare results
    print("\nParameter comparison:")
    for result in results:
        params = result['params']
        print(
            f"μ={params[0]}, λ={params[1]}, mutation_prob={params[2]}: Best fitness = {result['best_fitness']}, Time = {result['time']:.2f}s")


if __name__ == "__main__":
    test_parameters()
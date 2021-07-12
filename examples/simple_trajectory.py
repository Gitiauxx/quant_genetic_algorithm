import numpy as np
from source.genetic_algorithm import GA

class SimpleGA(GA):

    def fitness_function(self, individual):

        return np.sum(individual ** 2)

    def select_parents(self, fitness):

        min1 = -1
        min2 = -1

        min_value1 = np.inf
        min_value2 = np.inf

        for i, value in enumerate(fitness):
            if (value < min_value1):

                min_value2 = min_value1
                min2 = min1

                min_value1 = value
                min1 = i

            elif value < min_value2:
                min_value2 = value
                min2 = i

        return min1, min2

class SimpleTrajectory(SimpleGA):

    def fitness_function(self, individual):

        distance = 1 / self.n_genes * 1 / np.cos(individual[0])

        for i in range(1, self.n_genes):
            dist = 1 / self.n_genes * np.sqrt((np.tan(individual[i-1]) - np.tan(individual[i])) ** 2 + 1)
            distance += dist

        distance += np.sqrt((np.tan(individual[-1])) ** 2)

        return distance


if __name__ == '__main__':

    QuadGA = SimpleTrajectory(pop_size=1000, max_gen=20000, input_limits=[-np.pi/4, np.pi/4], mutation_rate=0.5, n_genes=10)
    sol, fitness = QuadGA.run()

    print(sol)
    print(fitness)
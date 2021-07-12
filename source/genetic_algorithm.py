import numpy as np
from abc import abstractmethod, ABC

class GA(ABC):

    """
    Implementation of genetic algorithm. Note that for a given iteration this implementation
    only requires O(1) operation besides the cost of selecting the fittest parents
    The implementation models population as a python list with O(1) access cost.

    It is based on a so called generational type of algorithm where crossover/mutation occurs on the
    two fittest at each iteration.
    """

    def __init__(self, pop_size=100, n_genes=2, mutation_rate=0.1, selection_rate=0.5, input_limits=[0, 1],
                 max_gen=100, stop=None):
        self.pop_size = pop_size
        self.n_genes = n_genes

        self.n_mutations = np.ceil((pop_size - 1) * n_genes * mutation_rate)
        self.pop_keep = np.floor(selection_rate * pop_size)
        self.input_limits = input_limits
        self.mutation_rate = mutation_rate
        self.max_gen = max_gen
        self.stop = stop

    @abstractmethod
    def fitness_function(self, individual):
        """
        Implements the logic that calculates the fitness
        measure of an individual.
        :param individual: chromosome of genes representing an individual
        :return: the fitness of the individual
        """

        raise NotImplementedError

    @abstractmethod
    def select_parents(self, fitness):
        """
        Selects the parents

        :param fitness: numpy array with fitness value for each individual in the population

        :return: 2 arrays with selected individuals corresponding to each parent
        """

        raise NotImplementedError

    def initialize_population(self):
        """
        Initializes the population of the problem according to the
        population size and number of genes.
        :param pop_size: number of individuals in the population
        :param n_genes: number of genes (variables) in the problem
        :param input_limits: tuple containing the minimum and maximum allowed
        :return: a numpy array with a randomly initialized population
        """

        population = [np.random.uniform(self.input_limits[0], self.input_limits[1], size=self.n_genes)
                      for _ in range(self.pop_size)]

        return population

    def crossover(self, first_parent, sec_parent):
        """
        Creates an offspring from 2 parents. It performs the crossover
        according the following rule:
        p_new = first_parent[crossover_pt] + beta * (first_parent[crossover_pt] - sec_parent[crossover_pt])
        offspring = [first_parent[:crossover_pt], p_new, sec_parent[crossover_pt + 1:]
        where beta is a random number between 0 and 1, and can be either positive or negative
        depending on if it's the first or second offspring
        :param first_parent: first parent's chromosome
        :param sec_parent: second parent's chromosome
        :param offspring_number: whether it's the first or second offspring from a pair of parents.

        :return: the resulting offspring.
        """
        crossover_pt = np.random.randint(0, high=self.n_genes)
        offspring1 = np.concatenate([first_parent[:crossover_pt], sec_parent[crossover_pt:]])
        offspring2 = np.concatenate([sec_parent[:crossover_pt], first_parent[crossover_pt:]])

        return offspring1, offspring2

    def mutate(self, individual):
        """
        Mutate an individual's i with probability self.mutation_rate
        :param population: the population at a given iteration
        :param n_mutations: number of mutations to be performed.
        :param input_limits: tuple containing the minimum and maximum allowed
         values of the problem space.

        :return: the mutated population
        """

        u = np.random.rand()
        if u < self.mutation_rate:
            mutation_genes = np.random.randint(0, high=self.n_genes)
            individual[mutation_genes] = np.random.uniform(self.input_limits[0], self.input_limits[1])

        return individual


    def calculate_fitness(self, population):
        """
        Calculates the fitness of the population
        :param population: population state at a given iteration
        :return: the fitness of the current population
        """

        return list(map(self.fitness_function, population))

    def run(self):

        """
            Performs the genetic algorithm optimization according to the
            global scope initialized parameters

            :return: (best individual, best fitness)
            """

        # initialize the population
        population = self.initialize_population()

        # Calculate the fitness of the population
        fitness = self.calculate_fitness(population)

        gen_n = 0
        while True:

            gen_n += 1

            # Get index for parents pairs: mother is the fittest
            ma, pa = self.select_parents(fitness)
            mother = population[ma]
            father = population[pa]

            # Get indices of individuals to be replaced
            to_replace = np.random.randint(0, self.pop_size, size=2)
            while len({to_replace[0], to_replace[1]}.intersection({ma, pa})) > 0:
                to_replace = np.random.randint(0, self.pop_size, size=2)

            # Get crossover point for each individual
            offspring1, offspring2 = self.crossover(mother, father)

            # Mutate offspring
            offspring1 = self.mutate(offspring1)
            offspring2 = self.mutate(offspring2)

            # update population
            population[to_replace[0]] = offspring1
            population[to_replace[1]] = offspring2

            # update fitness
            fitness_offspring1 = self.fitness_function(offspring1)
            fitness_offspring2 = self.fitness_function(offspring2)

            fitness[to_replace[0]] = fitness_offspring1
            fitness[to_replace[1]] = fitness_offspring2

            best_fitness = self.fitness_function(mother)

            print(f'Best fitness so far: {best_fitness}')

            if gen_n >= self.max_gen:
                break

            if self.stop is not None and best_fitness <= self.stop:
                break

        return mother, best_fitness
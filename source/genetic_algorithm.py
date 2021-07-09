import numpy as np
from abc import abstractmethod, ABC

class GA(ABC):

    def __init__(self, pop_size, n_genes, mutation_rate=0.1, selection_rate=0.5):
        self.pop_size = pop_size
        self.n_genes = n_genes

        self.n_mutations = np.ceil((pop_size - 1) * n_genes * mutation_rate)
        self.pop_keep = np.floor(selection_rate * pop_size)

    @abstractmethod
    def fitness_function(individual):
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

    def initialize_population(self, pop_size, n_genes, input_limits):
        """
        Initializes the population of the problem according to the
        population size and number of genes.
        :param pop_size: number of individuals in the population
        :param n_genes: number of genes (variables) in the problem
        :param input_limits: tuple containing the minimum and maximum allowed
        :return: a numpy array with a randomly initialized population
        """

        population = np.random.uniform(
            input_limits[0], input_limits[1], size=(pop_size, n_genes)
        )

        return population

    def crossover(self, first_parent, sec_parent, crossover_pt, offspring_number):
        """
        Creates an offspring from 2 parents. It performs the crossover
        according the following rule:
        p_new = first_parent[crossover_pt] + beta * (first_parent[crossover_pt] - sec_parent[crossover_pt])
        offspring = [first_parent[:crossover_pt], p_new, sec_parent[crossover_pt + 1:]
        where beta is a random number between 0 and 1, and can be either positive or negative
        depending on if it's the first or second offspring
        :param first_parent: first parent's chromosome
        :param sec_parent: second parent's chromosome
        :param crossover_pt: point(s) at which to perform the crossover
        :param offspring_number: whether it's the first or second offspring from a pair of parents.

        :return: the resulting offspring.
        """

        beta = (
            np.random.rand(1)[0]
            if offspring_number == "first"
            else -np.random.rand(1)[0]
        )

        p_new = first_parent[crossover_pt] - beta * (
                first_parent[crossover_pt] - sec_parent[crossover_pt]
        )

        return np.hstack(
            (first_parent[:crossover_pt], p_new, sec_parent[crossover_pt + 1:])
        )

    def mutate_population(self, population, n_mutations, input_limits):
        """
        Mutates the population by randomizing specific positions of the
        population individuals.
        :param population: the population at a given iteration
        :param n_mutations: number of mutations to be performed.
        :param input_limits: tuple containing the minimum and maximum allowed
         values of the problem space.

        :return: the mutated population
        """

        mutation_rows = np.random.choice(
            np.arange(1, population.shape[0]), n_mutations, replace=True
        )

        mutation_columns = np.random.choice(
            population.shape[1], n_mutations, replace=True
        )

        new_population = np.random.uniform(
            input_limits[0], input_limits[1], size=population.shape
        )

        population[mutation_rows, mutation_columns] = new_population[mutation_rows, mutation_columns]

        return population

    def calculate_fitness(self, population):
        """
        Calculates the fitness of the population
        :param population: population state at a given iteration
        :return: the fitness of the current population
        """

        return np.array(list(map(self.fitness_function, population)))

    def run(self):

        """
            Performs the genetic algorithm optimization according to the
            global scope initialized parameters

            :return: (best individual, best fitness)
            """

        # initialize the population
        population = self.initialize_population(self.pop_size, self.n_genes)

        # Calculate the fitness of the population
        fitness = self.calculate_fitness(population)

        gen_n = 0
        while True:

            gen_n += 1

            # Get index for parents pairs: mother is the fittest
            ma, pa = self.select_parents(fitness)
            mother = population[ma, :]
            father = population[pa, :]

            # Get indices of individuals to be replaced
            ix = np.arange(0, self.pop_size - self.pop_keep - 1, 2)

            # Get crossover point for each individual
            xp = np.random.randint(0, self.n_genes, size=(self.pop_size, 1))

            for i in range(xp.shape[0]):
                # create first offspring
                population[-1 - ix[i], :] = self.crossover(
                    population[ma[i], :], population[pa[i], :], xp[i], "first"
                )

                # create second offspring
                population[-1 - ix[i] - 1, :] = self.crossover(
                    population[pa[i], :], population[ma[i], :], xp[i], "second"
                )

            population = self.mutate_population(population, self.n_mutations, self.input_limits)

            # Get new population's fitness. Since the fittest element does not change,
            # we do not need to re calculate its fitness
            fitness = np.hstack((fitness[0], self.calculate_fitness(population[1:, :])))

            fitness, population = self.sort_by_fitness(fitness, population)

            if gen_n >= self.max_gen:
                break

            return population[0], fitness[0]


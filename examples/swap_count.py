import random
import argparse
import numpy as np
import sympy as sp
import datetime
import json
import os

from source.genetic_algorithm import GA


class Circuit(object):

    def __init__(self, depth, numAncillas, numInputs):

        a, b, c, d, e, f, g, h, z, y, u, t, l, m, n, o, p, s = sp.symbols('a,b,c,d,e,f,g,h,z,y,u,t, l, m, n, o, p, s')  # these variables correspond to inputs like phi1, phi2 etc.
        self.varList = [a, b, c, d, e, f, g, h, z, y, u, t, l, m, n, o, p, s]
        self.initialQuantumState = [[0] * numAncillas + self.varList[:numInputs]]
        self.initialQuantumState[0].append(1)

        self.numAncillas = numAncillas
        self.numQubits = numAncillas + numInputs
        self.numInputs = numInputs
        self.depth =depth

    def simplify(self, quantumState):
        """
        takes in a quantum state and returns a condensed version that makes sure
        any kets in the same state are combined
        :param quantumState:
        :return:
        """
        finalState = []
        condensed = []

        for i in range(len(quantumState)):
            currKet = []
            for j in range(len(quantumState[i]) - 1):
                currKet.append(quantumState[i][j])
            if currKet in condensed:
                continue
            else:
                coeff = quantumState[i][self.numAncillas + self.numInputs]
                for j in range(i + 1, len(quantumState)):
                    activate = True
                    for k in range(len(quantumState[j]) - 1):
                        if currKet[k] != quantumState[j][k]:
                            activate = False
                            break
                    if activate:
                        coeff += quantumState[j][self.numAncillas + self.numInputs]
                copy = currKet.copy()
                condensed.append(copy)
                currKet.append(coeff)
                finalState.append(currKet)
        return finalState

    def HAD(self, qubit):
        """
        # HAD - returns a hadamard gate: list with first element as 'h' to identify it as a H gate
        and the second element is the qubit upon which it is acting
        :return:
        """
        h = ['h']
        h.append(qubit)
        return h

    def CSWAP(self, control, targets):
        """
        returns a CSWAP gate: list with first element as 'cswap' to identify it as a CSWAP gate,
        the second element is the qubit number which is the control, and third element is a list
        of the two qubit numbers which are being swapped
        :param targets:
        :return:
        """
        c = ['cswap']
        c.append(control)
        c.append(targets)
        return c

    def hadamard(self, qubit, quantumState):
        """
        returns updated quantum state after applying hadamard gate to it
        :param qubit:
        :param quantumState:
        :return:
        """
        newQuantumState = quantumState.copy()
        for i in range(len(quantumState)):  # Goes through all the states in the quantum state
            if quantumState[i][
                qubit] == 0:  # if the qubit being acted upon is zero, it will create a new state with 1 as well and add that to the list
                newState = quantumState[i].copy()
                newState[qubit] = 1
                newQuantumState.append(newState)
            else:  # if the qubit acted upon is one, then it replaces the current state with a negative version, and creates a new state with 0 replacing the one
                newQuantumState[i][self.numQubits] = -1 * quantumState[i][self.numQubits]

                newState = quantumState[i].copy()
                newState[qubit] = 0
                newState[self.numQubits] = 1
                newQuantumState.append(newState)
        return newQuantumState

    def cswap(self, control, targets, quantumState):
        """
        returns updated quantum state after applying cswap gate to it
        :param control:
        :param targets:
        :param quantumState:
        :return:
        """
        newQuantumState = quantumState.copy()
        for i in range(len(quantumState)):
            if newQuantumState[i][control] == 0:
                continue
            else:
                firstTarget = newQuantumState[i][targets[0]]
                secondTarget = newQuantumState[i][targets[1]]
                newQuantumState[i][targets[0]] = secondTarget
                newQuantumState[i][targets[1]] = firstTarget
        return newQuantumState

    def run_circuit(self, circuit):
        """
        takes in a circuit and a quantum state upon which the circuit will act and
        returns the quantum state the circuit produces
        :type circuit: object
        :return:
        """

        quantumState = self.initialQuantumState

        for i in range(len(circuit)):
            if circuit[i][0] == 'h':
                quantumState = self.hadamard(circuit[i][1], quantumState)
            elif circuit[i][0] == 'cswap':
                quantumState = self.cswap(circuit[i][1], circuit[i][2], quantumState)

        quantumState = self.simplify(quantumState)
        return quantumState

    def randomCircuit(self,):
        """
        returns a circuit with random gates
        :return:
        """
        circuit = []

        # hadamard zone
        for i in range(self.numAncillas):
            circuit.append(self.HAD(i))

        # cswap zone
        for i in range(int(self.depth)):
            control = random.randint(0, self.numAncillas - 1)
            targets = [random.randint(self.numAncillas, self.numAncillas + self.numInputs - 1)]
            t2 = random.randint(self.numAncillas, self.numAncillas + self.numInputs - 1)
            while t2 == targets[0]:
                t2 = random.randint(self.numAncillas, self.numAncillas + self.numInputs - 1)
            targets.append(t2)
            circuit.append(self.CSWAP(control, targets))

        return circuit

    def count_pair_two_registers_i_j(self, quantum_state, i, j):
        """
        Count number of pairs that appear on input i and j in quantun_state. Order does not matter
        :param quantum_state:
        :param i:
        :param j:
        :return:
        """
        register_i_j = [{q[i], q[j]} for q in quantum_state]
        return len({frozenset(el) for el in register_i_j})


    def count_pair_two_register(self, quantum_state):
        """
        Count number of pairs that appear on input 0 and 1 in quantun_state . Order does not matter
        :param quantum_state:
        :return:
        """
        return self.count_pair_two_registers_i_j(quantum_state, self.numAncillas, self.numAncillas + 1)


class CircuitGA(GA):

    def __init__(self, depth, num_ancillas, pop_size=100, n_genes=2, mutation_rate=0.1, max_gen=100, stop=0):

        super().__init__(pop_size=pop_size, n_genes=n_genes, mutation_rate=mutation_rate, max_gen=max_gen, stop=stop)
        self.circuit = Circuit(depth, num_ancillas, n_genes - num_ancillas)

    def fitness_function(self, individual):
        quantum_state = self.circuit.run_circuit(individual)

        return self.circuit.numInputs * (self.circuit.numInputs - 1) / 2 - \
               self.circuit.count_pair_two_register(quantum_state)

    def select_parents(self, fitness):

        min1 = -1
        min2 = -1

        min_value1 = self.circuit.numInputs * (self.circuit.numInputs - 1) / 2
        min_value2 = self.circuit.numInputs * (self.circuit.numInputs - 1) / 2

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

    def initialize_population(self):
        population = [self.circuit.randomCircuit() for _ in range(self.pop_size)]

        return population

    def mutate(self, individual):
        index = np.random.randint(self.circuit.numAncillas, len(individual))
        control = np.random.randint(0, self.circuit.numAncillas)
        swap1 = np.random.randint(self.circuit.numAncillas, self.circuit.numQubits)
        swap2 = np.random.randint(self.circuit.numAncillas, self.circuit.numQubits)

        gate = self.circuit.CSWAP(control, [swap1, swap2])

        individual[index] = gate

        return individual

    def crossover(self, first_parent, sec_parent):
        crossover_pt = np.random.randint(self.circuit.numAncillas, high=self.circuit.numQubits)
        offspring1 = first_parent[:crossover_pt] + sec_parent[crossover_pt:]
        offspring2 = sec_parent[:crossover_pt] + first_parent[crossover_pt:]

        return offspring1, offspring2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_inputs', type=int, default=16)
    parser.add_argument('--num_ancillas', type=int, default=9)
    parser.add_argument('--depth', type=int, default=21)
    parser.add_argument('--mutation_rate', type=int, default=0.5)
    parser.add_argument('--num_iterations', type=int, default=5000)
    parser.add_argument('--gen_size', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--run', default='test')

    args = parser.parse_args()


    depth = args.depth
    num_ancillas = args.num_ancillas
    num_inputs = args.num_inputs
    mutation_rate = args.mutation_rate
    n_genes = num_ancillas + num_inputs

    np.random.seed(args.seed)

    ca = CircuitGA(depth,
                   num_ancillas,
                   pop_size=args.gen_size,
                   n_genes=n_genes,
                   mutation_rate=mutation_rate,
                   max_gen=args.num_iterations,
                   stop=0)
    circuit, fitness = ca.run()

    print(f'solution is: {fitness}')

    tstamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    save_dir = '/scratch/xgitiaux/quantdist/genetic_search'
    save_dir = f'{save_dir}/run_{args.run}_depth_{args.depth}_inputs_{args.num_inputs}_ancillas_{args.num_ancillas}_gensize_{args.gen_size}'
    os.makedirs(save_dir, exist_ok=True)

    results = {'succes': fitness == 0, 'circuit': circuit, 'cost': fitness}
    rind = np.random.randint(0, 2**16)
    print(rind)
    with open(f'{save_dir}/results_{tstamp}_{rind}.json', 'w') as file:
        json.dump(results, file)

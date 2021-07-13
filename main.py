import random
import argparse
import numpy as np
import sympy as sp
import datetime
import json
import os

from examples.swap_count import CircuitGA

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

    tstamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    save_dir = '/scratch/xgitiaux/quantdist/genetic_search'
    save_dir = f'{save_dir}/run_{args.run}_depth_{args.depth}_inputs_{args.num_inputs}_ancillas_{args.num_ancillas}_gensize_{args.gen_size}'
    os.makedirs(save_dir, exist_ok=True)

    results = {'succes': fitness == 0, 'circuit': circuit}
    with open(f'{save_dir}/results_{tstamp}.json', 'w') as file:
        json.dump(results, file)
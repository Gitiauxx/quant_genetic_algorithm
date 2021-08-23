#!/bin/bash

#SBATCH --job-name=gen_algo
#SBATCH --output=/scratch/xgitiaux/gen_algo_%j_%a.out
#SBATCH --error=/scratch/xgitiaux/gen_algo_%j_%a.error
#SBATCH --mail-user=xgitiaux@gmu.edu
#SBATCH --mail-type=END
#SBATCH --export=ALL
#SBATCH --partition=all-LoPri
#SBATCH --nodes 1
#SBATCH --tasks 1
#SBATCH --mem=64G
#SBATCH --qos=csqos
#SBATCH --array=0-100

source ../fvae-env/bin/activate

echo $SLURM_ARRAY_TASK_ID
../fvae-env/bin/python3 main.py --depth 8 --num_ancillas 6 --seed $SLURM_ARRAY_TASK_ID --num_iterations 10000 --run run22  --num_inputs 8 --gen_size 100000

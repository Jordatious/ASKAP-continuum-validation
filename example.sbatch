#!/bin/bash -l
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00
#SBATCH --job-name=continuum_validation
#SBATCH --cluster=galaxy
#SBATCH --export=NONE

module use /group/askap/continuum_validation
module load continuum_validation_env

aprun -n 1 -N 1 $ACES/UserScripts/col52r/ASKAP_continuum_validation.py -v gama_linmos.fits
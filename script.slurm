#!/bin/bash
#SBATCH --job-name=test # nom du job
##SBATCH --partition=gpu_p2 # specify partition
##SBATCH -C v100-16g # only GPUs with 16 Go of memory
#SBATCH -C v100-32g # only GPUs with 32 Go of memory
##SBATCH --qos=qos_gpu-t3 # QoS
#SBATCH --output=out_slurm/ouz%j.out # fichier de sortie (%j = job ID)
#SBATCH --error=out_slurm/ouz%j.err # fichier d’erreur (%j = job ID)
#SBATCH --time=20:00:00 # temps maximal d’allocation "(HH:MM:SS)"
#SBATCH --nodes=1 # reserver 1 noeud
#SBATCH --ntasks=1 # reserver 8 taches
#SBATCH --gres=gpu:1 # reserver 8 GPU
#SBATCH --cpus-per-task=10 # reserver 3 CPU par tache (et memoire associee)
#SBATCH --hint=nomultithread # desactiver l’hyperthreading
#SBATCH --account=irm@v100

#SBATCH --array=0



module purge # nettoyer les modules herites par defaut
module load pytorch-gpu/py3/2.8.0 # charger les modules

set -x # activer l’echo des commandes
srun python main.py  --n=$SLURM_ARRAY_TASK_ID # executer son script

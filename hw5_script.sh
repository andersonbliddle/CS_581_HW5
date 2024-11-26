#!/bin/bash
module load cuda



echo "Serialized Code Run:\n"

./hw5_init 5000 5000 16 /scratch/ualclsd0173/ 1


./hw5 1000 1000 16 /scratch/ualclsd0173/ 1
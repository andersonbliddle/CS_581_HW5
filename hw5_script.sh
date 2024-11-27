#!/bin/bash
source /opt/asn/etc/asn-bash-profiles-special/modules.sh
module load cuda/11.7.0

echo "GPU 5000 Code Run 1:"
./hw5 5000 5000 1 /scratch/ualclsd0173/1/ 1
./hw5 5000 5000 2 /scratch/ualclsd0173/2/ 1
./hw5 5000 5000 4 /scratch/ualclsd0173/4/ 1
./hw5 5000 5000 8 /scratch/ualclsd0173/8/ 1
./hw5 5000 5000 10 /scratch/ualclsd0173/10/ 1
./hw5 5000 5000 16 /scratch/ualclsd0173/16/ 1
./hw5 5000 5000 20 /scratch/ualclsd0173/20/ 1
./hw5 5000 5000 32 /scratch/ualclsd0173/32/ 1
./hw5 5000 5000 64 /scratch/ualclsd0173/64/ 1

echo "GPU 5000 Code Run 2:"
./hw5 5000 5000 1 /scratch/ualclsd0173/1/ 1
./hw5 5000 5000 2 /scratch/ualclsd0173/2/ 1
./hw5 5000 5000 4 /scratch/ualclsd0173/4/ 1
./hw5 5000 5000 8 /scratch/ualclsd0173/8/ 1
./hw5 5000 5000 10 /scratch/ualclsd0173/10/ 1
./hw5 5000 5000 16 /scratch/ualclsd0173/16/ 1
./hw5 5000 5000 20 /scratch/ualclsd0173/20/ 1
./hw5 5000 5000 32 /scratch/ualclsd0173/32/ 1
./hw5 5000 5000 64 /scratch/ualclsd0173/64/ 1

echo "GPU 5000 Code Run 3:"
./hw5 5000 5000 1 /scratch/ualclsd0173/1/ 1
./hw5 5000 5000 2 /scratch/ualclsd0173/2/ 1
./hw5 5000 5000 4 /scratch/ualclsd0173/4/ 1
./hw5 5000 5000 8 /scratch/ualclsd0173/8/ 1
./hw5 5000 5000 10 /scratch/ualclsd0173/10/ 1
./hw5 5000 5000 16 /scratch/ualclsd0173/16/ 1
./hw5 5000 5000 20 /scratch/ualclsd0173/20/ 1
./hw5 5000 5000 32 /scratch/ualclsd0173/32/ 1
./hw5 5000 5000 64 /scratch/ualclsd0173/64/ 1

echo "GPU 10,000 x 5000 Code Run 1:"
./hw5 10000 5000 1 /scratch/ualclsd0173/1/ 1
./hw5 10000 5000 2 /scratch/ualclsd0173/2/ 1
./hw5 10000 5000 4 /scratch/ualclsd0173/4/ 1
./hw5 10000 5000 8 /scratch/ualclsd0173/8/ 1
./hw5 10000 5000 10 /scratch/ualclsd0173/10/ 1
./hw5 10000 5000 16 /scratch/ualclsd0173/16/ 1
./hw5 10000 5000 20 /scratch/ualclsd0173/20/ 1
./hw5 10000 5000 32 /scratch/ualclsd0173/32/ 1
./hw5 10000 5000 64 /scratch/ualclsd0173/64/ 1

echo "GPU 10,000 x 5000 Code Run 2:"
./hw5 10000 5000 1 /scratch/ualclsd0173/1/ 1
./hw5 10000 5000 2 /scratch/ualclsd0173/2/ 1
./hw5 10000 5000 4 /scratch/ualclsd0173/4/ 1
./hw5 10000 5000 8 /scratch/ualclsd0173/8/ 1
./hw5 10000 5000 10 /scratch/ualclsd0173/10/ 1
./hw5 10000 5000 16 /scratch/ualclsd0173/16/ 1
./hw5 10000 5000 20 /scratch/ualclsd0173/20/ 1
./hw5 10000 5000 32 /scratch/ualclsd0173/32/ 1
./hw5 10000 5000 64 /scratch/ualclsd0173/64/ 1

echo "GPU 10,000 x 5000 Code Run 3:"
./hw5 10000 5000 1 /scratch/ualclsd0173/1/ 1
./hw5 10000 5000 2 /scratch/ualclsd0173/2/ 1
./hw5 10000 5000 4 /scratch/ualclsd0173/4/ 1
./hw5 10000 5000 8 /scratch/ualclsd0173/8/ 1
./hw5 10000 5000 10 /scratch/ualclsd0173/10/ 1
./hw5 10000 5000 16 /scratch/ualclsd0173/16/ 1
./hw5 10000 5000 20 /scratch/ualclsd0173/20/ 1
./hw5 10000 5000 32 /scratch/ualclsd0173/32/ 1
./hw5 10000 5000 64 /scratch/ualclsd0173/64/ 1

echo "GPU 10,000 x 10,000 Code Run 1:"
./hw5 10000 10000 1 /scratch/ualclsd0173/1/ 1
./hw5 10000 10000 2 /scratch/ualclsd0173/2/ 1
./hw5 10000 10000 4 /scratch/ualclsd0173/4/ 1
./hw5 10000 10000 8 /scratch/ualclsd0173/8/ 1
./hw5 10000 10000 10 /scratch/ualclsd0173/10/ 1
./hw5 10000 10000 16 /scratch/ualclsd0173/16/ 1
./hw5 10000 10000 20 /scratch/ualclsd0173/20/ 1
./hw5 10000 10000 32 /scratch/ualclsd0173/32/ 1
./hw5 10000 10000 64 /scratch/ualclsd0173/64/ 1

echo "GPU 10,000 x 10,000 Code Run 2:"
./hw5 10000 10000 1 /scratch/ualclsd0173/1/ 1
./hw5 10000 10000 2 /scratch/ualclsd0173/2/ 1
./hw5 10000 10000 4 /scratch/ualclsd0173/4/ 1
./hw5 10000 10000 8 /scratch/ualclsd0173/8/ 1
./hw5 10000 10000 10 /scratch/ualclsd0173/10/ 1
./hw5 10000 10000 16 /scratch/ualclsd0173/16/ 1
./hw5 10000 10000 20 /scratch/ualclsd0173/20/ 1
./hw5 10000 10000 32 /scratch/ualclsd0173/32/ 1
./hw5 10000 10000 64 /scratch/ualclsd0173/64/ 1

echo "GPU 10,000 x 10,000 Code Run 3:"
./hw5 10000 10000 1 /scratch/ualclsd0173/1/ 1
./hw5 10000 10000 2 /scratch/ualclsd0173/2/ 1
./hw5 10000 10000 4 /scratch/ualclsd0173/4/ 1
./hw5 10000 10000 8 /scratch/ualclsd0173/8/ 1
./hw5 10000 10000 10 /scratch/ualclsd0173/10/ 1
./hw5 10000 10000 16 /scratch/ualclsd0173/16/ 1
./hw5 10000 10000 20 /scratch/ualclsd0173/20/ 1
./hw5 10000 10000 32 /scratch/ualclsd0173/32/ 1
./hw5 10000 10000 64 /scratch/ualclsd0173/64/ 1

echo "Serialized Code Runs:"

./hw5_init 5000 5000 16 /scratch/ualclsd0173/validation_grid/ 1
./hw5_init 10000 5000 16 /scratch/ualclsd0173/validation_grid/ 1

echo "Running diffs between serial and GPU 5000 runs..."
diff /scratch/ualclsd0173/validation_grid/output5000_5000_16.txt /scratch/ualclsd0173/1/output5000_5000_1.txt
diff /scratch/ualclsd0173/validation_grid/output5000_5000_16.txt /scratch/ualclsd0173/2/output5000_5000_2.txt
diff /scratch/ualclsd0173/validation_grid/output5000_5000_16.txt /scratch/ualclsd0173/4/output5000_5000_4.txt
diff /scratch/ualclsd0173/validation_grid/output5000_5000_16.txt /scratch/ualclsd0173/8/output5000_5000_8.txt
diff /scratch/ualclsd0173/validation_grid/output5000_5000_16.txt /scratch/ualclsd0173/10/output5000_5000_10.txt
diff /scratch/ualclsd0173/validation_grid/output5000_5000_16.txt /scratch/ualclsd0173/16/output5000_5000_16.txt
diff /scratch/ualclsd0173/validation_grid/output5000_5000_16.txt /scratch/ualclsd0173/20/output5000_5000_20.txt
diff /scratch/ualclsd0173/validation_grid/output5000_5000_16.txt /scratch/ualclsd0173/32/output5000_5000_32.txt
diff /scratch/ualclsd0173/validation_grid/output5000_5000_16_init.txt /scratch/ualclsd0173/64/output5000_5000_64.txt

echo "Running diffs between serial and GPU 10,0000 runs..."
diff /scratch/ualclsd0173/validation_grid/output10000_5000_16.txt /scratch/ualclsd0173/2/output10000_5000_2.txt
diff /scratch/ualclsd0173/validation_grid/output10000_5000_16.txt /scratch/ualclsd0173/4/output10000_5000_4.txt
diff /scratch/ualclsd0173/validation_grid/output10000_5000_16.txt /scratch/ualclsd0173/8/output10000_5000_8.txt
diff /scratch/ualclsd0173/validation_grid/output10000_5000_16.txt /scratch/ualclsd0173/10/output10000_5000_10.txt
diff /scratch/ualclsd0173/validation_grid/output10000_5000_16.txt /scratch/ualclsd0173/16/output10000_5000_16.txt
diff /scratch/ualclsd0173/validation_grid/output10000_5000_16.txt /scratch/ualclsd0173/20/output10000_5000_20.txt
diff /scratch/ualclsd0173/validation_grid/output10000_5000_16.txt /scratch/ualclsd0173/32/output10000_5000_32.txt
diff /scratch/ualclsd0173/validation_grid/output10000_5000_16.txt /scratch/ualclsd0173/64/output10000_5000_64.txt

echo "If you saw nothing, then there were no differences!"
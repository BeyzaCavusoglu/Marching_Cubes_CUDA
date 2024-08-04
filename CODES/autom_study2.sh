#!/bin/bash

# Load necessary modules
module load gcc/9.3.0
module load cuda/11.8.0

# Compile the CUDA code
nvcc -arch=compute_35 -code=sm_35 main.cu -o mcubes -lcudart

# Define the output file
output_file="mcubes_study2_results.txt"

# Function to run mcubes with specified arguments and redirect output
run_mcubes() {
    echo "Running mcubes with $1 threads $2 blocks ..." >> "$output_file"
    ./mcubes $3 -t $1 -b $2 >> "$output_file"  # Fixed the order of arguments
    echo "--------------------------------------------"
}

# Specify configurations
configurations=(
    "-n 64 -f 1"
    "-n 4 -f 32768"
)

# Array of thread values
thread_values=(1024)
block_values=(1 10 20 40 80 160)

# Run mcubes for each configuration and thread value
for config in "${configurations[@]}"; do
    for threadsNum in "${thread_values[@]}"; do
        for blockNums in "${block_values[@]}"; do
            run_mcubes $threadsNum $blockNums "$config"
        done
    done
done

# Print a message indicating the completion of the script
echo "Results are saved in $output_file"

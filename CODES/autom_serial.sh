#!/bin/bash

# Load necessary modules
module load gcc/9.3.0
module load cuda/11.8.0

# Compile the CUDA code
nvcc -arch=compute_35 -code=sm_35 main.cu -o mcubes -lcudart

# Define the output file
output_file="mcubes_serial_results.txt"

# Function to run mcubes with specified arguments and redirect output
run_mcubes() {
    echo "Running mcubes with $1 ..." >> "$output_file"
    ./mcubes $2 >> "$output_file"
    echo "--------------------------------------------"
}

# Specify configurations
configurations=(
    "-n 64 -f 1 -b 1 -o"
    "-n 8 -f 64 -b 1 -o"
    "-n 4 -f 32768 -b 1 -o"
)


# Run mcubes for each configuration and thread value
for config in "${configurations[@]}"; do
        run_mcubes "$config"
done

# Print a message indicating the completion of the script
echo "Results are saved in $output_file"

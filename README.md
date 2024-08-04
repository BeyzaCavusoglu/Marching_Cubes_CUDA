# Marching_Cubes_CUDA
Parallel version of the Marching Cubes algorithm in CUDA

## Description
Marching Cubes is a simple algorithm for creating a triangle mesh from a uniform grid of cubes superimposed over a region of the function. Each vertex in the mesh defines whether it’s the inside or the outside of a shape that is being represented. If all 8 vertices of the cube are positive, or all 8 vertices are negative, the cube is entirely above or entirely below the surface and no triangles are emitted. Otherwise, the cube straddles the function and some triangles and vertices are generated. Since each vertex can either be positive or negative, there are technically 28 possible configurations, but many of these are equivalent to one another. There are only 15 unique cases.

<img width="312" alt="asdaasd" src="https://github.com/user-attachments/assets/1156cebf-7ccc-4795-910f-2db106a0a43d">

## Part 1: Single thread with CUDA

The provided serial code is used and run on the GPU, making sure that properly allocated the data and made CUDA API calls. The performance comparison of running a single GPU thread and a single CPU thread is provided in report. 

## Part 2. Parallel CUDA execution

The individual cubes are splitted between the threads and blocks. The indices of the cubes that belong to each thread are properly calculated and made sure that they are not overloading GPU shared memory. The state of the device memory before each kernel launch is all zeroes and the result is copied to the host afterwards. Just enough global memory is allocated for a single frame. Experiment with different block and thread counts are provided in the report.

## Part 3. Inter-Frame + Intra-Frame Parallelization

In Part 2 the implementation was a so-called discrete kernel approach, with a separate kernel launch per iteration (frame). Now, more memory allocated and calculated the result for each frame in a single kernel launch. This approach with the time loop moved inside the kernel is called persistent kernel. All the available threads were used. 

## Part 4. Double buffer

The overall running time of a specific problem is tried to be improved. Kernel execution and data movement are overlapped to reduce the overheads. The data is send for frame
i-1 while computing the frame i. CUDA Events and Streams is used. 

## Experiments

Performance data gathered across different numbers of threads with different mesh sizes (resolution) and number of frames. 

• The first experiment is a strong scaling study that runs the parallelized code multiple times under different thread counts and a single thread block.
<threadsNum> are: 1, 4, 16, 32, 64, 128, 256, 512, 1024. 

• Performance under different block counts, but with fixed thread count. Thread count that worked best for
is used.
<blockNum> are: 1, 10, 20, 40, 80, 160.

For more details, please refer to "ToDo.pdf" file.

*Note: This project is done for a course project, the serialized version of this code was already designed and given by the instructor. My implementation was to parallelize them using OPENMP and conducting the performance tests.*

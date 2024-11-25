#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <time.h>
#include <sys/time.h>

// Kernel to compute the next generation
__global__ void next_generation(int *grid, int *new_grid, int rows, int cols) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= 1 && i < rows - 1 && j >= 1 && j < cols - 1) {
        int neighbors = grid[(i - 1) * cols + (j - 1)] +
                        grid[(i - 1) * cols + j] +
                        grid[(i - 1) * cols + (j + 1)] +
                        grid[i * cols + (j - 1)] +
                        grid[i * cols + (j + 1)] +
                        grid[(i + 1) * cols + (j - 1)] +
                        grid[(i + 1) * cols + j] +
                        grid[(i + 1) * cols + (j + 1)];

        if (neighbors <= 1 || neighbors >= 4)
            new_grid[i * cols + j] = 0;  // Dies
        else if (neighbors == 3)
            new_grid[i * cols + j] = 1;  // Born
        else
            new_grid[i * cols + j] = grid[i * cols + j];  // Stays the same
    }
}

// Function to initialize the grid with random values
void initialize_grid(int *grid, int rows, int cols) {
    srand(42);  // Fixed seed for reproducibility
    for (int i = 1; i < rows - 1; i++) {
        for (int j = 1; j < cols - 1; j++) {
            grid[i * cols + j] = rand() % 2;
        }
    }
}

// Function to write the final grid to a file
void outputtofile(char *output_file, int* grid, int rows, int cols){
    FILE *file = fopen(output_file, "w");
    if (file == NULL) {
        printf("Error: Unable to open file %s for writing.\n", output_file);
        return;
    }

    for (int i = 1; i < rows - 1; i++) {  // Exclude ghost rows
        for (int j = 1; j < cols - 1; j++) {  // Exclude ghost columns
            fprintf(file, "%d ", grid[i * cols + j]);
        }
        fprintf(file, "\n");
    }

    fclose(file);
}

// Function to get the current time in seconds
double get_time() {
    struct timeval tval;
    gettimeofday(&tval, NULL);
    return (double)tval.tv_sec + (double)tval.tv_usec / 1000000.0;
}

// Main function
int main(int argc, char **argv) {
    if (argc != 6) {
        printf("Usage: %s <dimensions> <max_generations> <block_size> <stagnationcheck> <output_file>\n", argv[0]);
        return -1;
    }

    // Parse command line arguments
    int dimensions = atoi(argv[1]);
    int max_generations = atoi(argv[2]);
    int block_size = atoi(argv[3]);
    int stagnationcheck = atoi(argv[4]);
    // Output file and directory (format output_N_N_gen_threads.txt)
    char output_file[200];
    sprintf(output_file, "%s/output%s_%s_%s.txt", argv[5], argv[1], argv[2], argv[3]);

    int rows = dimensions + 2;  // Adding ghost rows
    int cols = dimensions + 2;

    size_t grid_size = rows * cols * sizeof(int);

    // Allocate memory for grids on host
    int *host_grid = (int *)malloc(grid_size);
    int *host_new_grid = (int *)malloc(grid_size);

    // Initialize the grid
    initialize_grid(host_grid, rows, cols);

    // Allocate memory for grids on device
    int *dev_grid, *dev_new_grid;
    cudaMalloc((void **)&dev_grid, grid_size);
    cudaMalloc((void **)&dev_new_grid, grid_size);

    // Copy initial grid to device
    cudaMemcpy(dev_grid, host_grid, grid_size, cudaMemcpyHostToDevice);

    // Set up block and grid dimensions
    dim3 block_dim(block_size, block_size);
    dim3 grid_dim((cols + block_size - 1) / block_size, (rows + block_size - 1) / block_size);

    // Start timing
    double start_time = get_time();

    // Main simulation loop
    for (int gen = 0; gen < max_generations; gen++) {
        next_generation<<<grid_dim, block_dim>>>(dev_grid, dev_new_grid, rows, cols);

        // Swap grids
        int *temp = dev_grid;
        dev_grid = dev_new_grid;
        dev_new_grid = temp;

        // Optional: Check for stagnation (if enabled)
        if (stagnationcheck) {
            // Add stagnation check logic here if required.
        }
    }

    // End timing
    double end_time = get_time();
    printf("Simulation completed in %.6f seconds\n", end_time - start_time);

    // Copy final grid back to host
    cudaMemcpy(host_grid, dev_grid, grid_size, cudaMemcpyDeviceToHost);

    // Write the final grid to the output file
    outputtofile(output_file, host_grid, rows, cols);

    // Free memory on device
    cudaFree(dev_grid);
    cudaFree(dev_new_grid);

    // Free memory on host
    free(host_grid);
    free(host_new_grid);

    return 0;
}
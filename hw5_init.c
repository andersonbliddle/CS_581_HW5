/*
 Name: Anderson B. Liddle
 Email: abliddle@crimson.ua.edu
 Course Section: CS 581
 Homework # 3
 Instructions to compile the program: gcc -O hw3.c -o hw3 -O3
 Instructions to run the program: ./hw3 <dimensions (int)>
                                    <max_generations (int)>
                                    <num_threads (int)>
                                    <output directory (string)>
                                    <stagnationcheck (boolean 1 or 0)>
 Please use this format for testing: ./hw3 1000 1000 16 . 1
 GITHUB LINK - https://github.com/andersonbliddle/CS_581_HW3 
*/
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <string.h>


// Dynamically allocate a 2D array of integers
// Based on Geeks for Geeks article. Find link in References section of Report
int **allocarray(int row, int col) {
  int i;
  
  int** arr = (int**)malloc(row * sizeof(int*));
  for (i = 0; i < row; i++)
      arr[i] = (int*)malloc(col * sizeof(int));

  return arr;
}

// Free allocated memory of arrays
void destroyarray(int** array, int rows){
    for (int i = 0; i < rows; i++)
        free(array[i]);
    free(array);
    return;
}


// Function based on code provided in matmul.c
// Sets all values of matrix to 0
int **initarray(int **a, int rows, int cols) {
  int i,j;

  for (i = 0; i < rows; i++)
    for (j = 0; j < cols; j++)
      a[i][j] = 0;
  
  return a;
}

// Function based on code provided in matmul.c
// Prints the array in a formatted manner
void printarray(int **array, int rows, int cols) {
  int i,j;
  
  for (i = 0; i < rows; i++) {
    for (j = 0; j < cols; j++)
      if (array[i][j]){
        printf("%i ",array[i][j]);
      }
      else{
        printf("%i ", array[i][j]);
      }
    printf("\n");
  }
  printf("\n");
}

// Print function used for comparing grid and lastgrid for debugging
void fullprint(int** grid, int** lastgrid, int rows, int cols, int gen){
  printf("Generation %i\n", gen);
  printf("LAST GRID\n");
  printarray(lastgrid, rows, cols);
  printf("\n");
  printf("CURRENT GRID\n");
  printarray(grid, rows, cols);
  printf("\n");
  return;
}

// Randomizes the grid with 0s or 1s to create a random initial state
int** genzero(int** array, int rows, int cols){
  srand(42);  // Fixed seed for reproducibility
  //unsigned int seed = 1;
  int i,j;

  for (i = 1; i < rows - 1; i++)
    for (j = 1; j < cols - 1; j++)
      array[i][j] = rand() % 2;
  
  return array;
}

// Generation function that processes the previous grid and creates the new grid based on neighbor values
int** generation(int** grid, int** lastgrid, int rows, int cols, int num_threads){
  int i, j, neighbors;

  // Iterate through the arrays, checking the previous grid and updating values for new grid
  for (i = 1; i < rows - 1; i++){
    for (j = 1; j < cols - 1; j++){
      // Sum all 8 neighboring values to check if cell should live or die
      neighbors = lastgrid[i - 1][j - 1]
                + lastgrid[i - 1][j]
                + lastgrid[i - 1][j + 1]
                + lastgrid[i][j + 1]
                + lastgrid[i][j - 1]
                + lastgrid[i + 1][j]
                + lastgrid[i + 1][j - 1]
                + lastgrid[i + 1][j + 1];
      if (neighbors <= 1){ // Dies of starvation
        grid[i][j] = 0;
      }
      else if (neighbors >= 4){ // Dies of overpopulation
        grid[i][j] = 0;
      }
      else if ((neighbors == 3) && (lastgrid[i][j] == 0)){ // Dead cells is born again
        grid[i][j] = 1;
      }
      else if ((2 <= neighbors) && (neighbors <= 3) && (lastgrid[i][j] == 1)){ // Alive cell remains 1
        grid[i][j] = 1;
      }
      else{ // Every other cell remains 0
        grid[i][j] = 0;
      }
    }
  }

  return grid; // returning the new grid with updated values
}

// Function based on code provided in matmul.c
// Gets the time and is used for benchmarking
double gettime(void) {
  struct timeval tval;

  gettimeofday(&tval, NULL);

  return( (double)tval.tv_sec + (double)tval.tv_usec/1000000.0 );
}

// Function for checking if a grid has stagnated between generations
int checkforchange(int** grid, int** lastgrid, int rows, int cols){
    int i,j;

    // Iterate through arrays and compare lastgrid and current grid cell values
    for (i = 1; i < rows - 1; i++) {
        for (j = 1; j < cols - 1; j++){
            if (grid[i][j] != lastgrid[i][j]){
                return 1; // Find a mismatched cell and returning 1, meaning the grid changed
            }
        }
    }
    return 0; // Iterated through all cells and found no change
}

void outputtofile(char *output_file, int** grid, int rows, int cols){
    FILE *file = fopen(output_file, "w");
    int i,j;
  
    for (i = 1; i < rows - 1; i++) {
        for (j = 1; j < cols - 1; j++)
          if (grid[i][j]){
            fprintf(file, "%i ",grid[i][j]);
          }
          else{
            fprintf(file, "%i ", grid[i][j]);
          }
        fprintf(file, "\n");
      }
}


int main(int argc, char **argv) {

    if (argc != 6) {
        printf("Usage: %s <dimensions (int)> <max_generations (int)> <num_threads (int)> <output directory (string)> <stagnationcheck (boolean 1 or 0)>\n", argv[0]);
        exit(-1);
    }

    // Getting the command line arguments
    // Rows and cols are increased by 2, adding "ghost" cells to the boundaries
    int ROWS        = atoi(argv[1]) + 2;
    int COLS        = atoi(argv[1]) + 2;
    int MAX_GEN     = atoi(argv[2]);
    // Getting number of threads for openmp execution
    int num_threads = atoi(argv[3]);
    // Output file and directory (format output_N_N_gen_threads.txt)
    char output_file[200];
    sprintf(output_file, "%s/output%s_%s_%s.txt", argv[4], argv[1], argv[2], argv[3]);
    // Boolean for turning on and off stagnation check
    int stagnationcheck = atoi(argv[5]);

    if (!((0 <= ROWS) && (ROWS <= 1000000)) || (!((0 <= COLS) && (COLS <= 1000000)))){
        printf("Dimensions must be between 0 and 1,000,000\n");
        exit(-1);
    }

    // Doubles to hold start and end time for benchmarking
    double starttime, endtime;

    // Allocating arrays and temp value for swapping
    int** grid          = allocarray(ROWS, COLS);
    int** lastgrid      = allocarray(ROWS, COLS);
    int** temp;

    // Initing arrays with 0s
    grid        = initarray(grid, ROWS, COLS);
    lastgrid    = initarray(lastgrid, ROWS, COLS);

    // Generating generation 0 with a random layout of 0s and 1s
    grid  = genzero(grid, ROWS, COLS);

    // Generation counter
    int gen = 0;
    
    // Getting start time for benchmarking
    starttime = gettime();

    // Iteration loop for main logic. Endes after required number of generations
    for (gen = 1; gen <= MAX_GEN; gen++){
      // Swapping grid pointers to make current grid the lastgrid
      // Simply assigns current grid as old last rid, as all values will be updated. No need to clear values.
      temp = lastgrid;
      lastgrid = grid;
      grid = temp;

      // Updating grid based on cell values
      // Settings for openMP parallelization for the generation for loop
      grid = generation(grid, lastgrid, ROWS, COLS, num_threads);

      fullprint(grid, lastgrid, ROWS, COLS, gen);

      // Checking for stagnation and breaking loop if grid has not changed
      // Checks stagnationcheck boolean first to ensure function is not run if false
      // Saves some time
      if ((stagnationcheck) && (!checkforchange(grid, lastgrid, ROWS, COLS))){
        printf("Breaking at generation %i\n", gen);
        break;
      }
    }

    // Getting endtime and getting benchmarks
    endtime = gettime();
    printf("Time taken = %lf seconds\n", endtime-starttime);

    outputtofile(output_file, grid, ROWS, COLS);

    // Freeing arrays
    destroyarray(grid, ROWS);
    destroyarray(lastgrid, ROWS);
}

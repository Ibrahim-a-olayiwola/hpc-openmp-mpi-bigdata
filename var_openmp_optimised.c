/*
 * example calc of variance
 * -- fine-grained OpenMP (with private)
 * -- var calcs inline
 *
 * Requirements
 * 1. file contains n, followed by the n datapoints
 *
 * mkbane (Nov 2024)
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
// function prototypes
// -- functions used in relative xxxRoutines.c
int get_num_data_points(FILE*);
int read_data(FILE*, int, double*);  // returns number of points successfully read, populates array with data points
// main routine reading args from command line
int main(int argc, char** argv) {
  int n;                 // number of data points
  double *x;             // pointer to array holding data points
  // time total code (and elements thereof)
  double startTotalCode = omp_get_wtime();
  
  // access file here (and then pass pointer to file). This allows >1 routine to access same file.
  FILE* filePtr;
  char *filename = argv[1]; // filename is 1st parameter on command line
  filePtr = fopen(filename, "r"); // open file, given by sole parameter, as read-only
  if (filePtr == NULL) {
    printf("Cannot open file %s\n", filename);
  }
  else {
    int totalNum = get_num_data_points(filePtr);
    printf("There are allegedly %d data points to read\n", totalNum);
    x = (double *) malloc(totalNum * sizeof(double));
    if (x == NULL) {
      // error in allocating memory
      printf("Error in allocating memory for data points\n");
    }
    else {
      double start_readData = omp_get_wtime();
      n = read_data(filePtr, totalNum, x); // this is actual number of points read
      printf("%d data points successfully read [%f seconds]\n", n, omp_get_wtime()-start_readData);
      if (n != totalNum) printf("*** WARNING ***\n actual number read (%d) differs from header value (%d)\n\n",n,totalNum);
      
      // close file after we're done reading it
      fclose(filePtr);
      
      /*
       * main data processing loop
       *
       */
      printf("x[0]=%f\n", x[0]);
      double sum = 0.0;
      double mean;
      double start = omp_get_wtime();
      double local_start = omp_get_wtime();
#pragma omp parallel for default(none) shared(n, x) reduction(+:sum)
      for (int i=0; i<n; i++) {
        sum += x[i];
      }
      //DEBUG: printf("sum of x values: %f [%f seconds]\n", sum, omp_get_wtime()-local_start);
      mean = sum/(double) n;
      //DEBUG: printf(" with mean: %f\n", mean);
      //DEBUG: local_start = omp_get_wtime();
      
      // calculate squared differences and sum them at the same time
      sum = 0.0; // reset sum for variance calculation
      double val;
#pragma omp parallel for default(none) shared(n, x, mean) private(val) reduction(+:sum)
      for (int i=0; i<n; i++) {
        val = (x[i] - mean);
        sum += val * val; // compute and add to sum in one step
      }
      //DEBUG: printf("sum of squared differences: %f [%f seconds]\n", sum, omp_get_wtime()-local_start);
      
      // minimum and maximum absolute val in one loop
      double minabs = fabs(x[0]);
      double maxabs = fabs(x[0]);
#pragma omp parallel for default(none) shared(n, x) private(val) reduction(min:minabs) reduction(max:maxabs)
      for (int i=0; i<n; i++) {
        val = fabs(x[i]);
        minabs = (val < minabs) ? val : minabs;
        maxabs = (val > maxabs) ? val : maxabs;
      }
      
      // end of all parallel regions
      printf("Total parallel regions time [%f seconds]\n", omp_get_wtime()-start);
      printf("min, max absolute values are: %f, %f\n", minabs, maxabs);
      double variance = sum/(double) n;
      printf("The mean is: %f\n", mean);
      printf("The variance is %f\n", variance);
    } // memory alloc (x)
  } // file open
  
  // free the memory
  free(x);
  
  printf("Completed. [%f seconds]\n", omp_get_wtime()-startTotalCode);
  return 0;
} //main

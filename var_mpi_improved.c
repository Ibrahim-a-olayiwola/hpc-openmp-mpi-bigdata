#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

// Function declarations (defined in autogenRoutines.c)
int get_num_data_points(FILE*);
int read_data(FILE*, int, double*);

int main(int argc, char** argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // ✅ Confirm each rank is running
    printf("Hello from rank %d out of %d\n", rank, size);

    double startTotalCode = MPI_Wtime();

    int totalNum = 0;
    double* x = NULL;

    if (rank == 0) {
        if (argc < 2) {
            printf("Usage: %s <filename>\n", argv[0]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        FILE* file = fopen(argv[1], "r");
        if (!file) {
            printf("Cannot open file %s\n", argv[1]);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        totalNum = get_num_data_points(file);
        printf("There are allegedly %d data points to read\n", totalNum);

        x = malloc(totalNum * sizeof(double));
        if (!x) {
            printf("Error in allocating memory for data points\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        double start_readData = MPI_Wtime();
        int n = read_data(file, totalNum, x);
        fclose(file);

        printf("%d data points successfully read [%f seconds]\n", n, MPI_Wtime() - start_readData);
        if (n != totalNum) {
            printf("*** WARNING *** actual number read (%d) differs from header value (%d)\n", n, totalNum);
        }

        printf("Rank %d: x[0] = %f\n", rank, x[0]);
    }

    MPI_Bcast(&totalNum, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int local_n = totalNum / size + (rank < totalNum % size ? 1 : 0);
    double* local_x = malloc(local_n * sizeof(double));

    int* sendcounts = NULL;
    int* displs = NULL;
    if (rank == 0) {
        sendcounts = malloc(size * sizeof(int));
        displs = malloc(size * sizeof(int));
        int offset = 0;
        for (int i = 0; i < size; i++) {
            sendcounts[i] = totalNum / size + (i < totalNum % size ? 1 : 0);
            displs[i] = offset;
            offset += sendcounts[i];
        }
    }

    MPI_Scatterv(x, sendcounts, displs, MPI_DOUBLE, local_x, local_n, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    double local_sum = 0.0;
    for (int i = 0; i < local_n; i++) {
        local_sum += local_x[i];
    }

    double global_sum;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    double mean = global_sum / totalNum;

    double local_sumsq = 0.0;
    double local_minabs = fabs(local_x[0]);
    double local_maxabs = fabs(local_x[0]);

    for (int i = 0; i < local_n; i++) {
        double diff = local_x[i] - mean;
        local_sumsq += diff * diff;

        double abs_val = fabs(local_x[i]);
        if (abs_val < local_minabs) local_minabs = abs_val;
        if (abs_val > local_maxabs) local_maxabs = abs_val;
    }

    double global_sumsq, global_minabs, global_maxabs;
    MPI_Reduce(&local_sumsq, &global_sumsq, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_minabs, &global_minabs, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_maxabs, &global_maxabs, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double variance = global_sumsq / totalNum;
        printf("min, max absolute values are: %f, %f\n", global_minabs, global_maxabs);
        printf(" with mean: %f\n", mean);
        printf("The variance is %f\n", variance);
        printf("Completed. [%f seconds]\n", MPI_Wtime() - startTotalCode);
    }

    free(local_x);
    if (rank == 0) {
        free(x);
        free(sendcounts);
        free(displs);
    }

    MPI_Finalize();
    return 0;
}

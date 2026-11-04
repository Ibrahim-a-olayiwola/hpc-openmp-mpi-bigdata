# High Performance Computing and Big Data

Two parts: optimising a parallel C program with **OpenMP** and **MPI**, and building **big data** workflows with PySpark and Kafka.

## Part 1: Parallel variance calculation in C

A program computing the mean, variance and min/max absolute values over **250 million data points**, run on a 12-core Linux VM hosted on Azure and compiled with Intel oneAPI (`icx`) at `-O0` as required by the brief.

### OpenMP optimisation

| Change | Why it helped |
|---|---|
| Removed an unnecessary intermediate array (~2GB) | Halved memory use and access overhead |
| Merged separate min/max loops into one | One pass over the data, better cache use, less synchronisation |
| Explicit memory cleanup | Prevents resource leaks |
| Consistent double precision | Avoids floating-point rounding issues |

**Result:** single-core runtime fell from **7.62s to 5.41s**, with a peak speedup of about **1.72x on 12 threads** and more consistent scaling.

### MPI version: debugging AI-generated code

I generated an initial MPI version with GitHub Copilot, then fixed what it got wrong:

- **Linker error:** the generated file redefined functions already in the shared source file, breaking the one-definition rule. I removed the duplicate definitions.
- **Missing validation:** added checks to confirm the data was read correctly, and rank-tagged output to show which process printed what.

The improved MPI version performed better up to 5 or 6 processes, then plateaued as the serial parts (file I/O, global reductions) became the bottleneck.

### Scaling analysis

Using **Amdahl's law** on measured timings, I projected performance to 100 cores: about **1.45s for OpenMP vs 3.59s for MPI**. For this single-node, shared-memory problem, OpenMP is the better choice. MPI would only win when scaling beyond one machine.

Speedup and efficiency were calculated as S(p) = T1 / Tp and E(p) = S(p) / p x 100, using the best of three runs per thread count.

## Part 2: Big data

### Review analysis with PySpark
- Loaded the Kaggle Amazon electronics ratings dataset (millions of reviews) into **PySpark** in Google Colab
- Converted Unix timestamps to dates and filtered to Q2 2014: **654,268 reviews**
- Labelled ratings of 3 and above as High and below 3 as Low: **82.8% High** (541,740) vs 112,528 Low
- Ran a **two-proportion z-test** (statsmodels) on High vs Low: z = 750.43, p < 0.001
- Found the top products by High rating count and plotted the High/Low breakdown with matplotlib

### Streaming pipeline on Azure
- Set up **Kafka** (2.12-3.9.0) and Zookeeper on a persistent Azure Linux VM, checked both services on ports 2181 and 9092, and created a topic called `mock-stream`
- Generated 1,000 mock records with **Mockaroo** (`mockaroo_generated.json`) to stream through the topic
- Converted the JSON array into newline-delimited JSON so each record could be sent as a separate Kafka message
- HDFS integration was attempted but not completed, because of a Hadoop configuration issue in the lab environment

## Project structure

```
var_openmp_optimised.c   my optimised OpenMP version (based on the course-provided var_openmp.c)
var_mpi_initial.c        initial MPI version generated with GitHub Copilot (redefines the data routines, causing the linker error)
var_mpi_improved.c       my corrected MPI version: shared routines declared, not redefined; rank and data checks; read timing
250M.txt                 input header file telling the program to generate 250 million data points

big_data_review_analysis.ipynb   PySpark review analysis, z-test and charts (Part 2)
mockaroo_generated.json          1,000 mock records used to test the Kafka stream (Part 2)
```

`autogenRoutines.c`, which generates the random data points, and the original `var.c` and `var_openmp.c` were supplied by the course, so they are not included in this repository.

## How to run

You will need the course-provided `autogenRoutines.c` in the same folder.

```bash
# OpenMP (Intel oneAPI)
icx -qopenmp -O0 var_openmp_optimised.c autogenRoutines.c -o var_openmp
export OMP_NUM_THREADS=4
./var_openmp 250M.txt

# MPI
mpicc -O0 -o var_mpi var_mpi_improved.c autogenRoutines.c
mpirun -np 4 ./var_mpi 250M.txt
```

The notebook runs in Google Colab: it installs Spark 3.4.4 and downloads the dataset from Kaggle in its first cells.

## Context

Built for the High Performance Computing and Big Data module of my MSc Data Science at Manchester Metropolitan University.

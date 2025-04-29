#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <unistd.h>

#define COST_PER_OPERATION 2000000

const double epsilon = 0.05;

int rank, nprocs;
volatile double tmp = 0.0;

#ifdef COST_PER_OPERATION
void slow_operation()
{
    int i;
    for (i = 0; i < COST_PER_OPERATION; ++ i)
        tmp += 0.1;
}
#endif

void forward_substitution_sequential(int n, double *L, double *b, double *x)
{
    int i, j;

    for (i = 0; i < n; ++ i)
    {
        x[i] = b[i];
        for (j = 0; j < i; ++ j)
        {
            x[i] -= L[i * n + j] * x[j];
            #ifdef COST_PER_OPERATION
            slow_operation();
            #endif
        }
    }
}

void forward_substitution_pipelined(int n, double *L, double *b, double *x)
{
    int i, j;
    double sum;
    int leftProc, rightProc;

    leftProc = (rank - 1 + nprocs) % nprocs; // The previous pipeline stage
    rightProc = (rank + 1) % nprocs; // The next pipeline stage

    for (i = 0; i < n; ++ i)
    {
        if (i % nprocs == rank) // Handle a row of equation in a round-robin way
        {
            sum = b[i];
            for (j = 0; j < i; ++ j)
            {   
                // Receive x[j] from the previous stage
                // The condition check saves communication if x[j] was calculated and stored by the current stage
                if (j % nprocs != rank)
                    MPI_Recv(&x[j], 1, MPI_DOUBLE, leftProc, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                
                // Send x[j] as a relay to the next stage
                // The second condition check saves communication if x[j] was calculated and stored by the next stage
                if (i + 1 < n && j % nprocs != rightProc)
                    MPI_Send(&x[j], 1, MPI_DOUBLE, rightProc, 0, MPI_COMM_WORLD);
                
                sum -= L[i * n + j] * x[j];
                #ifdef COST_PER_OPERATION
                slow_operation();
                #endif
            }
            
            // Send calulcated x[i] to the next 
            // x[i] will be spread to all the following stages 
            if (i + 1 < n)
                MPI_Send(&sum, 1, MPI_DOUBLE, rightProc, 0, MPI_COMM_WORLD);

            x[i] = sum;
        }
    }
}

int check_solution(int n, double *L, double *b, double *x)
{
    int i, j;
    double s;

    for (i = 0; i < n; ++ i)
    {
        s = 0.0;
        for (j = 0; j <= i; ++ j)
            s += L[i * n + j] * x[j];
        if (s > b[i] + epsilon || s < b[i] - epsilon)
            return 0;
    }
    return 1;
}

int main(int argc, char *argv[])
{
    int n, i, j;
    double *L, *b, *xseq, *xpipe;
    double start, end, elapsed;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // Get equation size
    if (argc < 2)
    {
        fprintf(stderr, "./solve <n>\n");
        exit(1);
    }
    n = atoi(argv[1]);

    // Prepare equation
    L = (double *) calloc(n * n, sizeof(double));
    b = (double *) calloc(n, sizeof(double));
    xseq = (double *) malloc(n * sizeof(double));
    xpipe = (double *) malloc(n * sizeof(double));
    for (i = 0; i < n; ++ i)
    {
        b[i] = (i + 1) * (i + 2) / 2.0;
        for (j = 0; j < i; ++ j)
        {
            L[i * n + j] = 1.0;
        }
        L[i * n + i] = 1.0;
    }

    // Sequential
    if (rank == 0)
    {
        start = MPI_Wtime();
        forward_substitution_sequential(n, L, b, xseq);
        end = MPI_Wtime();
        elapsed = 1e3 * (end - start);
        printf("sequential: %6.3lf ms\n", elapsed);
        if (check_solution(n, L, b, xseq))
            printf("sequential: correct!\n");
        else
            printf("sequential: incorrect!\n");
    }

    // Pipelined
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    forward_substitution_pipelined(n, L, b, xpipe);
    end = MPI_Wtime();
    elapsed = 1e3 * (end - start);
    if (rank == (n - 1) % nprocs)
    {
        printf("Pipelined: %6.3lf ms\n", elapsed);
        if (check_solution(n, L, b, xpipe))
            printf("pipelined: correct!\n");
        else
            printf("pipelined: incorrect!\n");
    }

    // Free the allocated
    free(L);
    free(b);
    free(xseq);
    free(xpipe);

    MPI_Finalize();

    return 0;
}
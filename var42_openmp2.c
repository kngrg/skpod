#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

#define Max(a, b) (((a)>(b))?(a):(b))
#define N (2*2*2*2*2*2+2) // 2^6 + 2 = 66

int NUM_THR = 1;
double maxeps = 0.1e-7;
int itmax = 300;
double w = 0.5;
double eps;

double A[N][N][N];
int p[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 40, 60, 80, 100, 120, 140, 160};
double *eps_thrs;

void relax();
void init();
void verify();

int main(int an, char **as) {

    int l;
    for (l = 0; l <= 17; l++) {
        double timeOpenMP_start;
        double timeOpenMP_end;
        NUM_THR = p[l];

        eps_thrs = calloc(p[l], sizeof(double));

        timeOpenMP_start = omp_get_wtime();

        int it;
	
        omp_set_num_threads(p[l]);

        init();

        for (it = 1; it <= itmax; it++) {
            eps = 0.0;
            relax();
            if (eps < maxeps) break;
        }

        verify();

        timeOpenMP_end = omp_get_wtime();

        printf("Threads: %d  N:  %d  Time: %10f   \n\n", p[l], N,
               timeOpenMP_end - timeOpenMP_start);

        free(eps_thrs);
    }
    return 0;
}

void init() {
    int i, j, k;
    for (k = 0; k <= N - 1; k++)
        for (j = 0; j <= N - 1; j++)
            for (i = 0; i <= N - 1; i++) {
                if (i == 0 || i == N - 1 || j == 0 || j == N - 1 || k == 0 || k == N - 1)
                    A[i][j][k] = 0.;
                else
                    A[i][j][k] = (4. + i + j + k);
            }
}


void relax() {
    omp_set_dynamic(0);
#pragma omp parallel shared(A, w, eps_thrs) default(none)
    {
#pragma omp master
        {
            int i, j, k;
            for (k = 1; k <= N - 2; k++)
#pragma omp task default(none) shared(A, w, eps_thrs) private(j, i) firstprivate(k)
            {
                int thr_num = omp_get_thread_num();
                double eps_th = 0.;

                for (j = 1; j <= N - 2; j++)
                    for (i = 1 + (k + j) % 2; i <= N - 2; i += 2) {
                        double b;
                        b = w * ((A[i - 1][j][k] + A[i + 1][j][k] + A[i][j - 1][k] + A[i][j + 1][k]
                                  + A[i][j][k - 1] + A[i][j][k + 1]) / 6. - A[i][j][k]);
                        eps_th = Max(fabs(b), eps_th);
                        A[i][j][k] = A[i][j][k] + b;
                    }
                eps_thrs[thr_num] = Max(eps_th, thr_num);
            }
#pragma omp taskwait
            for (k = 1; k <= N - 2; k++) {
#pragma omp task default(none) shared(A, w) private (j, i) firstprivate(k)
                {
                    for (j = 1; j <= N - 2; j++)
                        for (i = 1 + (k + j + 1) % 2; i <= N - 2; i += 2) {
                            double b;
                            b = w * ((A[i - 1][j][k] + A[i + 1][j][k] + A[i][j - 1][k] + A[i][j + 1][k]
                                      + A[i][j][k - 1] + A[i][j][k + 1]) / 6. - A[i][j][k]);
                            A[i][j][k] = A[i][j][k] + b;
                        }
                }
            }
        }
    }
#pragma omp barrier
    int l;
    for (l = 0; l < NUM_THR; l++) {
        eps = Max(eps_thrs[l], eps);
    }
}


void verify() {
    double s;
    int i, j, k;

    s = 0.;
    for (i = 0; i <= N - 1; i++)
        for (j = 0; j <= N - 1; j++)
            for (k = 0; k <= N - 1; k++) {
                s = s + A[i][j][k] * (i + 1) * (j + 1) * (k + 1) / (N * N * N);
            }
    printf("  S = %f\n", s);

}

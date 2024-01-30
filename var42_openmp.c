#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

#define  Max(a, b) ((a)>(b)?(a):(b))

#define  N   (2*2*2*2*2*2+2) //2^6 + 2 = 66

double maxeps = 0.1e-7;
int itmax = 100;
int i, j, k;
double w = 0.5;
double eps;
double b = 0;

double A[N][N][N];

void relax();

void init();

void verify();

int main(int an, char **as) {

    int p[] = {1, 2, 3, 4, 5, 6, 7, 8, 9,10, 20, 40, 60, 80, 100, 120, 140, 160};
    int i;
    for (i = 0; i <= 17; i++) {
        double timeOpenMP_start;
        double timeOpenMP_end;

        timeOpenMP_start = omp_get_wtime();

        int it;

        omp_set_num_threads(p[i]);

        init();

        for (it = 1; it <= itmax; it++) {
            eps = 0.0;
            relax();
            if (eps < maxeps) break;
        }

        verify();

        timeOpenMP_end = omp_get_wtime();

        printf("Threads: %d  N:  %d  Time: %10f   \n\n", p[i], N,
               timeOpenMP_end - timeOpenMP_start);
    }
    return 0;
}


void init() {
#pragma omp parallel for default(shared) private(i, j, k) collapse(3)
    for (k = 0; k <= N - 1; k++)
        for (j = 0; j <= N - 1; j++)
            for (i = 0; i <= N - 1; i++) {
                if (i == 0 || i == N - 1 || j == 0 || j == N - 1 || k == 0 || k == N - 1) A[i][j][k] = 0.;
                else A[i][j][k] = (4. + i + j + k);
            }
}


void relax() {
#pragma omp parallel for default(shared) private(i, j, k, b) reduction(max:eps) collapse(2)
    for (i = 1; i <= N - 2; i++) {
        for (j = 1; j <= N - 2; j++) {
            for (k = 1 + (i + j) % 2; k <= N - 2; k += 2) {
                b = w * ((A[i - 1][j][k] + A[i + 1][j][k] + A[i][j - 1][k] +
                          A[i][j + 1][k] + A[i][j][k - 1] + A[i][j][k + 1]) /
                         6. - A[i][j][k]);
                eps = Max(fabs(b), eps);
                A[i][j][k] += b;
            }
        }
    }

#pragma omp parallel for default(shared) private(i, j, k, b) collapse(2)
    for (i = 1; i <= N - 2; i++) {
        for (j = 1; j <= N - 2; j++) {
            for (k = 1 + (i + j + 1) % 2; k <= N - 2; k += 2) {
                b = w * ((A[i - 1][j][k] + A[i + 1][j][k] + A[i][j - 1][k] +
                          A[i][j + 1][k] + A[i][j][k - 1] + A[i][j][k + 1]) /
                         6. - A[i][j][k]);
                A[i][j][k] += b;
            }
        }
    }
}


void verify() {
    double s;
    s = 0.;

#pragma omp parallel for default(shared) private(i, j, k) reduction(+:s) collapse(3)
    for (k = 0; k <= N - 1; k++)
        for (j = 0; j <= N - 1; j++)
            for (i = 0; i <= N - 1; i++) {
                s = s + A[i][j][k] * (i + 1) * (j + 1) * (k + 1) / (N * N * N);
            }
    printf("  S = %f\n", s);

}



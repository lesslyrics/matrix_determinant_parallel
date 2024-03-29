#include <iostream>
#include <iomanip>
#include <cmath>
#include <omp.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

void make_sure_matrix_is_invertible(float **pDouble, int size);

using namespace std;

const int MAX_VAL = 5;

// LU decomposition
// a = l * u
// array size = size x size
void lu_decomposition(float **a, float **l, float **u, int size) {

// for loops of lu decomposition are executed in parallel
#pragma omp taskloop shared(a, l, u)
    for (int i = 0; i < size; i++) {
#pragma omp task shared(a, l, u)
        for (int j = 0; j < size; j++) {
            // right-upper part
            if (j < i) {
                l[j][i] = 0;
                continue;
            }
            l[j][i] = a[j][i];
            for (int k = 0; k < i; k++) {
                l[j][i] = l[j][i] - l[j][k] * u[k][i];
            }
        }
#pragma omp taskwait
// rows are processed in separate threads for u-matrix
#pragma omp task shared(a, l, u)
        for (int j = 0; j < size; j++) {
            // left-bottom part
            if (j < i) {
                u[i][j] = 0;
                continue;
            }
            // diagonal is set to 1
            if (j == i) {
                u[i][j] = 1;
                continue;
            }
            u[i][j] = a[i][j] / l[i][i];
            for (int k = 0; k < i; k++) {
                u[i][j] = u[i][j] - ((l[i][k] * u[k][j]) / l[i][i]);
            }

        }
    }
#pragma omp taskwait

}

void init_matrices(float **a, float **l, float **u, int size) {
// initialize matrices in separate threads
#pragma omp for schedule(static)
    for (int i = 0; i < size; ++i) {
        a[i] = new float[size];
        l[i] = new float[size];
        u[i] = new float[size];
    }
}

// fill matrix with random values
void random_fill(float **matrix, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            matrix[i][j] = static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / MAX_VAL));
        }
    }
    make_sure_matrix_is_invertible(matrix, size);

}

void make_sure_matrix_is_invertible(float **matrix, int size) {
    // to make sure that the matrix is invertible, it should be diagonal dominant
    float leftovers_sum = 0;
    int position_on_diag = 0; // column on the diagonal

    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            leftovers_sum += abs(matrix[i][j]);
        }
        // remove the diagonal value
        leftovers_sum -= abs(matrix[i][position_on_diag]);
        // add a random value to the leftovers_sum and place in diagonal position to make sure it is bigger
        matrix[i][position_on_diag] = leftovers_sum + ((rand() % 5) + 1);
        leftovers_sum = 0;
        position_on_diag++;
    }
}

void print_matrix(float **matrix, int size) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            cout << left << setw(9) << setprecision(3) << matrix[i][j] << left << setw(9);
        }
        cout << endl;
    }
}

long double find_det(float **matrix, int size) {
    long double det = 1;
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            if (i == j) {
                det *= matrix[i][j];
            }
        }
        cout << endl;
    }
    return det;
}


int main(int argc, char **argv) {
    double runtime;
    // number of threads to use
    int number_of_threads = 16;
    omp_set_num_threads(number_of_threads);

    // size of matrix
    int size = atoi(argv[1]);

    cout << "Matrix size: " << size << endl;

    // seed for random
    srand(1);

    auto **A = new float *[size];
    auto **L = new float *[size];
    auto **U = new float *[size];

    init_matrices(A, L, U, size);

    // fill A with random values
    random_fill(A, size);

//    cout << "Matrix A: " << endl;
//    print_matrix(A, size);

    runtime = omp_get_wtime();


#pragma omp parallel
    {
#pragma omp single
        {
#pragma omp task
            { lu_decomposition(A, L, U, size); }
        }
    }

//    results for testing purposes
//    cout << "Matrix L: " << endl;
//    print_matrix(L, size);
//    cout << "Matrix U:" << endl;
//    print_matrix(U, size);

    // det(A) = det(L) * det(U), det(U) = 1 => det(A) = det(L)
    long double det = find_det(L, size);
    cout << "Det of LU = " << det << endl;

    // time
    cout << "Runtime result for the size " << size << " and number of threads = " << number_of_threads << ": " <<
         omp_get_wtime() - runtime << endl;

    return 0;
}

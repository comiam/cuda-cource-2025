#ifndef MULTIPLICATION
#define MULTIPLICATION

#include <thread>
#include "utils.h"
#include "eigen_utils.h"

using namespace std;
using namespace Eigen;


template<typename T>
double simple_multiplication(
    vector<vector<T>> A, vector<vector<T>> B, vector<vector<T>> result, int M, int N, int P
){
    auto start_time = chrono::high_resolution_clock::now();
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < P; ++j) {
            double sum = 0.0;
            for (int k = 0; k < N; ++k) {
                sum += A[i][k] * B[k][j];
            }
            result[i][j] = sum;
        }
    }
    chrono::duration<double> elapsed = chrono::high_resolution_clock::now() - start_time;
    return elapsed.count();
}


template<typename T>
double eigen_multiplication(
    vector<vector<T>> A, vector<vector<T>> B, vector<vector<T>> result
){
    MatrixXd A_e = vectorToEigen(A);
    MatrixXd B_e = vectorToEigen(B);

    auto start_time = chrono::high_resolution_clock::now();
    MatrixXd result_e = A_e * B_e;
    chrono::duration<double> elapsed = chrono::high_resolution_clock::now() - start_time;
    
    eigenToVector<T>(result_e, result);

    return elapsed.count();
}


template<typename T>
void partial_multiplication(
    vector<vector<T>>& A, vector<vector<T>>& B, vector<vector<T>>& result, 
    int N, int P, int start_row, int end_row
) {
    for (int i = start_row; i < end_row; ++i) {
        for (int j = 0; j < P; ++j) {
            T sum = 0.0;
            for (int k = 0; k < N; ++k) {
                sum += A[i][k] * B[k][j];
            }
            result[i][j] = sum;
        }
    }
}


template<typename T>
double threaded_multiplication(
    vector<vector<T>> A, vector<vector<T>> B, vector<vector<T>> result, int M, int N, int P, int NUM_THREADS = 8
){
    auto start_time = chrono::high_resolution_clock::now();

    vector<thread> threads(NUM_THREADS);
    int rows_per_thread = M / NUM_THREADS;

    for (int t = 0; t < NUM_THREADS; ++t) {
        int start_row = t * rows_per_thread;
        int end_row = (t == NUM_THREADS - 1) ? M : (start_row + rows_per_thread);
        threads[t] = thread(partial_multiplication<T>, 
            ref(A), ref(B), ref(result),
            N, P, start_row, end_row
        );
    }

    for (auto& th : threads) {
        th.join();
    }

    chrono::duration<double> elapsed = chrono::high_resolution_clock::now() - start_time;
    return elapsed.count();
}

#endif
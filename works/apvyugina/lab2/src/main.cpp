#include <iostream>
#include <vector>
#include <chrono>

#include "multiplication.h"

using namespace std;

// Размер матриц
const int MATRIX_SIZE = 512;


int main() {


    init_random_generator();

    auto A = generateMatrix<double>(MATRIX_SIZE);
    auto B = generateMatrix<double>(MATRIX_SIZE);
    
    
    vector<vector<double>> res_simple(MATRIX_SIZE, vector<double>(MATRIX_SIZE));
    double elapsed = simple_multiplication<double>(A, B, res_simple, MATRIX_SIZE);
    cout << "Простое умножение заняло: " << elapsed << " секунд.\n";
    
    vector<vector<double>> res_eigen(MATRIX_SIZE, vector<double>(MATRIX_SIZE));
    double elapsed2 = eigen_multiplication<double>(A, B, res_eigen, MATRIX_SIZE);
    cout << "Умножение через Eigen заняло: " << elapsed2  << " секунд.\n";

    vector<vector<double>> res_threaded(MATRIX_SIZE, vector<double>(MATRIX_SIZE));
    double elapsed3 = threaded_multiplication<double>(A, B, res_eigen, MATRIX_SIZE);
    cout << "Многопоточное умножение заняло: " << elapsed3 << " секунд.\n";


    bool areEqual = compareMatrices(res_simple, res_threaded);
    cout << "Матрицы равны: " << areEqual << endl;

    return 0;
}
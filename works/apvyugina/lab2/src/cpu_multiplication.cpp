#include <iostream>
#include <vector>
#include <chrono>

#include "multiplication.h"

using namespace std;


int main(int argc, char* argv[]) {

    const int MATRIX_SIZE = stoi(argv[1]);

    init_random_generator();

    auto A = generateMatrix<double>(MATRIX_SIZE);
    auto B = generateMatrix<double>(MATRIX_SIZE);
    
    cout << "CPU" << endl;
    vector<vector<double>> res_simple(MATRIX_SIZE, vector<double>(MATRIX_SIZE));
    double elapsed = simple_multiplication<double>(A, B, res_simple, MATRIX_SIZE);
    cout << "Простое умножение заняло: " << elapsed << " секунд.\n";
    
    vector<vector<double>> res_eigen(MATRIX_SIZE, vector<double>(MATRIX_SIZE));
    double elapsed2 = eigen_multiplication<double>(A, B, res_eigen, MATRIX_SIZE);
    cout << "Умножение через Eigen заняло: " << elapsed2  << " секунд.\n";

    vector<vector<double>> res_threaded(MATRIX_SIZE, vector<double>(MATRIX_SIZE));
    double elapsed3 = threaded_multiplication<double>(A, B, res_eigen, MATRIX_SIZE);
    cout << "Многопоточное умножение заняло: " << elapsed3 << " секунд.\n";

    bool areEqual1 = compareMatrices(res_simple, res_threaded);
    bool areEqual2 = compareMatrices(res_simple, res_eigen);
    bool areEqual = areEqual1 & areEqual2;
    cout << "Матрицы равны: " << areEqual << endl;

    return 0;
}
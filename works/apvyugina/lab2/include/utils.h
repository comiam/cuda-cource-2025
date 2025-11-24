
#ifndef UTILS
#define UTILS

#include <iostream>
#include <vector>
#include <chrono>
#include <random>


using namespace std;

inline void init_random_generator() {
    srand(static_cast<unsigned>(time(NULL)));
}

template<typename T>
vector<vector<T>> generateMatrix(int M, int N){
    vector<vector<T>> matrix(M, vector<T>(N));
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            matrix[i][j] = static_cast<T>(rand()) / RAND_MAX;
        }
    }
    return matrix;
}

template<typename T>
bool compareMatrices(const vector<vector<T>>& a, const vector<vector<T>>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i].size() != b[i].size()) return false;
        for (size_t j = 0; j < a[i].size(); ++j) {
            if (a[i][j] != b[i][j]) return false;
        }
    }
    return true;
}

template<typename T>
vector<T> flattenMatrix(const vector<vector<T>>& mat) {
    size_t rows = mat.size();
    size_t cols = mat.front().size();
    vector<T> flat(rows * cols);
    for(size_t i = 0; i < rows; ++i) {
        memcpy(flat.data() + i * cols, mat[i].data(), cols * sizeof(T));
    }
    return flat;
}

template<typename T>
void printMatrix(const vector<vector<T>>& matrix, int precision = 8) {
    cout.precision(precision);
    for (const auto& row : matrix) {
        for (const auto& value : row) {
            cout << fixed << value << "\t";
        }
        cout << "\n";
    }
}

template<typename T>
void printRow(const vector<T>& matrix, int precision = 4) {
    cout.precision(precision);
    for (const auto& value : matrix) {
        cout << fixed << value << "\t";
    }
    cout << "\n";
}


#endif
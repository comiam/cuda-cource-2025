#ifndef EIGEN_UTILS
#define EIGEN_UTILS

#include <vector>
#include <Eigen/Dense>


using namespace std;


template<typename T>
Eigen::MatrixXd vectorToEigen(const vector<vector<T>>& data) {
    int rows = data.size();
    if (rows == 0) return Eigen::MatrixXd();

    int cols = data[0].size();
    Eigen::MatrixXd result(rows, cols);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            result(i, j) = data[i][j];
        }
    }

    return result;
}


template<typename T>
void eigenToVector(const Eigen::MatrixXd &mat, vector<vector<T>> result) {
    for (int i = 0; i < mat.rows(); ++i) {
        for (int j = 0; j < mat.cols(); ++j) {
            result[i][j] = mat(i, j);
        }
    }
}

#endif

//
// Created by Vladislav on 12.05.2022.
//

#ifndef PARALG_CPP_MYUTILS_HPP
#define PARALG_CPP_MYUTILS_HPP
#include <vector>
#include <algorithm>
#include <numeric>
#include <Eigen/Dense>

using std::vector;
template<typename T>
using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

vector<int> count_indices(int len, int task_cnt);
vector<int> count_displ(int len, int task_cnt);

template<typename T>
bool check_result(const vector<T> &vec, T x0, T result) {
    return (std::accumulate(vec.begin(),  vec.end(), 1, std::multiplies<>()) * x0) == result;
}

template<typename T>
vector<Matrix<T>> generate_matrices(int n, int matrix_dim, vector<T>&& coefficients) {
    vector<Matrix<T>> result(n);
    for(int i = 0; i < n; ++i) {
        auto matrix = Matrix<T>(matrix_dim, matrix_dim);
        auto first_row = matrix.row(0);
        std::copy(coefficients.begin() + (matrix_dim*i), coefficients.begin() + (matrix_dim*(i+1)), first_row.begin());
        auto zeros = matrix.block(1, 0, matrix_dim - 1, matrix_dim).reshaped();
        std::fill(zeros.begin(), zeros.end(), 0);
        matrix(matrix_dim - 1, matrix_dim - 1) = 1;
        if (matrix_dim != 2) {
            auto diag = matrix.block(1, 0, matrix_dim - 2, matrix_dim - 2).diagonal();
            std::fill(diag.begin(), diag.end(), 1);
        }
        result[i] = std::forward<Matrix<T>>(matrix);
    }
    return result;
}

template<typename T>
vector<T> generate_coefficients(int n, int coeff_num, std::function<T (void)>&& gen) {
    vector<T> result(n*coeff_num);
    std::generate(result.begin(), result.end(), gen);
    return result;
}


#endif //PARALG_CPP_MYUTILS_HPP

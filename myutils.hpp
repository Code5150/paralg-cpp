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

// Returns matrix of size matrix_dim x (n*matrix dim)
template<typename T>
Matrix<T> generate_matrix(int n, int matrix_dim, const vector<T>& coefficients) {
    Matrix<T> result(matrix_dim, matrix_dim*n);
    result.setZero();
    std::copy(coefficients.begin(), coefficients.end(), result.row(0).begin());
    for(int i = 1; i < matrix_dim - 1; ++i) {
        for (int j = 0; j < matrix_dim*n; j += matrix_dim) {
            result(i, j + i - 1) = 1;
        }
    }
    for (int j = matrix_dim - 1; j < matrix_dim*n; j += matrix_dim) {
        result(matrix_dim - 1, j) = 1;
    }
    return result;
}

template<typename T>
vector<Matrix<T>> generate_matrices(int n, int matrix_dim, const vector<T>& coefficients) {
    vector<Matrix<T>> result(n, Matrix<T>(matrix_dim, matrix_dim));
    for(int i = 0; i < n; ++i) {
        auto &matrix = result[i];
        auto first_row = matrix.row(0);
        std::copy(coefficients.begin() + (matrix_dim*i), coefficients.begin() + (matrix_dim*(i+1)), first_row.begin());
        auto zeros = matrix.block(1, 0, matrix_dim - 1, matrix_dim).reshaped();
        std::fill(zeros.begin(), zeros.end(), 0);
        matrix(matrix_dim - 1, matrix_dim - 1) = 1;
        if (matrix_dim != 2) {
            auto diag = matrix.block(1, 0, matrix_dim - 2, matrix_dim - 2).diagonal();
            std::fill(diag.begin(), diag.end(), 1);
        }
    }
    return result;
}

template<typename T>
T* generate_matrices(int n, int matrix_dim, const T* coefficients) {
    const size_t matrix_size = matrix_dim*matrix_dim;
    T* result = new T[n*matrix_size];
    std::fill(result, result + n*matrix_size, 0);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < matrix_dim; ++j) {
            result[i*matrix_size + j] = coefficients[i*matrix_dim + j];
        }
        for (size_t j = matrix_dim; j < matrix_size-matrix_dim; j += matrix_dim) {
            const size_t row_step = j + (j / matrix_dim) - 1;
            result[i*matrix_size + row_step] = 1;
        }
        result[(i+1)*matrix_size - 1] = 1;
    }
    return result;
}

template<typename T>
vector<T> generate_coefficients(int n, int coeff_num, std::function<T (void)>&& gen) {
    vector<T> result(n*coeff_num);
    std::generate(result.begin(), result.end(), gen);
    return result;
}

template<typename T>
T* gemm_v1(int M, int N, int K, const T *A, const T *B)
{
    T* C = new T[M*K];
    for (int i = 0; i < M; ++i)
    {
        T* c = C + i * N;
        for (int j = 0; j < N; ++j)
            c[j] = 0;
        for (int k = 0; k < K; ++k)
        {
            const T* b = B + k * N;
            T a = A[i*K + k];
            for (int j = 0; j < N; ++j)
                c[j] += a * b[j];
        }
    }
    return C;
}


#endif //PARALG_CPP_MYUTILS_HPP

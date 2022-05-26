//
// Created by Vladislav on 12.05.2022.
//

#ifndef PARALG_CPP_MYUTILS_HPP
#define PARALG_CPP_MYUTILS_HPP
#include <vector>
#include <algorithm>
#include <numeric>
#include <functional>

using std::vector;

vector<int> count_indices(int len, int task_cnt) {
    vector<int> count(task_cnt, len / task_cnt);
    std::transform(count.begin(), count.begin() + (len % task_cnt), count.begin(), [](int i){ return ++i; });
    return count;
}

vector<int> count_displ(int len, int task_cnt) {
    auto count = count_indices(len, task_cnt);
    vector<int> displ(count.size());
    std::partial_sum(count.begin(), count.end(), displ.begin());
    std::rotate(displ.begin(), displ.end() - 1, displ.end());
    displ[0] = 0;
    return displ;
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
T* gemm(int M, int N, int K, const T *A, const T *B)
{
    T* C = new T[M*K]();
    for (int i = 0; i < M; ++i)
    {
        T* c = C + i * N;
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

template<typename T>
T* gemm_opt(int M, int N, int K, const T *A, const T *B)
{
    T* C = new T[M*K]();
    for (int j = 0; j < N; ++j)
    {
        T* c = C + j * M;
        for (int k = 0; k < K; ++k)
        {
            const T* b = B + k * M;
            T a = A[j*K + k];
            for (int i = 0; i < M; ++i)
                c[i] += a * b[i];
        }
    }
    return C;
}

template<typename T>
T* gemm_opt2(int M, int N, int K, const T *A, const T *B)
{
    T* C = new T[M*K]();
    for (int j = 0; j < N; ++j)
    {
        for (int k = 0; k < K; ++k)
        {
            for (int i = 0; i < M; ++i) {
                C[i * N + j] += A[i * K + k]*B[k*N + j];
            }
        }
    }
    return C;
}

#endif //PARALG_CPP_MYUTILS_HPP

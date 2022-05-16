#include <cstdio>
#include "mpi.h"
#include <vector>
#include "myutils.hpp"
#include <numeric>
#include <Eigen/Dense>
#include <random>
#include <iostream>
#include <chrono>

using std::vector;

template<typename T>
using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

using mnum_t = int;
using hr_clock = std::chrono::high_resolution_clock;

//#define DEBUG

mnum_t sequential_3coef(int n, int coeff_num, const vector<mnum_t> &coefficients, mnum_t x0, mnum_t x_min1) {
    mnum_t x = 0;
    auto x_prev2 = x_min1;
    auto x_prev1 = x0;
    for (int i = 0; i < n*coeff_num; i += coeff_num) {
#ifdef DEBUG
        std::cout << "Coeffs:" << coefficients[i] << ", " << coefficients[i+1] << ", " << coefficients[i+2] << std::endl;
#endif
        x = x_prev2*coefficients[i+1] + coefficients[i+2];
        x = x_prev1*coefficients[i] + x;
        x_prev2 = x_prev1;
        x_prev1 = x;
    }
    return x;
}

int main() {
    int size = 2;
    int matrix_size = 3;
    std::random_device device;
    std::mt19937 gen;
    gen.seed(device());
    std::normal_distribution<> dist{10.0, 2.0};

    int n = 0, x0, x1;
    printf("Enter n:\n");
    scanf_s("%d", &n);
    auto sendbuf = generate_coefficients<mnum_t>(
            n, matrix_size, [&gen, &dist](){ return static_cast<mnum_t>(dist(gen)); }
            );

    printf("Enter x0:\n");
    scanf_s("%d", &x0);
    printf("Enter x1:\n");
    scanf_s("%d", &x1);

    /*Matrix<mnum_t> x0_vec = Matrix<mnum_t>({
            {x0},
            {x1},
            {1}
    });
#ifdef DEBUG
    std::cout << x0_vec << std::endl << std::endl;
#endif*/
    auto count = count_indices(n, size);
    auto displ = count_displ(n, size);

    auto t1 = hr_clock::now();
    auto seq_result = sequential_3coef(n, matrix_size, sendbuf, x0, x1);
    auto t2 = hr_clock::now();
    std::chrono::duration<double, std::milli> ms_double = t2 - t1;
    std::cout << "Seq duration: " << ms_double.count() << "ms\n" << std::endl;
    std::cout << "Result sequential:" << std::endl << seq_result << std::endl;

    auto task_results = vector<mnum_t>(matrix_size*matrix_size*n, 0); // n matrices of size x size
    for (int rank = 0; rank < size; ++rank) {
        auto recvbuf = vector<mnum_t>(matrix_size*matrix_size*count[rank], 0);
        std::copy(
                sendbuf.begin() + displ[rank]*matrix_size,
                sendbuf.begin() + (displ[rank] + count[rank]) * matrix_size,
                recvbuf.begin()
        );
        const size_t row_size = count[rank]*matrix_size;
        for (size_t i = row_size; i < row_size*(matrix_size-1); i += matrix_size) {
            recvbuf[i + (i / row_size) - 1] = 1;
        }
        for (size_t i = row_size*(matrix_size-1); i < row_size*matrix_size; i += matrix_size) {
            recvbuf[i + matrix_size - 1] = 1;
        }

        auto result = vector<mnum_t>(matrix_size*matrix_size);
        for (size_t i = 0; i < matrix_size*matrix_size; ++i) {
            result[i] = recvbuf[row_size - matrix_size + i % matrix_size + row_size*(i / matrix_size)];
        }
        std::cout << "Starting multiplication" << std::endl;
    }
    /*vector<Matrix<mnum_t>> task_results(size);
    t1 = hr_clock::now();
    for (int i = 0; i < size; ++i) {
        auto t_start_op = hr_clock::now();
        auto recvbuf = vector<mnum_t>(
                sendbuf.begin()+displ[i]*matrix_size,  sendbuf.begin()+(displ[i]+count[i])*matrix_size
        );
        auto t_end_op = hr_clock::now();
        std::chrono::duration<double, std::milli> ms_op = t_end_op - t_start_op;
        std::cout << "Recvbuf duration: " << ms_op.count() << "ms\n" << std::endl;
        t_start_op = hr_clock::now();
        auto matrices = generate_matrices(count[i], matrix_size, recvbuf);
        t_end_op = hr_clock::now();
        ms_op = t_end_op - t_start_op;
        std::cout << "Matrix generation duration: " << ms_op.count() << "ms\n" << std::endl;
#ifdef DEBUG
        for(auto &m: matrices) {
            std::cout << "Coeffs:" << m.row(0) << std::endl << std::endl;
        }
#endif
        Matrix<mnum_t> task_result;
        t_start_op = hr_clock::now();
        if (matrices.size() > 1) {
            task_result = std::accumulate(matrices.rbegin()+1,  matrices.rend(), matrices[count[i]-1], std::multiplies());
        } else {
            task_result = matrices[0];
        }
        t_end_op = hr_clock::now();
        ms_op = t_end_op - t_start_op;
        std::cout << "Matrix multiplication duration: " << ms_op.count() << "ms\n" << std::endl;
        task_results[i] = task_result;
    }

#ifdef DEBUG
    for(auto &m: task_results) {
        std::cout << m << std::endl << std::endl;
    }
#endif
    auto result = (std::accumulate(
            task_results.rbegin()+1,  task_results.rend(), task_results[size-1], std::multiplies()
    ) * x0_vec)(0, 0);
    t2 = hr_clock::now();
    ms_double = t2 - t1;

    std::cout << "Matrix duration: " << ms_double.count() << "ms\n" << std::endl;
    std::cout << "Result:" << std::endl << result << std::endl;

    return 0;*/
}

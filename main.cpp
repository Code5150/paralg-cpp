//#define DEBUG_PRINT
//#define NON_PARALLEL
#define PARALLEL_MPI

#include <cstdio>
#include "mpi.h"
#include <vector>
#include "myutils.hpp"
#include <numeric>
#include <random>
#include <chrono>

using std::vector;
using mnum_t = int;
using hr_clock = std::chrono::high_resolution_clock;

mnum_t sequential_3coef(int n, int coeff_num, const mnum_t* coefficients, mnum_t x0, mnum_t x_min1) {
    mnum_t x = 0;
    auto x_prev2 = x_min1;
    auto x_prev1 = x0;
    for (int i = 0; i < n * coeff_num; i += coeff_num) {
        x = x_prev2 * coefficients[i + 1] + coefficients[i + 2];
        x = x_prev1 * coefficients[i] + x;
        x_prev2 = x_prev1;
        x_prev1 = x;
    }
    return x;
}

int main() {
    const int matrix_dim = 3;
    const int matrix_size = matrix_dim * matrix_dim;
#ifdef NON_PARALLEL
    const int size = 2;
    std::random_device device;
    std::mt19937 gen;
    gen.seed(device());
    std::normal_distribution<> dist{10.0, 2.0};

    double seq_time_avg = 0.0;
    double mmul_time_avg = 0.0;

    int n = 0, x0 = 0, x1 = 0;
    printf_s("Enter n:\n");
    scanf_s("%d", &n);
    printf_s("Enter x0:\n");
    scanf_s("%d", &x0);
    printf_s("Enter x1:\n");
    scanf_s("%d", &x1);

    auto x0_vec = new mnum_t[matrix_size]();
    x0_vec[0] = x0;
    x0_vec[3] = x1;
    x0_vec[6] = 1;

    auto sendbuf = generate_coefficients<mnum_t>(
            n, matrix_dim, [&gen, &dist]() { return static_cast<mnum_t>(dist(gen)); }
    );

    auto count = count_indices(n, size);
    auto displ = count_displ(n, size);

    auto t1 = hr_clock::now();
    auto seq_result = sequential_3coef(n, matrix_dim, sendbuf.data(), x0, x1);
    auto t2 = hr_clock::now();
    std::cout << "Result:" << std::endl << seq_result << std::endl;
    std::chrono::duration<double, std::milli> ms_double = t2 - t1;
    seq_time_avg += ms_double.count();

    auto sendbuf_matrices = generate_matrices(n, matrix_dim, sendbuf.data());

    auto task_results = new mnum_t[matrix_size * size];// n matrices of size x size
    t1 = hr_clock::now();
    for (int rank = 0; rank < size; ++rank) {
        const int recvbuf_size = matrix_size * count[rank];
        auto recvbuf = new mnum_t[recvbuf_size];
        std::copy(
                sendbuf_matrices + displ[rank] * matrix_size,
                sendbuf_matrices + (displ[rank] + count[rank]) * matrix_size,
                recvbuf
        );
        auto task_result = new mnum_t[matrix_size];
        std::copy(recvbuf + recvbuf_size - matrix_size, recvbuf + recvbuf_size, task_result);
        if (count[rank] > 1) {
            for (int i = recvbuf_size - 2 * matrix_size; i > -1; i -= matrix_size) {
                task_result = gemm_v1(matrix_dim, matrix_dim, matrix_dim, task_result, recvbuf + i);
            }
        }
        std::copy(task_result, task_result + matrix_size, task_results + rank * matrix_size);
        delete[] recvbuf;
        delete[] task_result;
    }
    auto result = new mnum_t[matrix_size];
    std::copy(task_results + matrix_size * (size - 1), task_results + matrix_size * size, result);
    if (size > 1) {
        for (int i = matrix_size * (size - 2); i > -1; i -= matrix_size) {
            result = gemm_v1(matrix_dim, matrix_dim, matrix_dim, result, task_results + i);
        }
    }
    result = gemm_v1(matrix_dim, matrix_dim, matrix_dim, result, x0_vec);
    t2 = hr_clock::now();
    ms_double = t2 - t1;
    mmul_time_avg += ms_double.count();
    delete[] task_results;
    delete[] result;
    delete[] x0_vec;

    std::cout << "Sequential duration: " << seq_time_avg << "ms" << std::endl;
    std::cout << "Matrix duration: " << mmul_time_avg << "ms\n" << std::endl;
#endif
#ifdef PARALLEL_MPI

    int rank, size;
    MPI_Init(nullptr, nullptr);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = 0, x0 = 0, x1 = 0;
    mnum_t seq_result;
    mnum_t *sendbuf = nullptr;
    mnum_t *x0_vec = nullptr;
    vector<int> count(size);
    vector<int> displ(size);
    if (rank == 0) {
        printf_s("Enter n:\n");
        scanf_s("%d", &n);
        printf_s("Enter x0:\n");
        scanf_s("%d", &x0);
        printf_s("Enter x1:\n");
        scanf_s("%d", &x1);

        x0_vec = new mnum_t[matrix_size]();
        x0_vec[0] = x0;
        x0_vec[3] = x1;
        x0_vec[6] = 1;

        std::random_device device;
        std::mt19937 gen;
        gen.seed(device());
        std::normal_distribution<> dist{10.0, 2.0};

        count = count_indices(n, size);
        std::transform(count.begin(),  count.end(), count.begin(), [](int i){ return i*matrix_size; });
        displ = count_displ(n, size);
        std::transform(displ.begin(),  displ.end(), displ.begin(), [](int i){ return i*matrix_size; });

        auto coefficients = generate_coefficients<mnum_t>(
                n, matrix_dim, [&gen, &dist]() { return static_cast<mnum_t>(dist(gen)); }
        );

        auto t1 = hr_clock::now();
        seq_result = sequential_3coef(n, matrix_dim, coefficients.data(), x0, x1);
        auto t2 = hr_clock::now();
        std::chrono::duration<double, std::milli> ms_double = t2 - t1;
        printf("Result: %d\n", seq_result);
        printf("Time: %.6f ms\n", ms_double.count());
        printf("\n");

        sendbuf = generate_matrices(n, matrix_dim, coefficients.data());
#ifdef DEBUG_PRINT
        for (int i = 0; i < matrix_dim; ++i){
            for (int j = 0; j < n*matrix_dim; ++j) {
                printf("%-2d ", sendbuf[i*n*matrix_dim + j]);
            }
            printf("\n");
        }
        printf("\n");
#endif
    }

    MPI_Bcast(count.data(), size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(displ.data(), size, MPI_INT, 0, MPI_COMM_WORLD);

    const int recvbuf_size = count[rank];
    auto recvbuf = new mnum_t[recvbuf_size]();
    auto task_results = new mnum_t[matrix_size * size]();

    double mmul_start = MPI_Wtime();
    MPI_Scatterv(sendbuf, count.data(), displ.data(), MPI_INT, recvbuf, recvbuf_size, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 1) {
#ifdef DEBUG_PRINT
        for (int i = 0; i < matrix_dim; ++i){
            for (int j = 0; j < n*matrix_dim; ++j) {
                printf("%-2d ", sendbuf[i*n*matrix_dim + j]);
            }
            printf("\n");
        }
        printf("\n");
#endif
    }

    auto task_result = new mnum_t[matrix_size];
    std::copy(recvbuf + recvbuf_size - matrix_size, recvbuf + recvbuf_size, task_result);
    if (count[rank] > 1) {
        for (int i = recvbuf_size - 2 * matrix_size; i > -1; i -= matrix_size) {
            auto tmp_result = gemm(matrix_dim, matrix_dim, matrix_dim, task_result, recvbuf + i);
            auto swap = task_result;
            task_result = tmp_result;
            delete[] swap;
        }
    }

    MPI_Gather(task_result, matrix_size, MPI_INT, task_results, matrix_size, MPI_INT, 0, MPI_COMM_WORLD);

    auto result = new mnum_t[matrix_size];
    if (rank == 0) {
        std::copy(task_results + matrix_size * (size - 1), task_results + matrix_size * size, result);
        if (size > 1) {
            for (int i = matrix_size * (size - 2); i > -1; i -= matrix_size) {
                auto tmp_result = gemm(matrix_dim, matrix_dim, matrix_dim, result, task_results + i);
                auto swap = result;
                result = tmp_result;
                delete[] swap;
            }
        }

        auto tmp_result = gemm(matrix_dim, matrix_dim, matrix_dim, result, x0_vec);
        auto swap = result;
        result = tmp_result;
        delete[] swap;
    }

    double mmul_end = MPI_Wtime();

    delete[] task_result;
    delete[] recvbuf;
    delete[] task_results;
    delete[] sendbuf;
    delete[] x0_vec;

    if (rank == 0) {
        printf("Result: %d\n", result[0]);
        printf("Time: %.6f ms\n", mmul_end - mmul_start);
        printf("\n");
        if (result[0] == seq_result) {
            printf("Results equal\n");
        } else {
            fprintf(stderr, "Results not equal");
        }
    }
    delete[] result;

    MPI_Finalize();
#endif
    return 0;
}

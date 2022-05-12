#include <cstdio>
#include "mpi.h"
#include <vector>
#include "myutils.hpp"
#include <numeric>

using std::vector;

int main() {
    int rank, size;
    MPI_Init(nullptr, nullptr);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int n = 0, x0;
    vector<int> sendbuf;
    vector<int> count(size);
    vector<int> displ(size);
    if (rank == 0) {
        printf("Enter n: ");
        scanf_s("%d", &n);
        printf("Enter x0: ");
        scanf_s("%d", &x0);
        sendbuf = vector<int>(n);
        std::iota(sendbuf.begin(),  sendbuf.end(), 1);
        count = count_indices(n, size);
        displ = count_displ(n, size);
    }

    MPI_Bcast(count.data(), size, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(displ.data(), size, MPI_INT, 0, MPI_COMM_WORLD);

    vector<int> recvbuf(count[rank], 0);
    MPI_Scatterv(sendbuf.data(), count.data(), displ.data(), MPI_INT,
                 recvbuf.data(), count[rank], MPI_INT, 0, MPI_COMM_WORLD);
    vector<int> proc_result(1,
                            std::accumulate(recvbuf.begin(),  recvbuf.end(), 1, std::multiplies<>()));
    printf("The proc_result of process %d: %d\n", rank, proc_result[0]);
    recvbuf = vector<int>(size);
    MPI_Gather(proc_result.data(), 1, MPI_INT, recvbuf.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        auto result = std::accumulate(recvbuf.begin(),  recvbuf.end(), 1, std::multiplies<>()) * x0;
        printf("Result: %d\n", result);
        bool check = check_result(sendbuf, x0, result);
        if (check) {
            printf("Result correct");
        } else {
            fprintf(stderr, "Result incorrect");
        }
    }
    /*auto sendbuf = Matrix<int>(size, 4);
    std::iota(sendbuf.data(), sendbuf.data()+(size*4), 0);

    auto recvbuf = Matrix<int>(1, 4);
    printf("Vector %d before: \n%s\n", rank, to_string(recvbuf).c_str());
    if (rank == 0) {
        printf("Matrix: \n%s\n", to_string(sendbuf).c_str());
    }
    MPI_Scatter(sendbuf.data(), 4, MPI_INT, recvbuf.data(), 4, MPI_INT, 0, MPI_COMM_WORLD);
    printf("Vector %d after: \n%s\n", rank, to_string(recvbuf).c_str());*/
    MPI_Finalize();
    return 0;
}

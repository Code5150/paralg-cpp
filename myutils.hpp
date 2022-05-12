//
// Created by Vladislav on 12.05.2022.
//

#ifndef PARALG_CPP_MYUTILS_HPP
#define PARALG_CPP_MYUTILS_HPP
#include <vector>
#include <algorithm>
#include <numeric>
using std::vector;

vector<int> count_indices(int len, int task_cnt);
vector<int> count_displ(int len, int task_cnt);
template<typename T>
bool check_result(const vector<T> &vec, T x0, T result) {
    return (std::accumulate(vec.begin(),  vec.end(), 1, std::multiplies<>()) * x0) == result;
}

#endif //PARALG_CPP_MYUTILS_HPP

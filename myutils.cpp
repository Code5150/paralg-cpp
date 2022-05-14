//
// Created by Vladislav on 12.05.2022.
//
#include "myutils.hpp"

#include <vector>
#include <algorithm>
#include <numeric>

using namespace std;

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


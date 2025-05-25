#include "mitsuba/core/sparse_matrix.h"
#include <algorithm>

mitsuba::SparseMatrix::SparseMatrix(unsigned *rows, unsigned *cols,
                                    float *values, size_t cnt) {
    for (size_t i = 0; i < cnt; i++) {
        if (values[i] != 0.f) {
            matrix_.push_back({ rows[i], cols[i], values[i] });
        }
    }
}

void mitsuba::SparseMatrix::sum_duplicates() {
    std::sort(matrix_.begin(), matrix_.end());
}
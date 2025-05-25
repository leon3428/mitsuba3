#pragma once

#include <mitsuba/mitsuba.h>
#include <vector>

namespace mitsuba {

class MI_EXPORT_LIB SparseMatrix {
public:
    struct COOValue {
        unsigned row;
        unsigned col;
        float value;

        inline friend bool operator<(const mitsuba::SparseMatrix::COOValue &a,
                                     const mitsuba::SparseMatrix::COOValue &b) {
            if (a.row == b.row) {
                return a.col < b.col;
            }
            return a.row < b.row;
        };
    };

    SparseMatrix() = default;
    SparseMatrix(unsigned *rows, unsigned *cols, float *values, size_t cnt);

    void sum_duplicates();

private:
    std::vector<COOValue> matrix_;
};

} // namespace mitsuba

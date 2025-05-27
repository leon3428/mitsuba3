#pragma once

#include <H5Cpp.h>
#include <algorithm>
#include <chrono>
#include <mitsuba/core/light_transport.h>
#include <mitsuba/mitsuba.h>
#include <mitsuba/render/records.h>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mitsuba {

struct PairHash {
    template <class T1, class T2>
    std::size_t operator()(const std::pair<T1, T2> &p) const {
        size_t seed = 0;
        seed ^= std::hash<unsigned>{}(p.first) + 0x9e3779b9 + (seed << 6) +
                (seed >> 2);
        seed ^= std::hash<unsigned>{}(p.second) + 0x9e3779b9 + (seed << 6) +
                (seed >> 2);
        return seed;
    }
};

template <typename Float, typename Spectrum>
class MI_EXPORT_LIB DatasetGenerator {
public:
    MI_IMPORT_TYPES()

    using IndexArray = typename LightTransport<Float, Spectrum>::IndexArray;
    using ValueArray = typename LightTransport<Float, Spectrum>::ValueArray;
    using COOMatrix =
        std::unordered_map<std::pair<unsigned, unsigned>, float, PairHash>;

    DatasetGenerator(uint32_t rr_depth, bool hide_emitters,
                     std::pair<size_t, size_t> sensor_size,
                     std::pair<size_t, size_t> projector_size)
        : lt_(rr_depth, hide_emitters), sensor_size_(sensor_size),
          projector_size_(projector_size) {}

    void render(mitsuba::Scene<Float, Spectrum> *scene) {
        auto t1 = std::chrono::high_resolution_clock::now();
        auto [rows, cols, values] =
            lt_.render_light_transport(scene, sensor_size_, projector_size_);

        auto cpu_rows   = dr::migrate(std::move(rows), AllocType::Host);
        auto cpu_cols   = dr::migrate(std::move(cols), AllocType::Host);
        auto cpu_values = dr::migrate(std::move(values), AllocType::Host);

        if constexpr (drjit::is_jit_v<Float>) {
            static_assert(std::is_same_v<decltype(cpu_values), Float>);
        } else {
            static_assert(
                std::is_same_v<decltype(cpu_values), dr::Array<float, 4>>);
        }

        auto t2  = std::chrono::high_resolution_clock::now();
        auto mat = process_lt(std::move(cpu_rows), std::move(cpu_cols),
                              std::move(cpu_values));
        auto t3  = std::chrono::high_resolution_clock::now();
        write_mat(std::move(mat));
        auto t4 = std::chrono::high_resolution_clock::now();

        std::cout << "Render time: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(t2 -
                                                                           t1)
                         .count()
                  << '\n';
        std::cout << "Process time: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(t3 -
                                                                           t2)
                         .count()
                  << '\n';
        std::cout << "Write time: "
                  << std::chrono::duration_cast<std::chrono::milliseconds>(t4 -
                                                                           t3)
                         .count()
                  << '\n';
    }

    COOMatrix process_lt(IndexArray &&rows, IndexArray &&cols,
                         ValueArray &&values) {
        auto size = rows.size();
        std::cout << size << '\n';

        const unsigned *rows_array = rows.data();
        const unsigned *cols_array = cols.data();
        const float *values_array  = values.data();

        auto map = COOMatrix();
        map.reserve(size / 2);

        for (size_t i = 0; i < size; ++i) {
            const auto value = values_array[i];
            if (value != 0.f) {
                const auto key = std::make_pair(rows_array[i], cols_array[i]);
                auto [it, inserted] = map.try_emplace(key, value);
                if (!inserted) {
                    it->second += value;
                }
            }
        }

        return map;
    }

    void write_mat(COOMatrix &&mat) {
        std::vector<unsigned> rows;
        std::vector<unsigned> cols;
        std::vector<float> values;

        for (auto &[key, val] : mat) {
            rows.push_back(key.first);
            cols.push_back(key.second);
            values.push_back(val);
        }

        H5::H5File file("ltm.h5", H5F_ACC_TRUNC);
        hsize_t dims[1] = { values.size() };
        H5::DSetCreatPropList plist;
        hsize_t chunk_dims[1] = { values.size() };
        plist.setChunk(1, chunk_dims);
        plist.setShuffle();
        plist.setDeflate(1);
        H5::DataSet rows_dataset = file.createDataSet(
            "rows", H5::PredType::NATIVE_UINT, H5::DataSpace(1, dims), plist);
        rows_dataset.write(rows.data(), H5::PredType::NATIVE_UINT);

        H5::DataSet cols_dataset = file.createDataSet(
            "cols", H5::PredType::NATIVE_UINT, H5::DataSpace(1, dims), plist);
        cols_dataset.write(cols.data(), H5::PredType::NATIVE_UINT);

        H5::DataSet values_dataset =
            file.createDataSet("values", H5::PredType::NATIVE_FLOAT,
                               H5::DataSpace(1, dims), plist);
        values_dataset.write(values.data(), H5::PredType::NATIVE_FLOAT);
    }

private:
    LightTransport<Float, Spectrum> lt_;
    std::pair<size_t, size_t> sensor_size_;
    std::pair<size_t, size_t> projector_size_;
};

} // namespace mitsuba

#pragma once

#include "mitsuba/core/thread_safe_queue.h"
#include <H5Cpp.h>
#include <algorithm>
#include <filesystem>
#include <mitsuba/core/light_transport.h>
#include <mitsuba/mitsuba.h>
#include <mitsuba/render/records.h>
#include <thread>
#include <utility>
#include <vector>

namespace mitsuba {

template <typename Float, typename Spectrum>
class MI_EXPORT_LIB DatasetGenerator {
public:
    MI_IMPORT_TYPES()

    DatasetGenerator(uint32_t rr_depth, bool hide_emitters,
                     size_t sample_count_per_pass,
                     std::pair<size_t, size_t> sensor_size,
                     std::pair<size_t, size_t> projector_size,
                     size_t writer_cnt, float value_threshold = 0.0)
        : lt_(rr_depth, hide_emitters),
          sample_count_per_pass_(sample_count_per_pass),
          sensor_size_(sensor_size), projector_size_(projector_size),
          value_threshold_(value_threshold) {
        for (size_t i = 0; i < writer_cnt; ++i) {
            threads_.emplace_back([this]() { writer_(); });
        }
    }

    ~DatasetGenerator() {
        queue_.done();
        for (auto &th : threads_) {
            th.join();
        }
    }

    void render(std::string const &path,
                mitsuba::Scene<Float, Spectrum> *scene) {
        auto mat = lt_.render_light_transport(scene, sample_count_per_pass_,
                                              sensor_size_, projector_size_,
                                              value_threshold_);

        queue_.push({ path, std::move(mat) });
    }

    void write_mat(std::filesystem::path path, const CooMatrix &mat) {
        H5::H5File file(path, H5F_ACC_TRUNC);
        hsize_t dims[1] = { mat.values.size() };
        H5::DSetCreatPropList plist;
        hsize_t chunk_dims[1] = { mat.values.size() };
        plist.setChunk(1, chunk_dims);
        plist.setShuffle();
        plist.setDeflate(1);
        H5::DataSet rows_dataset = file.createDataSet(
            "rows", H5::PredType::NATIVE_UINT, H5::DataSpace(1, dims), plist);
        rows_dataset.write(mat.rows.data(), H5::PredType::NATIVE_UINT);

        H5::DataSet cols_dataset = file.createDataSet(
            "cols", H5::PredType::NATIVE_UINT, H5::DataSpace(1, dims), plist);
        cols_dataset.write(mat.cols.data(), H5::PredType::NATIVE_UINT);

        H5::DataSet values_dataset =
            file.createDataSet("values", H5::PredType::NATIVE_UINT16,
                               H5::DataSpace(1, dims), plist);
        values_dataset.write(mat.values.data(), H5::PredType::NATIVE_UINT16);
    }

private:
    void writer_() {
        while (auto opt = queue_.wait_and_pop()) {
            write_mat(opt->first, opt->second);
        }
    }

    LightTransport<Float, Spectrum> lt_;
    size_t sample_count_per_pass_;
    std::pair<size_t, size_t> sensor_size_;
    std::pair<size_t, size_t> projector_size_;
    float value_threshold_ = 0.0;

    std::vector<std::thread> threads_;
    ThreadSafeQueue<std::pair<std::filesystem::path, CooMatrix>> queue_;
};

} // namespace mitsuba

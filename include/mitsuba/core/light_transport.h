#pragma once

#include "drjit/array.h"
#include "drjit/array_traits.h"
#include <cassert>
#include <cstddef>
#include <mitsuba/core/light_transport_integrator.h>
#include <mitsuba/render/records.h>
#include <numeric>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace mitsuba {

struct CooMatrix {
    std::vector<unsigned> rows;
    std::vector<unsigned> cols;
    std::vector<uint16_t> values;

    struct Entry {
        unsigned row, col;
        uint16_t value;

        bool operator<(const Entry &other) const {
            if (row != other.row)
                return row < other.row;
            return col < other.col;
        }

        bool operator==(const Entry &other) const {
            return row == other.row && col == other.col;
        }
    };

    void sum_duplicates() {
        if (rows.empty())
            return;

        std::vector<Entry> entries;
        entries.reserve(rows.size());

        // Pack into struct
        for (size_t i = 0; i < rows.size(); ++i) {
            entries.push_back({ rows[i], cols[i], values[i] });
        }

        // Sort by (row, col)
        std::sort(entries.begin(), entries.end());

        size_t ind    = 0;
        Entry current = entries[ind];

        for (size_t i = 1; i < entries.size(); ++i) {
            if (entries[i].row == current.row &&
                entries[i].col == current.col) {
                current.value += entries[i].value;
            } else {
                rows[ind]   = current.row;
                cols[ind]   = current.col;
                values[ind] = current.value;
                current     = entries[i];
                ind++;
            }
        }

        rows[ind]   = current.row;
        cols[ind]   = current.col;
        values[ind] = current.value;

        rows.resize(ind + 1);
        cols.resize(ind + 1);
        values.resize(ind + 1);
    }
};

template <typename Float, typename Spectrum>
class MI_EXPORT_LIB LightTransport {
public:
    MI_IMPORT_TYPES(Scene, Sensor, Sampler, Medium, Emitter, EmitterPtr, BSDF,
                    BSDFPtr)

    constexpr static uint32_t max_depth = 4;
    using IndexArray = std::conditional_t<drjit::is_array_v<UInt32>, UInt32,
                                          dr::Array<unsigned, max_depth>>;
    using ValueArray = std::conditional_t<drjit::is_array_v<Float>, Float,
                                          dr::Array<float, max_depth>>;

    CooMatrix render_light_transport(mitsuba::Scene<Float, Spectrum> *scene,
                                     size_t sample_count_per_pass,
                                     std::pair<size_t, size_t> sensor_size,
                                     std::pair<size_t, size_t> projector_size,
                                     float value_threshold = 0.0) {

        auto sensor       = scene->sensors()[0];
        auto sampler      = sensor->sampler();
        auto sample_count = sampler->sample_count();

        if (sample_count % sample_count_per_pass != 0) {
            throw std::runtime_error("Sample count has to be dividable by the "
                                     "sample count per pass");
        }

        sampler->set_samples_per_wavefront(sample_count_per_pass);
        auto wavefront_size =
            sensor_size.first * sensor_size.second * sample_count_per_pass;
        sampler->seed(UInt32(0), wavefront_size);
        auto diff_scale_factor = dr::rsqrt(static_cast<float>(sample_count));

        auto idx = dr::arange<UInt32>(wavefront_size) / sample_count_per_pass;
        auto pos = Point2i();
        pos.y()  = idx / sensor_size.first;
        pos.x()  = idx - sensor_size.first * pos.y();
        CooMatrix lt;

        auto pass_cnt = sample_count / sample_count_per_pass;
        for (size_t i = 0; i < pass_cnt; ++i) {
            auto [rows, cols, values] = render_light_transport_sample(
                scene, sensor, sampler, pos, diff_scale_factor, sensor_size,
                projector_size, sample_count);

            auto cpu_rows   = dr::migrate(std::move(rows), AllocType::Host);
            auto cpu_cols   = dr::migrate(std::move(cols), AllocType::Host);
            auto cpu_values = dr::migrate(std::move(values), AllocType::Host);

            dr::eval(cpu_rows, cpu_cols, cpu_values);
            dr::sync_thread();

            process_render(cpu_rows, cpu_cols, cpu_values, lt, value_threshold);
            if (pass_cnt > 1) {
                sampler->advance();
                sampler->schedule_state();
            }
        }

        lt.sum_duplicates();
        return lt;
    }

    void process_render(IndexArray const &rows, IndexArray const &cols,
                        ValueArray const &values, CooMatrix &dst,
                        float value_threshold) {
        auto size = rows.size();
        dst.rows.reserve(dst.rows.size() + size);
        dst.cols.reserve(dst.cols.size() + size);
        dst.values.reserve(dst.values.size() + size);

        const unsigned *rows_array = rows.data();
        const unsigned *cols_array = cols.data();
        const float *values_array  = values.data();

        for (size_t i = 0; i < size; ++i) {
            auto value = values_array[i];
            if (value > value_threshold) {
                value = std::min(value, 1.0f);
                auto fixed_precision_value =
                    static_cast<uint16_t>(std::round(value * 65535.0f));
                dst.rows.push_back(rows_array[i]);
                dst.cols.push_back(cols_array[i]);
                dst.values.push_back(fixed_precision_value);
            }
        }
    }

    std::tuple<IndexArray, IndexArray, ValueArray>
    render_light_transport_sample(const Scene *scene, const Sensor *sensor,
                                  Sampler *sampler, const Vector2f &pos,
                                  ScalarFloat diff_scale_factor,
                                  std::pair<size_t, size_t> sensor_size,
                                  std::pair<size_t, size_t> projector_size,
                                  size_t sample_count) {

        auto scale =
            1.0 / ScalarVector2f(sensor_size.first, sensor_size.second);
        auto sample_pos   = pos + sampler->next_2d(true);
        auto adjusted_pos = sample_pos * scale;

        auto aperture_sample = Point2f(0.5);
        if (sensor->needs_aperture_sample()) {
            aperture_sample = sampler->next_2d(true);
        }

        Float time = sensor->shutter_open();
        if (sensor->shutter_open_time() > 0.f) {
            time += sampler->next_1d(true) * sensor->shutter_open_time();
        }

        auto [ray, ray_weight] = sensor->sample_ray_differential(
            time, 0.f, adjusted_pos, aperture_sample);
        if (ray.has_differentials)
            ray.scale_differential(diff_scale_factor);

        auto [values, us, vs] =
            m_integrator.sample(scene, sampler, ray, pos, true);
        values *= ray_weight.x() / Float(sample_count);

        auto mask =
            (values != 0) & (us >= 0) & (us <= 1) & (vs >= 0) & (vs <= 1);

        auto int_us =
            dr::Array<UInt32, max_depth>(us * (projector_size.first - 1));
        auto int_vs = dr::Array<UInt32, max_depth>((1.0 - vs) *
                                                   (projector_size.second - 1));

        auto rows =
            dr::Array<UInt32, max_depth>(pos.y() * sensor_size.first + pos.x());
        auto cols =
            int_vs * static_cast<unsigned>(projector_size.first) + int_us;

        values &= mask;
        rows &= mask;
        cols &= mask;

        dr::eval(values, rows, cols);
        auto v = dr::ravel(values);
        auto r = dr::ravel(rows);
        auto c = dr::ravel(cols);

        dr::sync_thread();

        if constexpr (drjit::is_jit_v<Float>) {
            static_assert(std::is_same_v<decltype(v), Float>);
        } else {
            static_assert(
                std::is_same_v<decltype(v), dr::Array<float, max_depth>>);
        }

        return { r, c, v };
    }

    LightTransport(uint32_t rr_depth, bool hide_emitters)
        : m_integrator(rr_depth, hide_emitters) {}

private:
    LightTransportIntegrator<Float, Spectrum, max_depth> m_integrator;
};

} // namespace mitsuba

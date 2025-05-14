#pragma once

#include "drjit-core/jit.h"
#include "drjit/dynamic.h"
#include "drjit/util.h"
#include "mitsuba/core/vector.h"
#include "mitsuba/render/scene.h"
#include <cstddef>
#include <cstdint>
#include <drjit/array.h>
#include <mitsuba/core/light_transport_integrator.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/ray.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/integrator.h>
#include <mitsuba/render/records.h>
#include <unordered_map>
#include <utility>
#include <vector>

namespace mitsuba {

struct pair_hash {
    template <class T1, class T2>
    std::size_t operator()(const std::pair<T1, T2> &pair) const {
        return std::hash<T1>()(pair.first) ^ std::hash<T2>()(pair.second);
    }
};

template <typename Float, typename Spectrum>
class MI_EXPORT_LIB LightTransport {
public:
    MI_IMPORT_TYPES(Scene, Sampler, Medium, Emitter, EmitterPtr, BSDF, BSDFPtr)

    constexpr static uint32_t max_depth = 4;

    std::tuple<std::vector<float>, std::vector<unsigned>, std::vector<unsigned>>
    render_light_transport(mitsuba::Scene<Float, Spectrum> *scene,
                           std::pair<size_t, size_t> sensor_size,
                           std::pair<size_t, size_t> projector_size) {

        auto sensor       = scene->sensors()[0];
        auto sampler      = sensor->sampler();
        auto sample_count = sampler->sample_count();

        sampler->set_samples_per_wavefront(sample_count);
        auto wavefront_size =
            sensor_size.first * sensor_size.second * sample_count;
        sampler->seed(UInt32(0), wavefront_size);
        auto diff_scale_factor = dr::rsqrt(static_cast<float>(sample_count));

        auto idx = dr::arange<UInt32>(wavefront_size) / sample_count;
        auto pos = Point2i();
        pos.y()  = idx / sensor_size.first;
        pos.x()  = idx - sensor_size.first * pos.y();

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
        values *= ray_weight.x();

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

        auto cpu_values = dr::migrate(dr::ravel(values), AllocType::Host);
        auto cpu_rows   = dr::migrate(dr::ravel(rows), AllocType::Host);
        auto cpu_cols   = dr::migrate(dr::ravel(cols), AllocType::Host);

        dr::sync_thread();

        size_t cnt = dr::width(values) * values.size();
        std::vector<unsigned> norm(sensor_size.first * sensor_size.second, 0);
        for (size_t i = 0; i < cnt; i++) {
            norm[cpu_rows.data()[i]] += 1;
        }

        std::vector<float> v;
        std::vector<unsigned> r;
        std::vector<unsigned> c;

        for (size_t i = 0; i < cnt; i++) {
            if (cpu_values.data()[i] != 0.f) {
                r.push_back(cpu_rows.data()[i]);
                c.push_back(cpu_cols.data()[i]);
                v.push_back(cpu_values.data()[i] /
                            static_cast<float>(norm[cpu_rows.data()[i]]));
            }
        }

        return { v, r, c };
    }

    LightTransport(uint32_t rr_depth, bool hide_emitters)
        : m_integrator(rr_depth, hide_emitters) {}

private:
    LightTransportIntegrator<Float, Spectrum, max_depth> m_integrator;
};

} // namespace mitsuba

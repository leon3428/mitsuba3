#pragma once

#include "drjit-core/jit.h"
#include "drjit/dynamic.h"
#include "mitsuba/core/spectrum.h"
#include <cstddef>
#include <cstdint>
#include <drjit/array.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/ray.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/integrator.h>
#include <mitsuba/render/records.h>
#include <type_traits>

NAMESPACE_BEGIN(mitsuba)

template <typename Float>
using ResultArray = typename std::conditional<dr::is_array_v<Float>, Float,
                                              dr::DynamicArray<Float>>::type;

template <typename Float, typename Spectrum> class MI_EXPORT_LIB LTM {
public:
    MI_IMPORT_TYPES(Scene, Sampler, Medium, Emitter, EmitterPtr, BSDF, BSDFPtr)

    constexpr static uint32_t max_depth = 4;

    LTM(size_t sensor_width, size_t sensor_height, size_t projector_width,
        size_t projector_height, uint32_t rr_depth, bool hide_emitters)
        : m_sensor_width(sensor_width), m_sensor_height(sensor_height),
          m_projector_width(projector_width),
          m_projector_height(projector_height), m_rr_depth(rr_depth),
          m_hide_emitters(hide_emitters) {}

    std::tuple<dr::Array<Float, max_depth>, dr::Array<Float, max_depth>,
               dr::Array<Float, max_depth>, Bool>
    sample(const Scene *scene, Sampler *sampler, const RayDifferential3f &ray_,
           const Point2i &ray_origin_, Bool active) const {
        MI_MASKED_FUNCTION(ProfilerPhase::SamplingIntegratorSample, active);

        if constexpr (dr::is_array_v<Float>) {
            // --------------------- Configure loop state ----------------------

            Ray3f ray           = Ray3f(ray_);
            Spectrum throughput = 1.f;
            Float eta           = 1.f;
            UInt32 depth        = 0;
            // If m_hide_emitters == false, the environment emitter will be
            // visible
            Mask valid_ray =
                !m_hide_emitters && (scene->environment() != nullptr);
            // Variables caching information from the previous bounce
            Interaction3f prev_si = dr::zeros<Interaction3f>();
            Float prev_bsdf_pdf   = 1.f;
            Bool prev_bsdf_delta  = true;
            BSDFContext bsdf_ctx;
            dr::Array<Float, max_depth> values = 0.f;
            dr::Array<Float, max_depth> us     = 0.f;
            dr::Array<Float, max_depth> vs     = 0.f;
            UInt32 ind                         = 0;

            dr::Array<UInt32, max_depth> indices =
                dr::arange<dr::Array<UInt32, max_depth>>(max_depth);

            /* Set up a Dr.Jit loop. This optimizes away to a normal loop in
               scalar mode, and it generates either a a megakernel (default) or
               wavefront-style renderer in JIT variants. This can be controlled
               by passing the '-W' command line flag to the mitsuba binary or
               enabling/disabling the JitFlag.LoopRecord bit in Dr.Jit.
            */
            struct LoopState {
                Ray3f ray;
                Point2i ray_origin;
                Spectrum throughput;
                Float eta;
                UInt32 depth;
                Mask valid_ray;
                Interaction3f prev_si;
                Float prev_bsdf_pdf;
                Bool prev_bsdf_delta;
                Bool active;
                Sampler *sampler;
                dr::Array<Float, max_depth> values;
                dr::Array<Float, max_depth> us = 0.f;
                dr::Array<Float, max_depth> vs = 0.f;
                UInt32 ind;

                DRJIT_STRUCT(LoopState, ray, ray_origin, throughput, eta, depth,
                             valid_ray, prev_si, prev_bsdf_pdf, prev_bsdf_delta,
                             active, sampler, values, us, vs, ind)
            } ls = { ray,
                     ray_origin_,
                     throughput,
                     eta,
                     depth,
                     valid_ray,
                     prev_si,
                     prev_bsdf_pdf,
                     prev_bsdf_delta,
                     active,
                     sampler,
                     values,
                     us,
                     vs,
                     ind };

            // jit_set_flag(JitFlag::LoopRecord, false);
            dr::tie(ls) = dr::while_loop(
                dr::make_tuple(ls),
                [](const LoopState &ls) { return ls.active; },
                [this, scene, bsdf_ctx, indices](LoopState &ls) {
                    /* dr::while_loop implicitly masks all code in the loop
                       using the 'active' flag, so there is no need to pass it
                       to every function */

                    SurfaceInteraction3f si =
                        scene->ray_intersect(ls.ray,
                                             /* ray_flags = */ +RayFlags::All,
                                             /* coherent = */ ls.depth == 0u);

                    // ---------------------- Direct emission
                    // ----------------------

                    /* dr::any_or() checks for active entries in the provided
                       boolean array. JIT/Megakernel modes can't do this test
                       efficiently as each Monte Carlo sample runs
                       independently. In this case, dr::any_or<..>() returns the
                       template argument (true) which means that the 'if'
                       statement is always conservatively taken. */
                    // auto emitters = si.emitter(scene)
                    if (dr::any_or<true>(si.emitter(scene) != nullptr)) {
                        DirectionSample3f ds(scene, si, ls.prev_si);
                        Float em_pdf = 0.f;
                        auto uv      = si.uv;

                        if (dr::any_or<true>(!ls.prev_bsdf_delta))
                            em_pdf = scene->pdf_emitter_direction(
                                ls.prev_si, ds, !ls.prev_bsdf_delta);

                        // Compute MIS weight for emitter sample from previous
                        // bounce
                        Float mis_bsdf = mis_weight(ls.prev_bsdf_pdf, em_pdf);

                        auto value =
                            ls.throughput *
                            ds.emitter->eval(si, ls.prev_bsdf_pdf > 0.f) *
                            mis_bsdf;

                        auto mask = value.x() != 0.f && indices == ls.ind;
                        ls.values += value.x() & mask;
                        ls.us += uv.x() & mask;
                        ls.vs += uv.y() & mask;
                    }

                    // Continue tracing the path at this point?
                    Bool active_next =
                        (ls.depth + 1 < max_depth) && si.is_valid();

                    if (dr::none_or<false>(active_next)) {
                        ls.active = active_next;
                        return; // early exit for scalar mode
                    }

                    BSDFPtr bsdf = si.bsdf(ls.ray);

                    // ---------------------- Emitter sampling
                    // ----------------------

                    // Perform emitter sampling?
                    Mask active_em = active_next &&
                                     has_flag(bsdf->flags(), BSDFFlags::Smooth);

                    DirectionSample3f ds = dr::zeros<DirectionSample3f>();
                    Spectrum em_weight   = dr::zeros<Spectrum>();
                    Vector3f wo          = dr::zeros<Vector3f>();

                    if (dr::any_or<true>(active_em)) {
                        // Sample the emitter
                        std::tie(ds, em_weight) =
                            scene->sample_emitter_direction(
                                si, ls.sampler->next_2d(), true, active_em);
                        active_em &= (ds.pdf != 0.f);

                        /* Given the detached emitter sample, recompute its
                           contribution with AD to enable light source
                           optimization.
                         */
                        if (dr::grad_enabled(si.p)) {
                            ds.d            = dr::normalize(ds.p - si.p);
                            Spectrum em_val = scene->eval_emitter_direction(
                                si, ds, active_em);
                            em_weight =
                                dr::select(ds.pdf != 0, em_val / ds.pdf, 0);
                        }

                        wo = si.to_local(ds.d);
                    }

                    // ------ Evaluate BSDF * cos(theta) and sample direction
                    // -------

                    Float sample_1   = ls.sampler->next_1d();
                    Point2f sample_2 = ls.sampler->next_2d();

                    auto [bsdf_val, bsdf_pdf, bsdf_sample, bsdf_weight] =
                        bsdf->eval_pdf_sample(bsdf_ctx, si, wo, sample_1,
                                              sample_2);

                    // --------------- Emitter sampling contribution
                    // ----------------

                    if (dr::any_or<true>(active_em)) {
                        bsdf_val = si.to_world_mueller(bsdf_val, -wo, si.wi);
                        auto uv  = ds.uv;

                        // Compute the MIS weight
                        Float mis_em = dr::select(ds.delta, 1.f,
                                                  mis_weight(ds.pdf, bsdf_pdf));

                        auto value =
                            ls.throughput * bsdf_val * em_weight * mis_em;

                        auto mask =  active_em && value.x() != 0.f && indices == ls.ind;
                        ls.values += value.x() & mask;
                        ls.us += uv.x() & mask;
                        ls.vs += uv.y() & mask;
                    }

                    // ---------------------- BSDF sampling
                    // ----------------------

                    bsdf_weight = si.to_world_mueller(bsdf_weight,
                                                      -bsdf_sample.wo, si.wi);

                    ls.ray = si.spawn_ray(si.to_world(bsdf_sample.wo));

                    /* When the path tracer is differentiated, we must be
                       careful that the generated Monte Carlo samples are
                       detached (i.e. don't track derivatives) to avoid bias
                       resulting from the combination of moving samples and
                       discontinuous visibility. We need to re-evaluate the BSDF
                       differentiably with the detached sample in that case. */
                    if (dr::grad_enabled(ls.ray)) {
                        ls.ray = dr::detach<true>(ls.ray);

                        // Recompute 'wo' to propagate derivatives to cosine
                        // term
                        Vector3f wo_2 = si.to_local(ls.ray.d);
                        auto [bsdf_val_2, bsdf_pdf_2] =
                            bsdf->eval_pdf(bsdf_ctx, si, wo_2, ls.active);
                        bsdf_weight[bsdf_pdf_2 > 0.f] =
                            bsdf_val_2 / dr::detach(bsdf_pdf_2);
                    }

                    // ------ Update loop variables based on current interaction
                    // ------

                    ls.throughput *= bsdf_weight;
                    ls.eta *= bsdf_sample.eta;
                    ls.valid_ray |=
                        ls.active && si.is_valid() &&
                        !has_flag(bsdf_sample.sampled_type, BSDFFlags::Null);

                    // Information about the current vertex needed by the next
                    // iteration
                    ls.prev_si       = si;
                    ls.prev_bsdf_pdf = bsdf_sample.pdf;
                    ls.prev_bsdf_delta =
                        has_flag(bsdf_sample.sampled_type, BSDFFlags::Delta);

                    // -------------------- Stopping criterion
                    // ---------------------

                    dr::masked(ls.depth, si.is_valid()) += 1;

                    Float throughput_max =
                        dr::max(unpolarized_spectrum(ls.throughput));

                    Float rr_prob =
                        dr::minimum(throughput_max * dr::square(ls.eta), .95f);
                    Mask rr_active   = ls.depth >= m_rr_depth,
                         rr_continue = ls.sampler->next_1d() < rr_prob;

                    /* Differentiable variants of the renderer require the the
                       russian roulette sampling weight to be detached to avoid
                       bias. This is a no-op in non-differentiable variants. */
                    ls.throughput[rr_active] *= dr::rcp(dr::detach(rr_prob));

                    ls.active = active_next && (!rr_active || rr_continue) &&
                                (throughput_max != 0.f);
                    ls.ind += 1;
                });

            // return { /* spec  = */ dr::select(ls.valid_ray, ls.result, 0.f),
            //          /* valid = */ ls.valid_ray };
            return { ls.values, ls.us, ls.vs, ls.valid_ray };
        } else {
            return { 0.f, 0.f, 0.f, false };
        }
    }

    /// Compute a multiple importance sampling weight using the power heuristic
    Float mis_weight(Float pdf_a, Float pdf_b) const {
        pdf_a *= pdf_a;
        pdf_b *= pdf_b;
        Float w = pdf_a / (pdf_a + pdf_b);
        return dr::detach<true>(dr::select(dr::isfinite(w), w, 0.f));
    }

private:
    size_t m_sensor_width;
    size_t m_sensor_height;
    size_t m_projector_width;
    size_t m_projector_height;
    uint32_t m_rr_depth;
    bool m_hide_emitters;
};

NAMESPACE_END(mitsuba)

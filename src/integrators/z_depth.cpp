#include <mitsuba/render/integrator.h>
#include <mitsuba/render/records.h>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _integrator-z-depth:

ZDepth integrator (:monosp:`zdepth`)
----------------------------------

Example of one an extremely simple type of integrator that is also
helpful for debugging: returns the distance from the camera to the closest
intersected object, or 0 if no intersection was found.

.. tabs::
    .. code-tab::  xml
        :name: z-depth-integrator

        <integrator type="zdepth"/>

    .. code-tab:: python

        'type': 'zdepth'

 */

template <typename Float, typename Spectrum>
class ZDepthIntegrator final : public SamplingIntegrator<Float, Spectrum> {
public:
    MI_IMPORT_BASE(SamplingIntegrator)
    MI_IMPORT_TYPES(Scene, Sampler, Medium)

    ZDepthIntegrator(const Properties &props) : Base(props) {}

    std::pair<Spectrum, Mask>
    sample(const Scene *scene, Sampler * /* sampler */,
           const RayDifferential3f &ray, const Medium * /* medium */,
           Float * /* aovs */, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::SamplingIntegratorSample, active);

        PreliminaryIntersection3f pi = scene->ray_intersect_preliminary(
            ray, /* coherent = */ true, active);

        auto intersection_point = pi.t * ray.d;

        return { dr::select(pi.is_valid(), intersection_point.z(), 0.f), pi.is_valid() };
    }

    MI_DECLARE_CLASS()
};

MI_IMPLEMENT_CLASS_VARIANT(ZDepthIntegrator, SamplingIntegrator)
MI_EXPORT_PLUGIN(ZDepthIntegrator, "Z Depth integrator");
NAMESPACE_END(mitsuba)

#include <cstddef>
#include <cstdint>
#include <mitsuba/core/light_transport.h>
#include <mitsuba/core/light_transport_integrator.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/python/python.h>
#include <nanothread/nanothread.h>

#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

MI_PY_EXPORT(ltm) {
    MI_PY_IMPORT_TYPES()

    nb::class_<LightTransportIntegrator<Float, Spectrum, 4>>(m, "LTM",
                                                             "bla bla")
        .def(nb::init<uint32_t, bool>(), "bla", "rr_depth"_a, "hide_emitters"_a)
        .def("sample", &LightTransportIntegrator<Float, Spectrum, 4>::sample);

    nb::class_<LightTransport<Float, Spectrum>>(m, "LightTransport", "bla bla")
        .def(nb::init<uint32_t, bool>(), "bla", "rr_depth"_a, "hide_emitters"_a)
        .def("render_light_transport",
             &LightTransport<Float, Spectrum>::render_light_transport);
}

#undef SET_PROPS

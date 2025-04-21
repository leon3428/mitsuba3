#include <cstddef>
#include <cstdint>
#include <mitsuba/core/ltm.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/python/python.h>
#include <nanothread/nanothread.h>

#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

MI_PY_EXPORT(ltm) {
    MI_PY_IMPORT_TYPES()

    nb::class_<LTM<Float, Spectrum>>(m, "LTM", "bla bla")
        .def(nb::init<size_t, size_t, size_t, size_t, uint32_t, uint32_t,
                      bool>(),
             "bla", "sensor_width"_a, "sensor_height"_a, "projector_width"_a,
             "projector_height"_a, "max_depth"_a, "rr_depth"_a,
             "hide_emitters"_a)
        .def("sample", &LTM<Float, Spectrum>::sample);
}

#undef SET_PROPS

#include <cstddef>
#include <cstdint>
#include <drjit/tensor.h>
#include <mitsuba/core/filesystem.h>
#include <mitsuba/core/fresolver.h>
#include <mitsuba/core/ltm.h>
#include <mitsuba/core/plugin.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/core/transform.h>
#include <mitsuba/python/python.h>
#include <nanothread/nanothread.h>

#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

MI_PY_EXPORT(ltm) {
    MI_PY_IMPORT_TYPES()

    nb::class_<LTM<Float, Spectrum>>(m, "LTM", "bla bla")
        .def(nb::init<size_t, size_t, uint32_t, uint32_t, bool>(), "bla",
             "projector_width"_a, "projector_height"_a, "max_depth"_a,
             "rr_depth"_a, "hide_emitters"_a)
        .def("sample", &LTM<Float, Spectrum>::sample);

    //     m.def(
    //         "add",
    //         [](int a, int b) {
    //             return mitsuba::ltm::add(a, b);
    //         },
    //         "a"_a, "b"_a,
    //         "doc");
}

#undef SET_PROPS
